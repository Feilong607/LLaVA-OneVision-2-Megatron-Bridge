# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate two fresh midtrain arms on the same four pods; no GPU imports."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from analyze_prod_logs import _ANSI, _read_pod, _statistics


logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MODES = {"custom": "1", "nccl": "0"}
_MODE = re.compile(r"\[OV2-HYBRID-AG\] rank=(\d+) custom=([01]) domains=(\d+)")


@dataclass(frozen=True)
class Experiment:
    """The shared, non-secret experiment contract, identical on all four pods."""

    root: str
    init: str
    steps: int = 400
    discard: int = 100
    order: tuple[str, ...] = ("custom", "nccl")
    timeout_min: int = 240

    def validate(self) -> None:
        """Reject short, unpaired or unsafe-to-interpret experiments before launch."""
        if not 100 <= self.discard < self.steps <= 2000:
            raise ValueError("require 100 <= AB_DISCARD < AB_STEPS <= 2000")
        if set(self.order) != set(MODES) or len(self.order) != 2:
            raise ValueError("AB_ORDER must be custom,nccl or nccl,custom")
        if not 1 <= self.timeout_min <= 1440:
            raise ValueError("AB_TIMEOUT_MIN must be in 1..1440")
        for value in (self.root, self.init):
            if not Path(value).is_absolute() or any(c.isspace() for c in value):
                raise ValueError("experiment and checkpoint paths must be absolute without whitespace")
        if not (Path(self.init) / ".metadata").is_file():
            raise ValueError(f"initial checkpoint .metadata missing: {self.init}")


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _failed(root: Path) -> None:
    failures = sorted((root / "control").glob("failed-*.json"))
    if failures:
        raise RuntimeError(f"a pod failed: {failures[0].read_text().strip()}")


def _wait(paths: list[Path], *, root: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        _failed(root)
        if all(path.is_file() for path in paths):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for {[p.name for p in paths if not p.is_file()]}")
        time.sleep(1)


def _join(experiment: Experiment, *, node: int, repo: Path = REPO) -> None:
    """Fresh per-process handshake: stale ready files cannot release a new pod."""
    root = Path(experiment.root)
    control = root / "control"
    token = uuid.uuid4().hex
    if node == 0:
        root.mkdir(parents=True, exist_ok=False)
        control.mkdir()
        _write(root / "experiment.json", asdict(experiment))
    _wait([root / "experiment.json"], root=root, seconds=300)
    # JSON normalizes tuples to lists for cross-process comparison.
    contract = json.loads(json.dumps(asdict(experiment)))
    if json.loads((root / "experiment.json").read_text()) != contract:
        raise ValueError("master and worker experiment Args differ")
    _write(control / f"join-{node}-{token}.json", {"node": node, "token": token})
    if node == 0:
        deadline = time.monotonic() + 300
        while True:
            _failed(root)
            joins = [sorted(control.glob(f"join-{rank}-*.json")) for rank in range(4)]
            if any(len(paths) > 1 for paths in joins):
                raise RuntimeError("duplicate pod incarnation; use a fresh workload name")
            if all(len(paths) == 1 for paths in joins):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("four pods did not join within 300s")
            time.sleep(1)
        # Apply once before any training interpreter imports mcore. Subsequent
        # existing launchers see the marker and do not race to apply this patch.
        subprocess.run(["bash", str(repo / "3rdparty/apply_megatron_patch.sh")], check=True)
        source = repo / "3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
        if "OV2_HYBRIDEP_CUSTOM_ALLGATHER" not in source.read_text():
            raise RuntimeError("allgather selection patch was not applied")
        for mode in experiment.order:
            (root / mode / "inputs").mkdir(parents=True, exist_ok=False)
        for paths in joins:
            joined = json.loads(paths[0].read_text())
            _write(control / f"ready-{joined['node']}-{joined['token']}.json", True)
    _wait([control / f"ready-{node}-{token}.json"], root=root, seconds=600)


def arm_environment(experiment: Experiment, *, mode: str, node: int) -> dict[str, str]:
    """Reuse production preparation with five microbatches/rank and a long LR schedule."""
    env = dict(os.environ)
    save = Path(experiment.root) / mode
    # Freeze both arms' tuning surface. In particular, no inherited SAVE can
    # cause B to resume A or either arm to load a production midtrain checkpoint.
    env.update(
        REPO=str(REPO),
        TP="1",
        NPROC="4",
        ACCEL="2",
        RECIPE="ov2_qwen35_35b_a3b_midtrain",
        OV2_MIDTRAIN_GBS="80",
        OV2_MIDTRAIN_N_SAMPLES="2666720",
        ITERS=str(experiment.steps),
        OV2_WARMUP_ITERS="66",
        OV2_LR="1e-5",
        OV2_MIN_LR="1e-6",
        OV2_MIDTRAIN_MUON="1",
        OV2_RECOMPUTE_FULL="0",
        OV2_RECOMPUTE_MOE="1",
        OV2_VISION_RECOMPUTE="1",
        OV2_LENGTH_SORT_KEY="patches",
        OV2_LENGTH_SORT_WINDOW="5",
        OV2_NUM_WORKERS="8",
        OV2_CUDA_MEM_FRACTION="0.88",
        PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8",
        OV2_SEQ_LEN="10192",
        OV2_MEM_PROBE="5",
        OV2_PHASE_TIMER="5",
        OV2_FSDP="0",
        OV2_EP_OVERLAP="0",
        OV2_OPT_OFFLOAD="false",
        DISABLE_RECOMPUTE="0",
        MIXED_PRECISION="bf16_mixed",
        FLEX_BACKEND="hybridep",
        OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS="auto",
        OV2_HYBRIDEP_CUSTOM_ALLGATHER=MODES[mode],
        OV2_HYBRIDEP_REQUIRE_NVLINK="1",
        OV2_AB_INPUT_DIR=str(save / "inputs"),
        SAVE=str(save),
        INIT_CKPT=experiment.init,
        SAVE_EVERY="0",
        LOG_EVERY="1",
        OV2_PROPAGATE_WORKER_RC="1",
        DATA_PATH=str(REPO / "examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml"),
        TRITON_CACHE_DIR=f"/tmp/ov2-hep-ab16/{Path(experiment.root).name}/{mode}/triton",
        TORCHINDUCTOR_CACHE_DIR=f"/tmp/ov2-hep-ab16/{Path(experiment.root).name}/{mode}/inductor",
        CUDA_CACHE_PATH=f"/tmp/ov2-hep-ab16/{Path(experiment.root).name}/{mode}/cuda",
        EXTRA_ARGS=(
            "rng.seed=1234 logger.log_throughput=true scheduler.lr_decay_iters=33334 "
            "checkpoint.finetune=true checkpoint.load=null checkpoint.save=null "
            "dataset.dataloader_save=null train.exit_interval=null "
            "model.pipeline_model_parallel_size=1 model.context_parallel_size=1 "
            "model.expert_model_parallel_size=8"
        ),
    )
    env.pop("OV2_PREFLIGHT_ONLY", None)
    # All four local ranks on a pod share its node-local caches; each arm has its own.
    logger.info("node=%d arm=%s save=%s GBS=80 DP=16 mb/rank=5", node, mode, save)
    return env


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Also remove background samplers belonging to this child launcher only.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_arm(experiment: Experiment, *, mode: str, node: int) -> None:
    root = Path(experiment.root)
    log = root / mode / f"controller_node{node}.log"
    env = arm_environment(experiment, mode=mode, node=node)
    started = time.monotonic()
    # A separate process group makes cleanup specific to this experiment. Logs
    # remain visible through SAVE/train_nodeN.log and this per-node controller log.
    with log.open("x") as output:
        process = subprocess.Popen(
            ["bash", str(HERE / "ax_ov2_qwen35_s15_prod32.sh")],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                _failed(root)
                if time.monotonic() - started > experiment.timeout_min * 60:
                    raise TimeoutError(f"arm {mode} exceeded AB_TIMEOUT_MIN; inspect {log}")
                time.sleep(2)
            rc = process.returncode
        finally:
            _kill_group(process)
    _write(root / "control" / f"done-{mode}-{node}.json", {"rc": rc, "wall_seconds": time.monotonic() - started})
    if rc != 0:
        raise RuntimeError(f"arm {mode} node {node} exited {rc}; inspect {log}")
    _wait([root / "control" / f"done-{mode}-{n}.json" for n in range(4)], root=root, seconds=300)
    if node == 0:
        summary = summarize_arm(root / mode, mode=mode, steps=experiment.steps, discard=experiment.discard)
        _write(root / mode / "summary.json", summary)
        logger.info("arm=%s: %s", mode, json.dumps(summary, ensure_ascii=False))
    _wait([root / mode / "summary.json"], root=root, seconds=300)


def summarize_arm(folder: Path, *, mode: str, steps: int, discard: int) -> dict[str, object]:
    """Require a complete run and evidence that all 16 ranks selected the intended path."""
    texts = [(folder / f"train_node{node}.log").read_text(errors="replace") for node in range(4)]
    selected = set()
    for text in texts:
        for rank, custom, domains in _MODE.findall(text):
            if custom != MODES[mode] or domains != "1":
                raise ValueError("backend or NVLink-domain mismatch: comparison invalid")
            selected.add(int(rank))
    if selected != set(range(16)):
        raise ValueError(f"backend selection evidence incomplete: {len(selected)}/16 ranks")
    pod = _read_pod(folder / "train_node3.log", "worker-2")
    if pod.regressions or pod.malformed_steps or [s.number for s in pod.steps] != list(range(1, steps + 1)):
        raise ValueError("missing, repeated or restarted iteration records; comparison invalid")
    if any(s.total != steps or s.batch_size != 80 or s.samples != s.number * 80 for s in pod.steps):
        raise ValueError("unexpected iteration budget or batch/sample accounting")
    if any(not math.isfinite(s.seconds) or s.seconds <= 0 for s in pod.steps):
        raise ValueError("invalid completed-step time")
    for rank in range(16):
        lines = (folder / "inputs" / f"rank{rank:03d}.txt").read_text().splitlines()
        if len(lines) != steps * 5 or any(line.split()[0] != str(i) for i, line in enumerate(lines, 1)):
            raise ValueError(f"input sequence evidence incomplete at rank {rank}")
    timings = _statistics(pod, warmup=discard - 1, window=steps - discard, min_steps=steps - discard)
    metrics: dict[str, list[float]] = {"lm loss": [], "grad norm": []}
    for raw in texts[3].splitlines():
        line = _ANSI.sub("", raw)
        if "elapsed time per iteration (ms):" not in line:
            continue
        for name in metrics:
            match = re.search(rf"{re.escape(name)}:\s*([^|\s]+)", line)
            if match is None:
                raise ValueError(f"missing {name} in iteration log")
            value = float(match[1])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {name}")
            metrics[name].append(value)
    return {
        "mode": mode,
        "timing": asdict(timings),
        "metrics": metrics,
        "timeout_print_lines": sum(t.count("HYBRID-EP ALLGATHER TIMEOUT") for t in texts),
        "stream_warning_lines": sum(sum("AccumulateGrad" in line for line in t.splitlines()) for t in texts),
        "input_metadata_batches_per_rank": steps * 5,
    }


def compare(experiment: Experiment) -> dict[str, object]:
    """Report paired evidence, not proof of tensor equality or a 48-GPU speedup."""
    root = Path(experiment.root)
    results = {
        mode: summarize_arm(root / mode, mode=mode, steps=experiment.steps, discard=experiment.discard)
        for mode in MODES
    }
    mismatches = []
    for rank in range(16):
        hashes = [(root / mode / "inputs" / f"rank{rank:03d}.txt").read_bytes() for mode in MODES]
        if hashes[0] != hashes[1]:
            mismatches.append(rank)
    if mismatches:
        raise ValueError(f"input text/geometry sequence differs at ranks {mismatches}; do not compare speed")
    custom = results["custom"]
    nccl = results["nccl"]
    a, b = custom["timing"], nccl["timing"]
    assert isinstance(a, dict) and isinstance(b, dict)
    speedup = (float(a["mean"]) / float(b["mean"]) - 1) * 100
    logger.info("16-GPU matched window: iter %d-%d", experiment.discard + 1, experiment.steps)
    for mode in MODES:
        item = results[mode]
        timing = item["timing"]
        assert isinstance(timing, dict)
        logger.info(
            "%s mean=%.3fs p95=%.3fs samples/s=%.3f TIMEOUT lines=%s",
            mode,
            timing["mean"],
            timing["p95"],
            timing["samples_per_second"],
            item["timeout_print_lines"],
        )
    logger.info("NCCL relative throughput change: %+.2f%% (16 GPUs only)", speedup)
    deltas = {}
    for name in ("lm loss", "grad norm"):
        ma, mb = custom["metrics"], nccl["metrics"]
        assert isinstance(ma, dict) and isinstance(mb, dict)
        differences = [abs(x - y) for x, y in zip(ma[name], mb[name])]
        deltas[name] = {
            "first_step_abs": differences[0],
            "first10_mean_abs": statistics.mean(differences[:10]),
            "max_abs": max(differences),
        }
        logger.info("%s paired differences: %s", name, deltas[name])
    if not custom["timeout_print_lines"]:
        logger.warning(
            "custom arm did not reproduce TIMEOUT: this does not establish that the original stall is fixed"
        )
    logger.warning(
        "Metadata hashes exclude image pixels. Loss/grad-norm agreement is not full gradient/routing parity."
    )
    logger.warning("Same node-local mb load; GBS80 and 2 EP groups differ from production GBS240 and 6 groups.")
    return {
        "experiment": asdict(experiment),
        "arms": results,
        "nccl_throughput_change_pct": speedup,
        "paired_metric_differences": deltas,
        "metadata_order_matches": True,
        "gpu_parity_proven": False,
    }


def main() -> None:
    """Run inside each pod, or inspect an already completed experiment with --report."""
    logging.basicConfig(level=logging.INFO, format="[hep-ab16] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="read a completed experiment root without launching")
    args = parser.parse_args()
    if args.report:
        values = json.loads((args.report / "experiment.json").read_text())
        values["order"] = tuple(values["order"])
        result = compare(Experiment(**values))
        _write(args.report / "comparison.json", result)
        return
    host = socket.gethostname()
    tag = re.sub(r"-(master|worker)-\d+$", "", host)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", tag):
        raise ValueError("invalid workload hostname")
    root = Path(os.environ.get("AB_ROOT", str(Path.home() / "ckpts_video_sft/qwen35_hep_ab16" / tag))).resolve()
    pool = os.environ.get("OV2_STAGE4_POOL", "/datasets/feilong-stage4-datasets")
    experiment = Experiment(
        root=str(root),
        init=os.environ.get("INIT_CKPT", f"{pool}/35b/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006000"),
        steps=int(os.environ.get("AB_STEPS", "400")),
        discard=int(os.environ.get("AB_DISCARD", "100")),
        order=tuple(os.environ.get("AB_ORDER", "custom,nccl").split(",")),
        timeout_min=int(os.environ.get("AB_TIMEOUT_MIN", "240")),
    )
    experiment.validate()
    if os.environ.get("AB_PREFLIGHT_ONLY") == "1":
        logger.info("preflight: %s", json.dumps(asdict(experiment)))
        logger.info("no GPU launch; topology, container/backend and asset mounts still need live checks")
        return
    if os.environ.get("PET_NNODES") != "4":
        raise ValueError("requires PET_NNODES=4: Workers=3, 4 GPUs per pod")
    node = int(os.environ.get("PET_NODE_RANK", str(int(os.environ.get("RANK", "0")) // 4)))
    if node not in range(4):
        raise ValueError("invalid node rank")

    # Signals must unwind the child-process cleanup rather than leave training running.
    def interrupted(signum: int, _frame: object) -> None:
        raise RuntimeError(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        _join(experiment, node=node)
        for mode in experiment.order:
            logger.info("starting arm=%s; watch %s/%s/train_node3.log", mode, root, mode)
            _run_arm(experiment, mode=mode, node=node)
        if node == 0:
            _write(root / "comparison.json", compare(experiment))
        _wait([root / "comparison.json"], root=root, seconds=300)
        logger.info("completed: %s/comparison.json", root)
    except BaseException as exc:
        if (root / "control").is_dir():
            _write(root / "control" / f"failed-{node}.json", {"error": str(exc), "node": node})
        raise


if __name__ == "__main__":
    main()
