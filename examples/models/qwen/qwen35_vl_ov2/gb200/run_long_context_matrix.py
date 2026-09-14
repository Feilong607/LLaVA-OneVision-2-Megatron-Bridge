# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-data long-video fit, timing and checkpoint/restart tests on 48 or 64 GPUs."""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any

from analyze_prod_logs import _ANSI, _read_pod
from run_hybridep_ablation import _failed, _kill_group, _wait, _write


logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PHASES = ("reference", "split", "resume")
DATASETS = {2: "mid_training_180s_packed_64k.yaml", 3: "stage3_mix_img10.yaml"}
BUDGETS = {2: 655368, 3: 617482}


@dataclass(frozen=True)
class Experiment:
    """All pods must agree on this contract; each workload uses a fresh root."""

    root: str
    init: str
    world: int
    stage: int = 2
    tps: tuple[int, ...] = (4, 2)
    steps: int = 100
    split: int = 60
    discard: int = 20
    timeout_min: int = 360
    seq: int = 73728
    min_long_tokens: int = 60000

    @property
    def nodes(self) -> int:
        return self.world // 4

    @property
    def gbs(self) -> int:
        return self.world // 2  # 24 / 32; same GBS for TP2 and TP4 on a given world.

    @property
    def schedule(self) -> int:
        return math.ceil(BUDGETS[self.stage] / self.gbs)

    def case(self, tp: int) -> dict[str, int]:
        """Return one explicit attention/expert topology."""
        etp = 2 if self.world == 48 else tp
        dp = self.world // tp
        return dict(world=self.world, tp=tp, etp=etp, ep=8, dp=dp, expert_dp=self.world // (etp * 8))

    def validate(self) -> None:
        """Validate controls without importing torch or contacting GPUs."""
        if self.world not in (48, 64) or self.stage not in DATASETS:
            raise ValueError("LC_WORLD must be 48/64 and LC_STAGE must be 2/3")
        if not self.tps or len(set(self.tps)) != len(self.tps) or not set(self.tps) <= {2, 4}:
            raise ValueError("LC_TPS must be 4,2 (default), 4, 2, or 2,4")
        if (
            not 1 <= self.discard < self.split < self.steps <= 2000
            or self.split < self.steps // 2
            or self.split + self.discard >= self.steps
        ):
            raise ValueError("require DISCARD < SPLIT, STEPS/2 <= SPLIT < STEPS-DISCARD, STEPS <= 2000")
        if not 1 <= self.timeout_min <= 1440:
            raise ValueError("LC_TIMEOUT_MIN must be 1..1440 per phase")
        if self.seq not in (65536, 73728) or not 1 <= self.min_long_tokens <= self.seq:
            raise ValueError("LC_SEQ_LEN must be 65536/73728; MIN_LONG_TOKENS must be 1..SEQ_LEN")
        if not Path(self.root).is_absolute() or any(c.isspace() for c in self.root + self.init):
            raise ValueError("ROOT and INIT paths must be absolute and have no whitespace")
        for tp in self.tps:
            case = self.case(tp)
            if self.world % (case["etp"] * 8) or self.gbs % case["dp"]:
                raise ValueError("invalid expert topology or GBS")

    def folder(self, tp: int, phase: str) -> Path:
        return Path(self.root) / f"tp{tp}" / phase


def dataset_path(e: Experiment) -> Path:
    """Select the same packed mixture as the production stage under review."""
    return REPO / "examples/models/qwen/qwen3_vl_ov2/gb200" / DATASETS[e.stage]


def asset_environment() -> dict[str, str]:
    """Use existing Qwen3.5 asset discovery without the seed85m wrapper preflight."""
    pool = Path(os.environ.get("OV2_STAGE4_POOL", "/datasets/feilong-stage4-datasets"))
    candidates = {
        "OV2_LLM_HF_QWEN35": (
            "config.json",
            [
                pool / "35b/Qwen3.5-35B-A3B-text",
                Path.home() / "Qwen3.5-35B-A3B-text",
                Path("/datasets/llava/11May/Qwen3.5-35B-A3B-text"),
            ],
        ),
        "OV2_HF_PROC_QWEN35_P16M33": (
            "preprocessor_config.json",
            [
                pool / "35b/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model",
                pool / "35b/auto_model",
                Path.home() / "qwen35_p16m33_auto_model",
            ],
        ),
    }
    result = {}
    for name, (marker, paths) in candidates.items():
        if os.environ.get(name):
            paths = [Path(os.environ[name])]
        valid = next((p for p in paths if (p / marker).is_file()), None)
        if valid is None:
            raise FileNotFoundError(f"set {name}: no {marker} in candidate assets")
        result[name] = str(valid)
    model_type = json.loads((Path(result["OV2_LLM_HF_QWEN35"]) / "config.json").read_text()).get("model_type")
    if model_type != "qwen3_5_moe_text":
        raise ValueError("LLM asset must be the Qwen3.5 MoE text extract")
    return result


def source_manifest(e: Experiment, assets: dict[str, str]) -> dict[str, Any]:
    """Capture current code/data/config bytes; record no credentials or weight tensors."""
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh"))
    paths += list((REPO / "src/megatron/bridge/models/qwen_vl_ov2").glob("*.py"))
    paths += list((REPO / "src/megatron/bridge/recipes/ov2").glob("*.py"))
    paths += list((REPO / "3rdparty").glob("*.patch")) + [dataset_path(e)]
    for folder in assets.values():
        paths += list(Path(folder).glob("*.json"))
    return {
        "assets": assets,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
    }


def expected_runtime(e: Experiment, *, tp: int, phase: str) -> dict[str, Any]:
    """The GPU hook compares resolved groups, inner LLM and restored counters."""
    case = e.case(tp)
    step = e.split if phase == "resume" else 0
    return {
        **case,
        "inner_tp": tp,
        "inner_etp": case["etp"],
        "inner_ep": 8,
        "dispatcher": "flex" if tp == 4 else "alltoall",
        "recompute": "selective" if tp == 4 else "full",
        "seq": e.seq,
        "gbs": e.gbs,
        "mbs": 1,
        "step": step,
        "samples": step * e.gbs,
        "finetune": phase != "resume",
        "load_optim": True,
        "load_rng": True,
        "optimizer": "dist_muon",
        "scheduler_from_checkpoint": True,
    }


def phase_environment(e: Experiment, *, tp: int, phase: str, assets: dict[str, str]) -> dict[str, str]:
    """Freeze both reference and restart settings, except their load/save locations."""
    # Keep cluster/network/library plumbing, but never inherit another experiment's tuning.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OV2_", "LC_"))}
    for key in (
        "FLEX_BACKEND",
        "HYBRID_EP_MAX_TOKENS_PER_RANK",
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
        "MOE_CAPACITY_FACTOR",
        "MOE_PAD_TO_CAPACITY",
        "NVTE_FUSED_ATTN",
        "PYTORCH_ALLOC_CONF",
    ):
        env.pop(key, None)
    for key in ("OV2_K8S_NAMESPACE", "OV2_EXTRA_PYLIBS", "OV2_NVSHMEM_LIB"):
        if key in os.environ:
            env[key] = os.environ[key]
    folder = e.folder(tp, phase)
    split = e.folder(tp, "split")
    checkpoint = split / f"iter_{e.split:07d}"
    case = e.case(tp)
    overrides = [
        "rng.seed=1234",
        "checkpoint.finetune=false",
        "checkpoint.load_optim=true",
        "checkpoint.load_rng=true",
        "checkpoint.save_optim=true",
        "checkpoint.save_rng=true",
        "checkpoint.async_save=false",
        "checkpoint.most_recent_k=1",
        "optimizer.muon_split_qkv=false",
        "optimizer.adam_beta2=" + ("0.95" if e.stage == 2 else "0.99"),
        "optimizer.muon_extra_scale_factor=" + ("0.15" if e.stage == 2 else "0.2"),
        "optimizer.weight_decay=" + ("0.01" if e.stage == 2 else "0"),
        "scheduler.start_weight_decay=" + ("0.01" if e.stage == 2 else "0"),
        "scheduler.end_weight_decay=" + ("0.01" if e.stage == 2 else "0"),
        f"scheduler.lr_decay_iters={e.schedule}",
        "scheduler.override_opt_param_scheduler=false",
        "scheduler.use_checkpoint_opt_param_scheduler=true",
        "dataset.shuffle_buffer_size=16",
        "model.pipeline_model_parallel_size=1",
        "model.context_parallel_size=1",
        "model.expert_model_parallel_size=8",
        "logger.log_throughput=true",
        "logger.log_timers_to_tensorboard=true",
        f"train.exit_interval={e.split if phase == 'split' else 'null'}",
    ]
    if phase != "split":
        overrides += ["checkpoint.save=null", "dataset.dataloader_save=null"]
    if phase == "resume":
        overrides += [f"checkpoint.load={split}", f"dataset.dataloader_load={split}"]
    else:
        overrides += ["dataset.dataloader_load=null"]
    env.update(
        REPO=str(REPO),
        RECIPE="ov2_qwen35_35b_a3b_midtrain",
        TP=str(tp),
        NPROC="4",
        ACCEL="2" if tp == 4 else "0",
        OV2_ETP=str(case["etp"]),
        OV2_MIDTRAIN_GBS=str(e.gbs),
        OV2_MIDTRAIN_N_SAMPLES=str(BUDGETS[e.stage]),
        ITERS=str(e.steps),
        OV2_WARMUP_ITERS=str(e.schedule * 2 // 1000),
        OV2_LR="1e-5" if e.stage == 2 else "2e-5",
        OV2_MIN_LR="1e-6",
        OV2_MIDTRAIN_MUON="1",
        OV2_RECOMPUTE_FULL="0" if tp == 4 else "1",
        OV2_RECOMPUTE_MOE="1",
        OV2_VISION_RECOMPUTE="1",
        OV2_LENGTH_SORT_WINDOW="1",
        OV2_NUM_WORKERS="2",
        OV2_PARALLEL_SHARD_ITERS="1",
        OV2_CUDA_MEM_FRACTION="0.88",
        PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8",
        OV2_SEQ_LEN=str(e.seq),
        OV2_MEM_PROBE=str(e.gbs // case["dp"]),
        OV2_PHASE_TIMER=str(e.gbs // case["dp"]),
        OV2_FSDP="0",
        OV2_EP_OVERLAP="0",
        OV2_OPT_OFFLOAD="false",
        DISABLE_RECOMPUTE="0",
        MIXED_PRECISION="bf16_mixed",
        FLEX_BACKEND="hybridep" if tp == 4 else "",
        OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS="auto",
        OV2_HYBRIDEP_CUSTOM_ALLGATHER="1",
        OV2_HYBRIDEP_REQUIRE_NVLINK="1",
        OV2_ALLOW_DATALOADER_FRESH="0",
        OV2_LONG_CONTEXT_PROBE_DIR=str(folder / "evidence"),
        OV2_LONG_CONTEXT_EXPECT=json.dumps(expected_runtime(e, tp=tp, phase=phase)),
        SAVE=str(folder),
        INIT_CKPT=str(checkpoint) if phase == "resume" else e.init,
        SAVE_EVERY="0",
        LOG_EVERY="1",
        DATA_PATH=str(dataset_path(e)),
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        TRITON_CACHE_DIR=f"/tmp/ov2-long-context/{Path(e.root).name}/tp{tp}/triton",
        TORCHINDUCTOR_CACHE_DIR=f"/tmp/ov2-long-context/{Path(e.root).name}/tp{tp}/inductor",
        CUDA_CACHE_PATH=f"/tmp/ov2-long-context/{Path(e.root).name}/tp{tp}/cuda",
        EXTRA_ARGS=" ".join(overrides),
        **assets,
    )
    return env


def validate_checkpoint(e: Experiment, tp: int) -> dict[str, Any]:
    """Check the completed split checkpoint, including counters, config and DP cursors."""
    import torch
    import yaml

    root = e.folder(tp, "split")
    folder = root / f"iter_{e.split:07d}"
    for name in (".metadata", "metadata.json", "train_state.pt", "run_config.yaml"):
        if not (folder / name).is_file():
            raise ValueError(f"incomplete checkpoint: {folder / name}")
    state = torch.load(folder / "train_state.pt", map_location="cpu", weights_only=True)
    if int(state["step"]) != e.split or int(state["consumed_train_samples"]) != e.split * e.gbs:
        raise ValueError("checkpoint train counters differ from split boundary")
    tracker = root / "latest_train_state.pt"
    if tracker.exists():
        step = int(torch.load(tracker, map_location="cpu", weights_only=True)["step"])
    else:
        step = int((root / "latest_checkpointed_iteration.txt").read_text())
    # Energon explicitly reads the text tracker, even when Bridge prefers its .pt tracker.
    if step != e.split or int((root / "latest_checkpointed_iteration.txt").read_text()) != e.split:
        raise ValueError("Bridge/Energon checkpoint trackers disagree")
    dp = e.case(tp)["dp"]
    actual = {p.name for p in folder.glob("train_dataloader_dprank*.pt") if p.stat().st_size > 0}
    if actual != {f"train_dataloader_dprank{r:03d}.pt" for r in range(dp)}:
        raise ValueError("missing/wrong DP dataloader cursors")
    cfg = yaml.safe_load((folder / "run_config.yaml").read_text())
    expected: dict[str, dict[str, Any]] = {
        "model": {
            "tensor_model_parallel_size": tp,
            "expert_tensor_parallel_size": e.case(tp)["etp"],
            "expert_model_parallel_size": 8,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "train": {"global_batch_size": e.gbs, "micro_batch_size": 1, "train_iters": e.steps},
        "checkpoint": {"save_optim": True, "save_rng": True, "ckpt_format": "torch_dist"},
        "optimizer": {"optimizer": "dist_muon", "lr": 1e-5 if e.stage == 2 else 2e-5},
        "scheduler": {"lr_decay_iters": e.schedule, "lr_warmup_iters": e.schedule * 2 // 1000},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError(f"checkpoint {section}.{key} disagrees with experiment")
    return {"checkpoint": str(folder), "step": step, "dp_cursors": dp, "config_checked": True}


def join(e: Experiment, *, node: int, manifest: dict[str, Any]) -> None:
    """Nonce handshake, identical contracts, patch once, and no stale-output reuse."""
    root = Path(e.root)
    control = root / "control"
    token = uuid.uuid4().hex
    if node == 0:
        root.mkdir(parents=True, exist_ok=False)
        control.mkdir()
        _write(root / "experiment.json", asdict(e))
    _wait([root / "experiment.json"], root=root, seconds=300)
    if (root / "complete.json").exists() or list(control.glob("done-*.json")):
        raise ValueError("experiment root already used; choose a new workload name/LC_ROOT")
    if json.loads((root / "experiment.json").read_text()) != json.loads(json.dumps(asdict(e))):
        raise ValueError("master/worker Args differ")
    _write(control / f"join-{node}-{token}.json", {"token": token, "node": node, "manifest": manifest})
    if node == 0:
        deadline = time.monotonic() + 300
        while True:
            _failed(root)
            joins = [list(control.glob(f"join-{n}-*.json")) for n in range(e.nodes)]
            if any(len(paths) > 1 for paths in joins):
                raise RuntimeError("duplicate pod incarnation; use a new workload name")
            if all(len(paths) == 1 for paths in joins):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("not all pods joined")
            time.sleep(1)
        items = [json.loads(paths[0].read_text()) for paths in joins]
        if any(item["manifest"] != manifest for item in items):
            raise ValueError("code/data/asset config differs between pods")
        subprocess.run(["bash", str(REPO / "3rdparty/apply_megatron_patch.sh")], check=True)
        _write(root / "manifest.json", manifest)
        for tp in e.tps:
            for phase in PHASES:
                (e.folder(tp, phase) / "evidence").mkdir(parents=True, exist_ok=False)
        for item in items:
            _write(control / f"ready-{item['node']}-{item['token']}.json", True)
    _wait([control / f"ready-{node}-{token}.json"], root=root, seconds=600)


def run_phase(e: Experiment, *, tp: int, phase: str, node: int, assets: dict[str, str]) -> None:
    """Launch the existing base script, propagate failures, and wait for every pod."""
    root = Path(e.root)
    folder = e.folder(tp, phase)
    if phase == "resume":
        _wait([e.folder(tp, "split") / "checkpoint-check.json"], root=root, seconds=300)
    started = time.monotonic()
    sampled_at = 0.0
    with (folder / f"controller_node{node}.log").open("x") as output:
        process = subprocess.Popen(
            ["bash", str(HERE / "ax_ov2_qwen35_35b_a3b_gb200.sh")],
            env=phase_environment(e, tp=tp, phase=phase, assets=assets),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                _failed(root)
                if time.monotonic() - started > e.timeout_min * 60:
                    raise TimeoutError(f"tp{tp}/{phase} exceeded timeout; inspect {folder}")
                if time.monotonic() - sampled_at >= 10:
                    sample_memory(folder / f"gpu-memory-node{node}.jsonl")
                    sampled_at = time.monotonic()
                time.sleep(2)
            rc = process.returncode
        finally:
            _kill_group(process)
    if rc:
        raise RuntimeError(f"tp{tp}/{phase}/node{node} exited {rc}; inspect {folder}")
    _write(
        root / "control" / f"done-tp{tp}-{phase}-{node}.json", {"rc": rc, "wall_seconds": time.monotonic() - started}
    )
    _wait([root / "control" / f"done-tp{tp}-{phase}-{n}.json" for n in range(e.nodes)], root=root, seconds=600)
    if node == 0:
        if phase == "split":
            _write(folder / "checkpoint-check.json", validate_checkpoint(e, tp))
        _write(folder / "phase-summary.json", summarize_phase(e, tp=tp, phase=phase))
    _wait([folder / "phase-summary.json"], root=root, seconds=600)


def sample_memory(path: Path) -> None:
    """Sample device memory including allocations outside torch; no GPU mutation."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rows = [line.split(",") for line in result.stdout.splitlines()]
        with path.open("a") as stream:
            for uuid_, used, total in rows:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "uuid": uuid_.strip(),
                            "used_mib": float(used),
                            "total_mib": float(total),
                        }
                    )
                    + "\n"
                )
    except (OSError, subprocess.SubprocessError, ValueError):
        # Unavailable telemetry is reported as null, never as zero memory use.
        return


def summarize_phase(e: Experiment, *, tp: int, phase: str) -> dict[str, Any]:
    """Reject missing steps/ranks, nonfinite numerics, silent fresh starts and short-only runs."""
    folder = e.folder(tp, phase)
    first = e.split + 1 if phase == "resume" else 1
    last = e.split if phase == "split" else e.steps
    pod = _read_pod(folder / f"train_node{e.nodes - 1}.log", f"worker-{e.nodes - 2}")
    if pod.regressions or pod.malformed_steps or [s.number for s in pod.steps] != list(range(first, last + 1)):
        raise ValueError(f"{phase}: missing/repeated/restarted iteration records")
    if any(s.total != e.steps or s.batch_size != e.gbs or s.samples != s.number * e.gbs for s in pod.steps):
        raise ValueError("wrong batch/sample accounting")
    metrics: dict[str, list[float]] = {name: [] for name in ("lm loss", "grad norm", "learning rate")}
    texts = [(folder / f"train_node{n}.log").read_text(errors="replace") for n in range(e.nodes)]
    if tp == 4:
        modes = [
            m for text in texts for m in re.findall(r"\[OV2-HYBRID-AG\] rank=(\d+) custom=([01]) domains=(\d+)", text)
        ]
        if {int(r) for r, _, _ in modes} != set(range(e.world)) or any(c != "1" or d != "1" for _, c, d in modes):
            raise ValueError("missing/wrong HybridEP mode or NVLink-domain evidence")
    for raw in texts[-1].splitlines():
        line = _ANSI.sub("", raw)
        if "elapsed time per iteration (ms):" not in line:
            continue
        for name in metrics:
            match = re.search(rf"{re.escape(name)}:\s*([^|\s]+)", line)
            if not match or not math.isfinite(float(match[1])):
                raise ValueError(f"missing/nonfinite {name}")
            metrics[name].append(float(match[1]))
        for name in ("number of skipped iterations", "number of nan iterations"):
            match = re.search(rf"{name}:\s*(\d+)", line)
            if not match or int(match[1]):
                raise ValueError(f"missing/nonzero {name}")
    rows = []
    timed_rows = []
    cutoff = e.split + e.discard if phase == "resume" else e.discard
    for rank in range(e.world):
        evidence = folder / "evidence"
        runtime = json.loads((evidence / f"runtime-rank{rank:03d}.json").read_text())
        wanted = expected_runtime(e, tp=tp, phase=phase)
        if runtime.get("mismatches") or any(runtime.get(k) != v for k, v in wanted.items()):
            raise ValueError(f"runtime mismatch on rank {rank}")
        inputs = read_inputs(e, tp=tp, phase=phase, rank=rank)
        if len(inputs) != (last - first + 1) * (e.gbs // e.case(tp)["dp"]):
            raise ValueError(f"missing input records on rank {rank}")
        if [r["index"] for r in inputs] != list(range(1, len(inputs) + 1)):
            raise ValueError(f"duplicate/out-of-order input records on rank {rank}")
        if rank % tp:
            leader = read_inputs(e, tp=tp, phase=phase, rank=rank - rank % tp)
            if [r["digest"] for r in inputs] != [r["digest"] for r in leader]:
                raise ValueError(f"input mismatch within attention TP group at rank {rank}")
        # Collapse TP replicas for aggregate lengths and token throughput.
        if rank % tp == 0:
            rows.extend(inputs)
            timed_rows.extend(inputs[max(0, cutoff - first + 1) * (e.gbs // e.case(tp)["dp"]) :])
    visual = [r for r in rows if r["patches"] > 0]
    video = [r for r in visual if r["temporal_max"] > 0]
    if not video or max(r["payload_tokens"] for r in video) < e.min_long_tokens:
        raise ValueError("no sufficiently long pack with temporal patches observed; not a long-video validation")
    timed = [s for s in pod.steps if s.number > cutoff]
    if not timed or any(s.seconds <= 0 or not math.isfinite(s.seconds) for s in timed):
        raise ValueError("insufficient finite timing samples after warmup")
    times = sorted(s.seconds for s in timed)
    mean = statistics.mean(times)
    wall = (timed[-1].stamp - timed[0].stamp).total_seconds()
    gpu_samples = [
        json.loads(line) for p in folder.glob("gpu-memory-node*.jsonl") for line in p.read_text().splitlines()
    ]
    return {
        "phase": phase,
        "first": first,
        "last": last,
        "metrics": metrics,
        "timing_count": len(times),
        "mean_seconds": mean,
        "p95_seconds": times[math.ceil(len(times) * 0.95) - 1],
        "samples_per_second": e.gbs / mean,
        "payload_tokens_per_second": sum(r["payload_tokens"] for r in timed_rows) / sum(times),
        "patches_per_second": sum(r["patches"] for r in timed_rows) / sum(times),
        "wall_samples_per_second": (timed[-1].samples - timed[0].samples) / wall if wall > 0 else None,
        "max_visual_pack_tokens": max(r["payload_tokens"] for r in rows if r["patches"] > 0),
        "max_video_pack_tokens": max(r["payload_tokens"] for r in video),
        "long_video_packs": sum(r["payload_tokens"] >= e.min_long_tokens for r in video),
        "long_visual_packs": sum(r["patches"] > 0 and r["payload_tokens"] >= e.min_long_tokens for r in rows),
        "timeout_print_lines": sum(t.count("HYBRID-EP ALLGATHER TIMEOUT") for t in texts),
        "skip_sample_mentions": sum(len(re.findall(r"SkipSample|Skipping sample|exceeds.*seq", t)) for t in texts),
        "peak_allocated_gib": max(
            [float(x) for t in texts for x in re.findall(r"max_allocated=([\d.]+)G", t)], default=None
        ),
        "peak_reserved_gib": max(
            [float(x) for t in texts for x in re.findall(r"max_reserved=([\d.]+)G", t)], default=None
        ),
        "sampled_device_peak_mib": max([r["used_mib"] for r in gpu_samples], default=None),
    }


def read_inputs(e: Experiment, *, tp: int, phase: str, rank: int) -> list[dict[str, Any]]:
    """Read the CPU input evidence for one process incarnation."""
    path = e.folder(tp, phase) / "evidence" / f"inputs-rank{rank:03d}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare(e: Experiment, tp: int) -> dict[str, Any]:
    """Compare continuous versus saved/restarted execution for this topology only."""
    summaries = {p: summarize_phase(e, tp=tp, phase=p) for p in PHASES}
    mismatches = []
    for rank in range(e.world):
        inputs = {p: read_inputs(e, tp=tp, phase=p, rank=rank) for p in PHASES}
        if [r["digest"] for r in inputs["reference"]] != [r["digest"] for r in inputs["split"] + inputs["resume"]]:
            mismatches.append(rank)
    differences = {}
    numerical_pass = True
    for name, rtol, atol in (("learning rate", 1e-6, 1e-12), ("lm loss", 0.01, 0.001), ("grad norm", 0.05, 0.01)):
        a = summaries["reference"]["metrics"][name]
        b = summaries["split"]["metrics"][name] + summaries["resume"]["metrics"][name]
        diff = [abs(x - y) for x, y in zip(a, b)]
        passed = all(d <= atol + rtol * abs(x) for x, d in zip(a, diff))
        numerical_pass &= passed
        differences[name] = {
            "max_abs": max(diff),
            "mean_abs": statistics.mean(diff),
            "first_after_resume_abs": diff[e.split],
            "max_relative_to_reference": max(d / max(abs(x), 1e-12) for x, d in zip(a, diff)),
            "rtol": rtol,
            "atol": atol,
            "within_screening_tolerance": passed,
        }
    result = {
        "topology": e.case(tp),
        "gbs": e.gbs,
        "summaries": summaries,
        "input_metadata_mismatch_ranks": mismatches,
        "paired_metrics": differences,
        "resume_screen_passed": not mismatches and numerical_pass,
        "full_gradient_or_routing_parity_proven": False,
        "boundary": "CPU hashes exclude pixels; tolerance is a screening gate, not bitwise equivalence. "
        "Resume is graceful checkpoint/relaunch, not kill-during-save fault injection. "
        "TP arms change DP/data partition/dispatcher/recompute; compare complete configurations.",
    }
    logger.info(
        "tp%s mean=%.3fs samples/s=%.3f resume_screen=%s input_mismatch_ranks=%s",
        tp,
        summaries["reference"]["mean_seconds"],
        summaries["reference"]["samples_per_second"],
        result["resume_screen_passed"],
        mismatches,
    )
    return result


def main() -> None:
    """Run in each pod; --plan/--report use no GPUs. Report rewrites derived JSON only."""
    logging.basicConfig(level=logging.INFO, format="[qwen35-long] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.report:
        values = json.loads((args.report / "experiment.json").read_text())
        values["tps"] = tuple(values["tps"])
        e = Experiment(**values)
        e.validate()
        for tp in e.tps:
            _write(args.report / f"tp{tp}/comparison.json", compare(e, tp))
        return
    tag = re.sub(r"-(master|worker)-\d+$", "", socket.gethostname())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("invalid workload name")
    world = int(os.environ.get("LC_WORLD", str(int(os.environ.get("PET_NNODES", "0")) * 4)))
    e = Experiment(
        root=str(
            Path(os.environ.get("LC_ROOT", str(Path.home() / "ckpts_video_sft/qwen35_long_context" / tag))).resolve()
        ),
        init=os.environ.get("INIT_CKPT", ""),
        world=world,
        stage=int(os.environ.get("LC_STAGE", "2")),
        tps=tuple(map(int, os.environ.get("LC_TPS", "4,2").split(","))),
        steps=int(os.environ.get("LC_STEPS", "100")),
        split=int(os.environ.get("LC_SPLIT", "60")),
        discard=int(os.environ.get("LC_DISCARD", "20")),
        timeout_min=int(os.environ.get("LC_TIMEOUT_MIN", "360")),
        seq=int(os.environ.get("LC_SEQ_LEN", "73728")),
        min_long_tokens=int(os.environ.get("LC_MIN_LONG_TOKENS", "60000")),
    )
    e.validate()
    if args.plan or os.environ.get("LC_PREFLIGHT_ONLY") == "1":
        logger.info("plan only; no mount/GPU checks: %s", json.dumps(asdict(e)))
        for tp in e.tps:
            logger.info(
                "topology=%s GBS=%s microbatches=%s schedule=%s dataset=%s",
                e.case(tp),
                e.gbs,
                e.gbs // e.case(tp)["dp"],
                e.schedule,
                dataset_path(e),
            )
        return
    if int(os.environ.get("PET_NNODES", "0")) * 4 != e.world:
        raise ValueError("requires PET_NNODES=12/16 with four GPUs per pod")
    node = int(os.environ.get("PET_NODE_RANK", str(int(os.environ.get("RANK", "-4")) // 4)))
    if node not in range(e.nodes):
        raise ValueError("missing or invalid node rank")
    root = Path(e.root)
    if root.exists() and (
        node == 0 or (root / "complete.json").exists() or list((root / "control").glob("done-*.json"))
    ):
        raise FileExistsError("LC_ROOT already exists; choose a new workload name/LC_ROOT")

    def interrupted(signum: int, _frame: object) -> None:
        raise RuntimeError(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if not Path(e.init).is_absolute() or not (Path(e.init) / ".metadata").is_file():
            raise ValueError("INIT_CKPT must be an existing Qwen3.5 iter_* directory with .metadata")
        assets = asset_environment()
        paths = re.findall(r"^\s*-?\s*path:\s*(\S+)", dataset_path(e).read_text(), re.MULTILINE)
        if not paths or any(not (Path(p) / ".nv-meta").is_dir() for p in paths):
            raise FileNotFoundError("stage mixture contains missing dataset/.nv-meta mounts")
        manifest = source_manifest(e, assets)
        join(e, node=node, manifest=manifest)
        for tp in e.tps:
            for phase in PHASES:
                if source_manifest(e, assets) != manifest:
                    raise ValueError("code/data/asset configs changed during this workload")
                logger.info(
                    "starting tp%s/%s; last-rank log=%s/train_node%d.log", tp, phase, e.folder(tp, phase), e.nodes - 1
                )
                run_phase(e, tp=tp, phase=phase, node=node, assets=assets)
            if node == 0:
                _write(root / f"tp{tp}/comparison.json", compare(e, tp))
            _wait([root / f"tp{tp}/comparison.json"], root=root, seconds=600)
        if node == 0:
            _write(
                root / "complete.json", {"tps": e.tps, "note": "inspect each comparison.json for numerical verdict"}
            )
        _wait([root / "complete.json"], root=root, seconds=300)
        logger.info("completed: %s; checkpoints retained for inspection, no production SAVE touched", root)
    except BaseException as exc:
        if (root / "control").is_dir():
            _write(root / "control" / f"failed-{node}.json", {"error": str(exc), "node": node})
        raise


if __name__ == "__main__":
    main()
