"""CPU-only coordination, patch and paired-report checks; no distributed GPU init."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[4]
FOLDER = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200"
sys.path.insert(0, str(FOLDER))
import run_hybridep_ablation as ab  # noqa: E402


def _load_inputs():
    path = REPO / "src/megatron/bridge/models/qwen_vl_ov2/ablation_inputs.py"
    spec = importlib.util.spec_from_file_location("ablation_inputs_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inputs = _load_inputs()
pytestmark = pytest.mark.unit


def test_repository_resolution():
    assert ab.REPO == REPO


@pytest.fixture
def experiment(tmp_path):
    initial = tmp_path / "stage2/iter_0006000"
    initial.mkdir(parents=True)
    (initial / ".metadata").touch()
    return ab.Experiment(root=str(tmp_path / "result"), init=str(initial), steps=104, discard=100)


def test_arm_only_mode_output_and_caches_change(experiment, monkeypatch):
    monkeypatch.setenv("SAVE", "/do/not/touch/production")
    monkeypatch.setenv("EXTRA_ARGS", "checkpoint.load=/bad rng.seed=99")
    a = ab.arm_environment(experiment, mode="custom", node=0)
    b = ab.arm_environment(experiment, mode="nccl", node=0)
    differing = {key for key in a if a[key] != b[key]}
    assert differing == {
        "OV2_HYBRIDEP_CUSTOM_ALLGATHER",
        "OV2_AB_INPUT_DIR",
        "SAVE",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
    }
    assert a["OV2_MIDTRAIN_GBS"] == "80" and a["TP"] == "1"
    assert a["ITERS"] == "104" and a["SAVE_EVERY"] == "0"
    assert int(a["OV2_MIDTRAIN_N_SAMPLES"]) // 80 == 33334
    assert "scheduler.lr_decay_iters=33334" in a["EXTRA_ARGS"]
    assert "checkpoint.save=null" in a["EXTRA_ARGS"] and "rng.seed=1234" in a["EXTRA_ARGS"]
    assert a["INIT_CKPT"] == b["INIT_CKPT"] == experiment.init


@pytest.mark.parametrize("changes", [{"steps": 100}, {"discard": 99}, {"order": ("nccl", "nccl")}, {"timeout_min": 0}])
def test_contract_rejects_invalid_controls(experiment, changes):
    values = {**ab.asdict(experiment), **changes}
    with pytest.raises(ValueError):
        ab.Experiment(**values).validate()


def test_metadata_fingerprint_boundary_and_stale_output(tmp_path, monkeypatch):
    batch = {"tokens": torch.tensor([[1, 2]]), "labels": torch.tensor([[2, 3]]), "pixel_values": torch.zeros(2, 4)}
    baseline = inputs.batch_digest(batch)
    assert inputs.batch_digest({**batch, "tokens": torch.tensor([[2, 1]])}) != baseline
    assert inputs.batch_digest({**batch, "pixel_values": torch.zeros(3, 4)}) != baseline
    assert inputs.batch_digest({**batch, "pixel_values": torch.ones(2, 4)}) == baseline
    monkeypatch.setenv("OV2_AB_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    inputs.record_batch(batch)
    inputs.record_batch(batch)
    assert (tmp_path / "rank000.txt").read_text().splitlines() == [f"1 {baseline}", f"2 {baseline}"]
    inputs._COUNTS.clear()
    with pytest.raises(FileExistsError):
        inputs.record_batch(batch)


@pytest.fixture
def patched_core(tmp_path):
    path = "megatron/core/transformer/moe/fused_a2a.py"
    upstream = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=REPO / "3rdparty/Megatron-LM")
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(upstream)
    for name in ("megatron_lm_ov2_hybridep_pad.patch", "megatron_lm_ov2_hybridep_allgather.patch"):
        subprocess.run(
            ["patch", "-p1", "-i", str(REPO / "3rdparty" / name)], cwd=tmp_path, check=True, capture_output=True
        )
    return target


@pytest.mark.parametrize("mode,expected", [(None, None), ("0", False), ("1", True)])
def test_actual_patched_constructor_preserves_default(patched_core, monkeypatch, mode, expected):
    if mode is None:
        monkeypatch.delenv("OV2_HYBRIDEP_CUSTOM_ALLGATHER", raising=False)
    else:
        monkeypatch.setenv("OV2_HYBRIDEP_CUSTOM_ALLGATHER", mode)
    tree = ast.parse(patched_core.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_hybrid_ep_buffer")
    captured = {}

    def buffer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(num_of_nodes=1)

    namespace = {
        "torch": SimpleNamespace(distributed=SimpleNamespace(ProcessGroup=object, get_rank=lambda: 0)),
        "Optional": __import__("typing").Optional,
        "HybridEPBuffer": buffer,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(patched_core), "exec"), namespace)
    namespace["init_hybrid_ep_buffer"](None, 2048, 10240, 32)
    if mode is None:
        assert "enable_custom_allgather" not in captured
    else:
        assert captured["enable_custom_allgather"] is expected


@pytest.fixture
def cluster(tmp_path, experiment):
    fake_repo = tmp_path / "repo"
    patcher = fake_repo / "3rdparty/apply_megatron_patch.sh"
    patcher.parent.mkdir(parents=True)
    patcher.write_text("#!/bin/bash\nexit 0\n")
    core = fake_repo / "3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
    core.parent.mkdir(parents=True)
    core.write_text("OV2_HYBRIDEP_CUSTOM_ALLGATHER\n")
    fake = tmp_path / "launchers"
    fake.mkdir()
    worker = fake / "fake.py"
    worker.write_text("""
import os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
n=int(os.environ['PET_NODE_RANK']); count=int(os.environ['ITERS'])
mode=os.environ['OV2_HYBRIDEP_CUSTOM_ALLGATHER']; save=Path(os.environ['SAVE'])
if os.environ.get('FAIL_NODE') == str(n) and mode == '1':
    sys.exit(7)
time.sleep(0.2)
duration=20 if mode=='1' else 18
with (save/f'train_node{n}.log').open('w') as f:
    for rank in range(n*4,n*4+4):
        f.write(f'[OV2-HYBRID-AG] rank={rank} custom={mode} domains=1\\n')
        (save/'inputs'/f'rank{rank:03d}.txt').write_text(''.join(f'{i} metadata-{rank}-{i}\\n' for i in range(1,count*5+1)))
    if n==3:
        for i in range(1,count+1):
            stamp=datetime(2026,9,6)+timedelta(seconds=duration*i)
            f.write(f'[{stamp}] iteration {i}/{count} | consumed samples: {i*80} | elapsed time per iteration (ms): {duration*1000} | global batch size: 80 | lm loss: 1.25 | grad norm: 200.0 |\\n')
""")
    (fake / "ax_ov2_qwen35_s15_prod32.sh").write_text(f'#!/bin/bash\nexec "{sys.executable}" "{worker}"\n')
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"import sys\nsys.path.insert(0, {str(FOLDER)!r})\nimport run_hybridep_ablation as ab\n"
        f"ab.REPO=ab.Path({str(fake_repo)!r})\nab.HERE=ab.Path({str(fake)!r})\n"
        "original=ab._join\nab._join=lambda e, node: original(e, node=node, repo=ab.REPO)\nab.main()\n"
    )

    def launch(fail=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AB_", "OV2_", "PET_"))}
        env.update(
            AB_ROOT=experiment.root, INIT_CKPT=experiment.init, PET_NNODES="4", AB_STEPS="104", AB_DISCARD="100"
        )
        if fail is not None:
            env["FAIL_NODE"] = str(fail)
        processes = []
        files = []
        for node in range(4):
            log = tmp_path / f"node{node}.txt"
            output = log.open("w")
            files.append((log, output))
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(driver)],
                    env={**env, "PET_NODE_RANK": str(node)},
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            )
        try:
            returns = [p.wait(timeout=30) for p in processes]
        finally:
            for p in processes:
                if p.poll() is None:
                    p.kill()
                p.wait()
            for _, output in files:
                output.close()
        return returns, "\n".join(log.read_text() for log, _ in files)

    return launch, Path(experiment.root)


def test_four_pod_sequential_run_and_report(cluster, experiment):
    launch, root = cluster
    returns, logs = launch()
    assert returns == [0, 0, 0, 0], logs
    report = json.loads((root / "comparison.json").read_text())
    assert report["nccl_throughput_change_pct"] == pytest.approx(100 * (20 / 18 - 1))
    assert report["arms"]["custom"]["timing"]["first"] == 101
    assert report["arms"]["custom"]["timing"]["count"] == 4
    assert report["metadata_order_matches"] and not report["gpu_parity_proven"]
    with (root / "nccl/inputs/rank000.txt").open("a") as f:
        f.write("extra\n")
    with pytest.raises(ValueError, match="input sequence evidence"):
        ab.compare(experiment)


def test_worker_failure_stops_other_pods_and_does_not_start_second_arm(cluster):
    launch, root = cluster
    returns, logs = launch(fail=2)
    assert all(code != 0 for code in returns), logs
    assert not (root / "comparison.json").exists()
    assert not list((root / "nccl").glob("train_node*.log"))


def test_stale_ready_does_not_release_new_worker(tmp_path, experiment):
    root = Path(experiment.root)
    (root / "control").mkdir(parents=True)
    ab._write(root / "experiment.json", ab.asdict(experiment))
    ab._write(root / "control/ready-1-oldnonce.json", True)
    with pytest.raises(TimeoutError):
        ab._wait([root / "control/ready-1-newnonce.json"], root=root, seconds=0)


def test_wrapper_rejects_uncontrolled_overrides():
    result = subprocess.run(
        ["bash", str(FOLDER / "ax_ov2_qwen35_s15_hep_ab16.sh"), "SAVE=/production"], capture_output=True, text=True
    )
    assert result.returncode != 0 and "unsupported ablation argument" in result.stderr
