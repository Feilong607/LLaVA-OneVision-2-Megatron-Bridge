"""CPU-only launcher regressions; run with --confcutdir=tests/unit_tests/models/ov2."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


REPO = Path(__file__).resolve().parents[4]
LAUNCHERS = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200"
pytestmark = pytest.mark.unit


@pytest.fixture
def launch(tmp_path):
    folder = tmp_path / "qwen/qwen35_vl_ov2/gb200"
    folder.mkdir(parents=True)
    # Redirect logs in the test copies only. Never change HOME or real checkpoints.
    for name in ("ax_ov2_qwen35_s15_prod48.sh", "ax_ov2_qwen35_s15_prod32.sh"):
        source = (LAUNCHERS / name).read_text()
        (folder / name).write_text(source.replace("$HOME/train_logs", str(tmp_path / "logs")))
    stage2 = tmp_path / "stage2/iter_0006000"
    stage2.mkdir(parents=True)
    (stage2 / ".metadata").touch()
    dataset = tmp_path / "qwen/qwen3_vl_ov2/gb200"
    dataset.mkdir(parents=True)
    (dataset / "mid_training_seed85m.yaml").write_text(f"path: {tmp_path}\n")
    capture = tmp_path / "captured.json"
    (folder / "ax_ov2_qwen35_35b_a3b_gb200.sh").write_text(
        "#!/usr/bin/env bash\npython3 - <<'CAPTURE'\n"
        "import json, os\nfrom pathlib import Path\n"
        f"Path({str(capture)!r}).write_text(json.dumps(dict(os.environ)))\nCAPTURE\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {"hostname": "echo prod48-master-0", "sleep": "exit 1"}.items():
        script = binaries / name
        script.write_text(f"#!/usr/bin/env bash\n{body}\n")
        script.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("OV2_", "PET_"))}
    for key in (
        "TP",
        "NPROC",
        "ITERS",
        "SAVE",
        "INIT_CKPT",
        "EXTRA_ARGS",
        "ACCEL",
        "SAVE_EVERY",
        "RESUME_SRC",
        "RESUME_ITER",
        "SRC_GBS",
        "PRIOR_CONSUMED",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{binaries}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
        PET_NNODES="12",
        SAVE=str(tmp_path / "save"),
        INIT_CKPT=str(stage2),
        OV2_PREFLIGHT_ONLY="1",
        OV2_LLM_HF_QWEN35=str(tmp_path / "model-text"),
        OV2_HF_PROC_QWEN35_P16M33=str(tmp_path / "processor"),
    )

    def run(*args, **overrides):
        result = subprocess.run(
            ["bash", str(folder / "ax_ov2_qwen35_s15_prod48.sh"), *args],
            env={**env, **overrides},
            text=True,
            capture_output=True,
            timeout=30,
        )
        return result.returncode, result.stdout + result.stderr

    return run, tmp_path, capture


def checkpoint(tmp_path, step=500, *, dp=48, bridge_tracker=True, config_changes=None):
    root = tmp_path / "save"
    folder = root / f"iter_{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in (".metadata", "metadata.json"):
        (folder / name).touch()
    state = {"step": torch.tensor(step), "consumed_train_samples": torch.tensor(step * 240)}
    torch.save(state, folder / "train_state.pt")
    if bridge_tracker:
        torch.save(state, root / "latest_train_state.pt")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    for rank in range(dp):
        (folder / f"train_dataloader_dprank{rank:03d}.pt").touch()
    config = {
        "model": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 8,
        },
        "train": {"global_batch_size": 240, "micro_batch_size": 1, "train_iters": 33334},
        "optimizer": {"optimizer": "dist_muon", "lr": 1e-5, "min_lr": 1e-6},
        "scheduler": {"lr_warmup_iters": 66, "lr_decay_iters": 33334},
        "checkpoint": {"ckpt_format": "torch_dist", "save_optim": True, "save_rng": True},
    }
    for section, fields in (config_changes or {}).items():
        config[section].update(fields)
    (folder / "run_config.yaml").write_text(yaml.safe_dump(config))
    return folder


def test_fresh_start_full_budget_and_inherited_launcher(launch):
    run, _, capture = launch
    rc, output = run(OV2_PREFLIGHT_ONLY="0")
    assert rc == 0, output
    values = json.loads(capture.read_text())
    assert "fresh midtrain" in output
    assert "iters=33334 warmup=66" in output
    assert values["INIT_CKPT"].endswith("stage2/iter_0006000")
    assert values["OV2_MIDTRAIN_N_SAMPLES"] == "8000000"
    assert values["TP"] == "1" and values["OV2_MIDTRAIN_GBS"] == "240"
    assert values["ACCEL"] == "2" and values["OV2_VISION_RECOMPUTE"] == "1"


def test_resume_without_initial_or_32gpu_source(launch):
    run, tmp_path, capture = launch
    folder = checkpoint(tmp_path)
    shutil.rmtree(tmp_path / "stage2")
    rc, output = run(OV2_PREFLIGHT_ONLY="0")
    assert rc == 0, output
    values = json.loads(capture.read_text())
    assert values["INIT_CKPT"] == str(folder)
    assert "continue to iter 33334" in output
    assert values["OV2_MIDTRAIN_N_SAMPLES"] == "8000000"
    for arg in (
        "checkpoint.finetune=false",
        "checkpoint.load_optim=true",
        "checkpoint.load_rng=true",
        "scheduler.override_opt_param_scheduler=false",
        "scheduler.use_checkpoint_opt_param_scheduler=true",
    ):
        assert arg in values["EXTRA_ARGS"].split()


@pytest.mark.parametrize("bridge_tracker", [False, True])
def test_resume_uses_tracker_not_unfinished_later_directory(launch, bridge_tracker):
    run, tmp_path, _ = launch
    checkpoint(tmp_path, bridge_tracker=bridge_tracker)
    (tmp_path / "save/iter_0001000").mkdir()
    if bridge_tracker:
        # Bridge's .pt tracker takes precedence over a stale legacy text tracker.
        (tmp_path / "save/latest_checkpointed_iteration.txt").write_text("1000")
    rc, output = run()
    assert rc == 0, output
    assert "resume:" in output and "iter_0000500" in output


@pytest.mark.parametrize("dp", [0, 32, 47])
def test_reject_wrong_or_missing_dataloader_states(launch, dp):
    run, tmp_path, _ = launch
    checkpoint(tmp_path, dp=dp)
    (tmp_path / "save/iter_0001000").mkdir()
    rc, output = run()
    assert rc != 0 and "exactly DP48" in output


@pytest.mark.parametrize(
    "fields",
    [
        {"train": {"global_batch_size": 288}},
        {"train": {"train_iters": 30000}},
        {"model": {"tensor_model_parallel_size": 2}},
        {"optimizer": {"optimizer": "adam"}},
        {"optimizer": {"lr": 2e-5}},
        {"scheduler": {"lr_warmup_iters": 50}},
        {"checkpoint": {"save_optim": False}},
        {"checkpoint": {"save_rng": False}},
    ],
)
def test_resume_rejects_changed_training_configuration(launch, fields):
    run, tmp_path, _ = launch
    checkpoint(tmp_path, config_changes=fields)
    rc, output = run()
    assert rc != 0 and "keep the original Args" in output


def test_reject_incomplete_tracker_target(launch):
    run, tmp_path, _ = launch
    folder = checkpoint(tmp_path)
    (folder / "run_config.yaml").unlink()
    rc, output = run()
    assert rc != 0 and "incomplete checkpoint" in output


def test_reject_iteration_directory_as_save(launch):
    run, tmp_path, _ = launch
    folder = checkpoint(tmp_path)
    rc, output = run(SAVE=str(folder))
    assert rc != 0 and "run root" in output


def test_before_first_checkpoint_restarts_from_stage2(launch):
    run, tmp_path, _ = launch
    (tmp_path / "save/iter_0000500").mkdir(parents=True)
    rc, output = run()
    assert rc == 0 and "fresh midtrain" in output


@pytest.mark.parametrize(
    "arg",
    [
        "OV2_MIDTRAIN_GBS=256",
        "TP=4",
        "PET_NNODES=8",
        "OV2_MIDTRAIN_N_SAMPLES=0",
        "OV2_LENGTH_SORT_WINDOW=8",
        "RESUME_ITER=latest",
        "ITERS=100",
    ],
)
def test_reject_invalid_launch_args(launch, arg):
    run, _, _ = launch
    rc, output = run(arg)
    assert rc != 0 and "FATAL:" in output


def test_288_batch_and_explicit_zero_warmup(launch):
    run, _, _ = launch
    rc, output = run("OV2_MIDTRAIN_GBS=288", "OV2_WARMUP_ITERS=0")
    assert rc == 0, output
    assert "mb_per_rank=6" in output and "iters=27778 warmup=0" in output


@pytest.mark.parametrize("propagate,expected_rc", [("0", 0), ("1", 7)])
def test_ablation_can_propagate_worker_failure_without_changing_default(launch, propagate, expected_rc):
    run, tmp_path, _ = launch
    (tmp_path / "bin/hostname").write_text("#!/usr/bin/env bash\necho prod48-worker-0\n")
    base = tmp_path / "qwen/qwen35_vl_ov2/gb200/ax_ov2_qwen35_35b_a3b_gb200.sh"
    with base.open("a") as stream:
        stream.write("exit 7\n")
    rc, output = run(OV2_PREFLIGHT_ONLY="0", OV2_PROPAGATE_WORKER_RC=propagate)
    assert rc == expected_rc, output
