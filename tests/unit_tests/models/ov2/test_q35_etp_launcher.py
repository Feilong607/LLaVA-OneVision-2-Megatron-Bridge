"""CPU-only checks of the ETP dependency used by merged-stage smoke launchers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[4]
FOLDER = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200"


@pytest.mark.parametrize(
    "world,tp,etp,valid",
    [
        (16, 4, "2", True),
        (48, 4, "2", True),
        (64, 4, "4", True),
        (48, 4, "", False),
        (16, 4, "4", False),
        (48, 4, "0", False),
        (48, 1, "", True),
        (32, 2, "", True),
    ],
)
def test_real_launcher_folding_and_legacy_defaults(tmp_path, world, tp, etp, valid):
    repo = tmp_path / "repo"
    (repo / "src/megatron/bridge").mkdir(parents=True)
    (repo / "3rdparty").mkdir()
    (repo / "3rdparty/apply_megatron_patch.sh").write_text("exit 0\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "capture.json"
    stub = binaries / "python"
    stub.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        f"open({str(capture)!r}, 'w').write(json.dumps(dict(args=sys.argv, env=dict(os.environ))))\n"
    )
    stub.chmod(0o755)
    env = {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "REPO": str(repo),
        "PET_NNODES": str(world // 4),
        "PET_NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "TP": str(tp),
        "OV2_ETP": etp,
        "ACCEL": "0",
        "OV2_MIDTRAIN_GBS": str(world),
        "SAVE": str(tmp_path / "save"),
        "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
    }
    result = subprocess.run(
        ["bash", str(FOLDER / "ax_ov2_qwen35_35b_a3b_gb200.sh")], env=env, capture_output=True, text=True, timeout=15
    )
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    if valid:
        args = json.loads(capture.read_text())["args"]
        assert f"model.tensor_model_parallel_size={tp}" in args
        if etp:
            assert f"model.expert_tensor_parallel_size={etp}" in args
        else:
            assert not any(arg.startswith("model.expert_tensor_parallel_size=") for arg in args)
    else:
        assert not capture.exists()


def test_merged_blend_weights_and_no_duplicate_paths():
    config = yaml.safe_load((FOLDER / "merged_s23_img30.yaml").read_text())
    rows = config["splits"]["train"]["datasets"]
    assert config["__class__"] == "Metadataset"
    assert len(rows) == 59
    assert len({row["path"] for row in rows}) == len(rows)
    assert all(row["weight"] > 0 for row in rows)
    total = sum(row["weight"] for row in rows)
    image = sum(row["weight"] for row in rows if "47m_v3" in row["path"])
    assert total == 180000 and image / total == 0.3
