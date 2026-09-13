"""CPU subprocess tests of 12-pod ladder coordination and result acceptance."""

import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[4]
FOLDER = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200"
FAKE = """import os,time
from pathlib import Path
home=Path.home(); host=os.environ['FAKE_HOST']; leg=os.environ['OV2_SMOKE_LEG']
tag=host.rsplit('-',2)[0]; logs=home/'train_logs'
(logs/f'{leg}.{host}.started').write_text(str(time.time()))
if host.endswith('worker-10') and leg=='A': time.sleep(1.5)
bad=os.environ.get('FAKE_MODE')=='worker_fail' and host.endswith('worker-5') and leg=='A'
seconds=100 if leg=='A' else 150
log=logs/f'smoke_qwen35_merged64k_{tag}-{leg}_{host}.log'
log.write_text(('prefix Step Time : '+str(seconds)+'s GPU utilization: 0\\n')*80 + f'[qwen35-smoke] rc={7 if bad else 0} pod_peak_mem_mib=1\\n')
if host.endswith('worker-10'):
    result=logs/f'smoke_qwen35_merged64k_result_{tag}-{leg}.txt'
    n=17 if os.environ.get('FAKE_MODE')=='short_run' else 77
    result.write_text(f'iters: n={n} p50=1ms\\ntorch memory: max_allocated=115.0 max_reserved=120.0\\nVERDICT: PASS\\n')
(logs/f'{leg}.{host}.finished').write_text(str(time.time()))
"""


@pytest.fixture
def run_ladder(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    script = folder / "ax_ov2_qwen35_speed_ladder48.sh"
    script.write_text((FOLDER / script.name).read_text())
    fake = folder / "fake.py"
    fake.write_text(FAKE)
    (folder / "ax_ov2_qwen35_merged64k_smoke.sh").write_text(f'exec "{sys.executable}" "{fake}"\n')
    for name in ("Qwen3.5-35B-A3B-text/config.json", "qwen35_p16m33_auto_model/preprocessor_config.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "hostname").write_text('#!/bin/sh\nprintf "%s\\n" "$FAKE_HOST"\n')
    (binary / "hostname").chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": f"{binary}:{Path(sys.executable).parent}:/usr/bin:/bin", "PET_NNODES": "12"}

    def run(mode="", nodes=12):
        if nodes != 12:
            script.write_text(script.read_text().replace("90 * 60", "2"))
        children = []
        for rank in range(nodes):
            host = "job-master-0" if rank == 0 else f"job-worker-{rank - 1}"
            children.append(
                subprocess.Popen(
                    ["bash", str(script)],
                    env={**env, "FAKE_HOST": host, "PET_NODE_RANK": str(rank), "FAKE_MODE": mode},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        outputs = []
        try:
            for child in children:
                out, err = child.communicate(timeout=30)
                outputs.append((child.returncode, out, err))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate()
        return outputs

    return run, tmp_path


def test_missing_pod_aborts_before_model_work(run_ladder):
    run, root = run_ladder
    outputs = run(nodes=11)
    assert all(rc == 3 for rc, _, _ in outputs), outputs
    assert not list((root / "train_logs").glob("A.*.started"))


def test_all_pods_wait_and_receive_same_summary(run_ladder):
    run, root = run_ladder
    outputs = run()
    assert all(rc == 0 for rc, _, _ in outputs), outputs
    logs = root / "train_logs"
    last_a = max(float(p.read_text()) for p in logs.glob("A.*.finished"))
    first_t = min(float(p.read_text()) for p in logs.glob("T.*.started"))
    assert first_t >= last_a
    summary = (logs / "smoke_speed_ladder_job.txt").read_text()
    assert "1.333x" in summary  # 96/150 divided by 48/100, not 100/150.
    assert len(list((logs / ".ladder_barrier_job").glob("summary_read.*"))) == 12
    assert all(rc != 0 for rc, _, _ in run()), "old attempts must not reuse barrier markers"


@pytest.mark.parametrize("mode", ["worker_fail", "short_run"])
def test_shared_pass_cannot_hide_worker_failure_or_short_run(run_ladder, mode):
    run, root = run_ladder
    outputs = run(mode)
    assert all(rc == 1 for rc, _, _ in outputs), outputs
    summary = (root / "train_logs/smoke_speed_ladder_job.txt").read_text()
    assert "T vs A speedup: n/a" in summary
