"""CPU-only production log diagnostics; no Bridge, torch, or GPU imports."""

import importlib.util
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "examples/models/qwen/qwen35_vl_ov2/gb200/analyze_prod_logs.py"
SPEC = importlib.util.spec_from_file_location("ov2_analyze_prod_logs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)
pytestmark = pytest.mark.unit
JOB = "test-qwen35-s15-48gpu"
BASE = datetime(2026, 1, 1)
HEADER = "[ov2-qwen35-gb200] in-container | world=48 dp=48 tp=1 gbs=240 nnodes=12\n"
WRAPPER = "[qwen35-s15-prod] tp=1 gbs=240 save=/example\n"


def _step(number: int, *, seconds: float = 20, elapsed: float | None = None, samples: int | None = None) -> str:
    stamp = BASE + timedelta(seconds=number * 20 if elapsed is None else elapsed)
    consumed = number * 240 if samples is None else samples
    return (
        f" [{stamp:%Y-%m-%d %H:%M:%S}] iteration {number:8d}/{33334:8d} |"
        f" consumed samples: {consumed:12d} | elapsed time per iteration (ms): {seconds * 1000:.1e} |"
        " tokens/s/GPU: 2000.0 | learning rate: 1e-5 | global batch size: 240 | loss: 1.1 |\n"
    )


def _path(folder: Path, *, job: str = JOB, pod: str = "worker-10") -> Path:
    return folder / f"prod_qwen35_s15_{job}_{job}-{pod}.log"


def test_window_excludes_startup_and_uses_correct_wall_boundaries(tmp_path: Path) -> None:
    path = _path(tmp_path)
    elapsed = 0.0
    lines = [WRAPPER, HEADER]
    for number in range(1, 152):
        seconds = 200 if number <= 50 else (100 if number == 150 else 20)
        elapsed += seconds
        lines.append(_step(number, seconds=seconds, elapsed=elapsed))
    path.write_text("".join(lines))
    pod = audit._read_pod(path, "worker-10")
    stats = audit._statistics(pod, warmup=50, window=100, min_steps=20)
    assert pod.starts == 1  # Wrapper and in-container banner describe one launch.
    assert (stats.first, stats.last, stats.count) == (52, 151, 100)
    assert stats.mean == pytest.approx(20.8)
    assert (stats.median, stats.p95, stats.maximum, stats.slow) == (20, 20, 100, 1)
    assert stats.samples_per_second == pytest.approx(24000 / 2080)
    assert stats.wall_samples_per_second == pytest.approx(24000 / 2080)


@pytest.mark.parametrize("restart", [HEADER, WRAPPER, WRAPPER + HEADER])
def test_restart_without_steps_does_not_report_old_progress(tmp_path: Path, restart: str) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + _step(100) + _TIMEOUT + restart)
    pod = audit._read_pod(path, "worker-10")
    assert pod.starts == 2
    assert pod.steps == []
    assert pod.timeouts == 1 and pod.current_timeouts == 0


_TIMEOUT = "HYBRID-EP ALLGATHER TIMEOUT:SM 22 [0]:expecting 1320 got 1318\n"


@pytest.mark.parametrize("next_number", [100, 20])
def test_duplicate_or_regressed_steps_reset_segment(tmp_path: Path, next_number: int) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + _step(100) + _step(next_number))
    pod = audit._read_pod(path, "worker-10")
    assert pod.regressions == 1
    assert [row.number for row in pod.steps] == [next_number]


def test_ansi_and_partial_final_line(tmp_path: Path) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + "\x1b[32m" + _step(1) + "\x1b[0m\n" + _step(2).rstrip())
    pod = audit._read_pod(path, "worker-10")
    assert [row.number for row in pod.steps] == [1]


def test_timeout_counts_are_lines_not_deduplicated_incidents(tmp_path: Path) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + _step(1) + _TIMEOUT * 3 + "UserWarning: AccumulateGrad stream mismatch\n")
    pod = audit._read_pod(path, "worker-10")
    assert (pod.timeouts, pod.current_timeouts, pod.stream_warnings) == (3, 3, 1)
    assert pod.last_timeout_line > pod.steps[-1].line


@pytest.mark.parametrize("bad", ["gap", "counter", "clock", "duration", "malformed"])
def test_bad_windows_are_not_summarized(tmp_path: Path, bad: str) -> None:
    records = [_step(number) for number in range(1, 25)]
    if bad == "gap":
        del records[10]
    elif bad == "counter":
        records[10] = _step(11, samples=0)
    elif bad == "clock":
        records[10] = _step(11, elapsed=1)
    elif bad == "duration":
        records[10] = _step(11, seconds=0)
    else:
        records[10] = records[10].replace("2026-01-01", "2026-99-01")
    path = _path(tmp_path)
    path.write_text(HEADER + "".join(records))
    pod = audit._read_pod(path, "worker-10")
    with pytest.raises(ValueError):
        audit._statistics(pod, warmup=0, window=100, min_steps=20)


def test_exact_job_match_and_numeric_last_worker_selection(tmp_path: Path) -> None:
    _path(tmp_path, pod="worker-9").write_text(HEADER)
    _path(tmp_path).write_text(HEADER + _step(100))
    _path(tmp_path, job=JOB + "-new").write_text(HEADER + _step(999))
    snapshot = audit._collect(tmp_path, JOB)
    assert len(snapshot.pods) == 2
    assert snapshot.last_rank.pod == "worker-10"
    assert snapshot.last_rank.steps[-1].number == 100


def test_missing_final_rank_does_not_fall_back_to_worker9(tmp_path: Path) -> None:
    _path(tmp_path, pod="worker-9").write_text(HEADER)
    with pytest.raises(ValueError, match="worker-10"):
        audit._collect(tmp_path, JOB)


def test_inconsistent_topology_is_rejected(tmp_path: Path) -> None:
    _path(tmp_path, pod="worker-9").write_text(HEADER.replace("nnodes=12", "nnodes=8"))
    _path(tmp_path).write_text(HEADER)
    with pytest.raises(ValueError, match="nnodes"):
        audit._collect(tmp_path, JOB)


def test_missing_header_infers_highest_worker_but_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _path(tmp_path, pod="worker-9").write_text("")
    _path(tmp_path).write_text(_step(1))
    snapshot = audit._collect(tmp_path, JOB)
    assert snapshot.nodes is None and snapshot.last_rank.pod == "worker-10"
    with caplog.at_level(logging.INFO):
        audit._report(snapshot, warmup=50, window=100, min_steps=20)
    assert "无法确认是否收齐" in caplog.text


@pytest.mark.parametrize("number,expected", [(103, "12.00 samples/s"), (100, "没有新的完成步"), (90, "进度回退")])
def test_observed_progress_includes_stalls_and_rollback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, number: int, expected: str
) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + _step(100))
    first = audit._collect(tmp_path, JOB)
    path.write_text(HEADER + _step(number))
    second = audit._collect(tmp_path, JOB)
    second.observed_at = first.observed_at + 60
    with caplog.at_level(logging.INFO):
        audit._report_progress(first, second)
    assert expected in caplog.text


def test_observation_during_restart_does_not_report_stale_speed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _path(tmp_path)
    path.write_text(HEADER + _step(100))
    first = audit._collect(tmp_path, JOB)
    path.write_text(HEADER + _step(100) + HEADER)
    second = audit._collect(tmp_path, JOB)
    second.observed_at = first.observed_at + 60
    with caplog.at_level(logging.INFO):
        audit._report_progress(first, second)
    assert "未取得有效完成步" in caplog.text
    assert "净推进=" not in caplog.text


def test_cli_uses_only_stdlib_and_does_not_change_logs(tmp_path: Path) -> None:
    path = _path(tmp_path)
    contents = HEADER + "".join(_step(number) for number in range(1, 90))
    path.write_text(contents)
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--job", JOB, "--log-dir", str(tmp_path), "--interval", "0.01"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "12.00 samples/s" in result.stdout
    assert "没有新的完成步" in result.stdout
    assert path.read_text() == contents


@pytest.mark.parametrize("args", [["--interval", "nan"], ["--window", "1"], ["--warmup", "-1"]])
def test_invalid_cli_values_fail(tmp_path: Path, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--job", JOB, "--log-dir", str(tmp_path), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_missing_job_is_not_reported_as_healthy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未找到"):
        audit._collect(tmp_path, JOB)
