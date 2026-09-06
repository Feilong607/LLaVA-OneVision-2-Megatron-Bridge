"""CPU-only checks for conditional timeout attribution and TensorBoard reporting."""

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
FOLDER = ROOT / "examples/models/qwen/qwen35_vl_ov2/gb200"
sys.path.insert(0, str(FOLDER))
SPEC = importlib.util.spec_from_file_location("ov2_diagnose_hybridep", FOLDER / "diagnose_hybridep.py")
assert SPEC is not None and SPEC.loader is not None
diagnose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnose
SPEC.loader.exec_module(diagnose)
from analyze_prod_logs import PodLog, Snapshot, Step  # noqa: E402


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("expected", "position"),
    [(664, (1, 2, 1, 2)), (167944, (52, 2, 1, 257)), (278152, (85, 5, 1, 425)), (483480, (148, 3, 1, 738))],
)
def test_counter_positions(expected: int, position: tuple[int, ...]) -> None:
    assert diagnose._position(expected, 5) == position


@pytest.mark.parametrize("expected", [0, -8, 8356])
def test_invalid_counters_not_rounded_into_plausible_steps(expected: int) -> None:
    assert diagnose._position(expected, 5) is None


def _snapshot(tmp_path: Path) -> Snapshot:
    pods = []
    for node in range(12):
        pod = "master-0" if node == 0 else f"worker-{node - 1}"
        path = tmp_path / f"{pod}.log"
        path.write_text(
            "[qwen35-s15-prod] tp=1 gbs=240 recompute_full=0 recompute_moe=1 save=/test\n"
            "in-container | tp=1 nproc=4 accel=2 nnodes=12 dp=48\n"
        )
        pods.append(PodLog(path=path, pod=pod, starts=1))
    pods[-1].steps = [
        Step(
            number=number,
            total=33334,
            samples=number * 240,
            seconds=20,
            batch_size=240,
            stamp=datetime(2026, 1, 1),
            line=100 + number,
        )
        for number in range(1, 151)
    ]
    return Snapshot(pods=pods, last_rank=pods[-1], observed_at=0, nodes=12)


def test_sm_duplicates_merge_but_distinct_ep_groups_do_not(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    snapshot = _snapshot(tmp_path)
    events = [
        diagnose.Event(0, 278152, 278151, "master-0", 40),
        diagnose.Event(0, 278152, 278151, "worker-0", 41),
        diagnose.Event(1, 278152, 278151, "worker-1", 42),
    ]
    phases = [diagnose.Phase(0, 425, 2000, 30000, 3000), diagnose.Phase(1, 424, 90000, 4000, 90000)]
    with caplog.at_level(logging.INFO):
        inferred = diagnose._report_events(snapshot, events, phases, True)
    assert inferred == {85}
    assert "同一fwd#425采样 1/8 ranks" in caplog.text
    assert "EP1" in caplog.text
    assert "90.00" not in caplog.text  # Never substitute another forward's timing.


def test_segment_isolation_and_resume_blocks_counter_inference(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    pod = snapshot.pods[0]
    pod.path.write_text(
        "HYBRID-EP ALLGATHER TIMEOUT:SM 1 [0]:expecting 664 got 662\n"
        + pod.path.read_text()
        + "HYBRID-EP ALLGATHER TIMEOUT:SM 1 [0]:expecting 1320 got 1318\n"
    )
    pod.segment_start = 2
    events, _, save, supported = diagnose._evidence(snapshot)
    assert [event.expected for event in events] == [1320]
    assert save == Path("/test") and supported
    pod.starts = 2
    assert not diagnose._evidence(snapshot)[3]
    pod.starts = 1
    snapshot.last_rank.steps = snapshot.last_rank.steps[100:]
    assert not diagnose._evidence(snapshot)[3]


def test_missing_pod_or_conflicting_save_blocks_merge(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    snapshot.pods[0].path.write_text(snapshot.pods[0].path.read_text().replace("save=/test", "save=/wrong"))
    with pytest.raises(ValueError, match="SAVE"):
        diagnose._evidence(snapshot)
    snapshot.pods = snapshot.pods[1:]
    assert not diagnose._evidence(snapshot)[3]


def test_nonfinite_metrics_reported_not_averaged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        diagnose._compare({1: 30, 2: float("nan"), 3: 10}, {1, 2}, "日志秒/步")
    assert "非有限值=1" in caplog.text
    assert "mean=30" in caplog.text and "mean=10" in caplog.text
    assert "+20.000s" in caplog.text and "不能当可恢复速度" in caplog.text


@pytest.mark.parametrize("duplicate", [False, True])
def test_real_tensorboard_events(tmp_path: Path, caplog: pytest.LogCaptureFixture, duplicate: bool) -> None:
    pytest.importorskip("tensorboard")
    # Protobuf generates these message classes dynamically.
    from tensorboard.compat.proto.event_pb2 import Event  # type: ignore[attr-defined]
    from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore[attr-defined]
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    # TensorBoard's public writer API is untyped.
    writer = EventFileWriter(str(tmp_path))  # type: ignore[no-untyped-call]
    data = [(1, 999), (51, 50), (52, 20), (53, 20)]
    if duplicate:
        data.append((53, 30))
    for step, value in data:
        writer.add_event(  # type: ignore[no-untyped-call]
            Event(
                wall_time=1000 + step,
                step=step,
                summary=Summary(value=[Summary.Value(tag="iteration-time", simple_value=value)]),
            )
        )
    writer.close()  # type: ignore[no-untyped-call]
    with caplog.at_level(logging.INFO):
        diagnose._tensorboard(tmp_path, {51}, 51, 53)
    if duplicate:
        assert "重复step" in caplog.text and "mean=" not in caplog.text
    else:
        assert "mean=50" in caplog.text and "mean=20" in caplog.text
    assert "mean=999" not in caplog.text
