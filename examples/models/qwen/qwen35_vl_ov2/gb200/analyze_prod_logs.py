# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read Qwen3.5 production logs using only the Python standard library.

Run with --job WORKLOAD_NAME; add --interval 60 for two observations of net
sample progress. See production-log-audit.md for the measurement boundaries.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
_STEP = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*?"
    r"iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([\d.eE+-]+).*?global batch size:\s*(\d+)"
)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TIMEOUT = "HYBRID-EP ALLGATHER TIMEOUT"


@dataclass(frozen=True)
class Step:
    """One completed iteration record, with its location in the source log."""

    number: int
    total: int
    samples: int
    seconds: float
    batch_size: int
    stamp: datetime
    line: int


@dataclass
class PodLog:
    """Lifetime event counts and progress in the latest identifiable launch segment."""

    path: Path
    pod: str
    nodes: int | None = None
    starts: int = 0
    regressions: int = 0
    timeouts: int = 0
    current_timeouts: int = 0
    stream_warnings: int = 0
    last_timeout_line: int = 0
    malformed_steps: int = 0
    steps: list[Step] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)

    def reset_segment(self) -> None:
        """Discard previous-attempt timing without discarding lifetime event counts."""
        self.steps.clear()
        self.current_timeouts = 0
        self.last_timeout_line = 0
        self.malformed_steps = 0


@dataclass
class Snapshot:
    """A read-only observation of all retained pod logs for one exact job name."""

    pods: list[PodLog]
    last_rank: PodLog
    observed_at: float
    nodes: int | None


@dataclass(frozen=True)
class WindowStats:
    """Completed-step timing and timestamp-delimited consumed-sample throughput."""

    first: int
    last: int
    count: int
    mean: float
    median: float
    p95: float
    maximum: float
    slow: int
    samples_per_second: float
    wall_samples_per_second: float


def _read_pod(path: Path, pod: str) -> PodLog:
    result = PodLog(path=path, pod=pod)
    pending_wrapper = False
    with path.open(errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            # A live tee may not have finished writing its final record yet.
            if not raw.endswith("\n"):
                continue
            line = _ANSI.sub("", raw)
            wrapper = "[qwen35-s15-prod] tp=" in line
            container = "in-container |" in line
            if wrapper or container:
                if wrapper or not pending_wrapper:
                    result.starts += 1
                    result.reset_segment()
                pending_wrapper = wrapper
                if container:
                    nodes = re.search(r"\bnnodes=(\d+)\b", line)
                    if nodes:
                        result.nodes = int(nodes[1])
            if _TIMEOUT in line:
                result.timeouts += 1
                result.current_timeouts += 1
                result.last_timeout_line = line_number
            if "AccumulateGrad" in line or "stream mismatch" in line:
                result.stream_warnings += 1
            if any(marker in line for marker in (_TIMEOUT, "Starting training loop", "FATAL", "rc=")):
                result.events.append((line_number, line.strip()))
                result.events = result.events[-3:]
            match = _STEP.search(line)
            if not match:
                if "elapsed time per iteration (ms):" in line:
                    result.malformed_steps += 1
                continue
            stamp, number, total, samples, milliseconds, batch = match.groups()
            try:
                step = Step(
                    number=int(number),
                    total=int(total),
                    samples=int(samples),
                    seconds=float(milliseconds) / 1000,
                    batch_size=int(batch),
                    stamp=datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S"),
                    line=line_number,
                )
            except ValueError:
                result.malformed_steps += 1
                continue
            if result.steps and step.number <= result.steps[-1].number:
                result.regressions += 1
                result.reset_segment()
            result.steps.append(step)
    return result


def _collect(log_dir: Path, job: str) -> Snapshot:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", job):
        raise ValueError("--job 必须是完整 workload 名，不能包含路径或通配符")
    pattern = re.compile(rf"prod_qwen35_s15_{re.escape(job)}_{re.escape(job)}-(master|worker)-(\d+)\.log")
    paths = sorted(log_dir.glob(f"prod_qwen35_s15_{job}_*.log"))
    pods = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            pods.append(_read_pod(path, f"{match[1]}-{match[2]}"))
    if not pods:
        raise ValueError(f"未找到该 job 的生产日志：{log_dir} / {job}")
    node_counts = {pod.nodes for pod in pods if pod.nodes is not None}
    if len(node_counts) > 1:
        raise ValueError(f"各 pod 的 nnodes 不一致：{sorted(node_counts)}；先检查日志/启动配置")
    nodes = next(iter(node_counts), None)
    if nodes is not None:
        final_pod = f"worker-{nodes - 2}" if nodes > 1 else "master-0"
    else:
        workers = [int(pod.pod.split("-")[-1]) for pod in pods if pod.pod.startswith("worker-")]
        final_pod = f"worker-{max(workers)}" if workers else "master-0"
    last_rank = next((pod for pod in pods if pod.pod == final_pod), None)
    if last_rank is None:
        raise ValueError(f"缺少最后一个 rank 的日志 {final_pod}；不能用其他 pod 代算吞吐")
    return Snapshot(pods=pods, last_rank=last_rank, observed_at=time.monotonic(), nodes=nodes)


def _statistics(pod: PodLog, *, warmup: int, window: int, min_steps: int) -> WindowStats:
    if pod.malformed_steps:
        raise ValueError(f"最新启动段有 {pod.malformed_steps} 行 iteration 无法解析，暂不统计")
    # Keep one boundary record so wall time and consumed samples cover identical steps.
    rows = pod.steps[warmup:][-(window + 1) :]
    if len(rows) < min_steps + 1:
        raise ValueError(f"排除本次启动前 {warmup} 步后，需至少 {min_steps} 个步间区间；当前样本不足")
    for first, second in zip(rows, rows[1:]):
        if second.number != first.number + 1:
            raise ValueError("窗口有缺步或 log_interval != 1；停止统计以免误算")
        if second.stamp < first.stamp or second.samples - first.samples != second.batch_size:
            raise ValueError("时间戳或 consumed samples 不连续；停止统计以免误算")
    durations = [row.seconds for row in rows[1:]]
    if any(not math.isfinite(seconds) or seconds <= 0 for seconds in durations):
        raise ValueError("窗口包含无效 iteration 耗时")
    wall_seconds = (rows[-1].stamp - rows[0].stamp).total_seconds()
    if wall_seconds <= 0:
        raise ValueError("窗口时间戳跨度不足，无法计算实际吞吐")
    median = statistics.median(durations)
    return WindowStats(
        first=rows[1].number,
        last=rows[-1].number,
        count=len(durations),
        mean=statistics.mean(durations),
        median=median,
        p95=sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
        maximum=max(durations),
        slow=sum(seconds > 1.5 * median for seconds in durations),
        samples_per_second=sum(row.batch_size for row in rows[1:]) / sum(durations),
        wall_samples_per_second=(rows[-1].samples - rows[0].samples) / wall_seconds,
    )


def _report(snapshot: Snapshot, *, warmup: int, window: int, min_steps: int) -> None:
    logger.info("pod                 starts  回退  TIMEOUT总/当前段  stream警告行")
    for pod in snapshot.pods:
        logger.info(
            "%-19s %6d %5d %7d/%-7d %8d",
            pod.pod,
            pod.starts,
            pod.regressions,
            pod.timeouts,
            pod.current_timeouts,
            pod.stream_warnings,
        )
    logger.info("TIMEOUT 是打印行数，不是独立故障次数；stream 警告行数也不代表同步发生次数。")
    if snapshot.nodes is None:
        logger.warning("缺少 nnodes，暂按最大 worker 编号选日志；无法确认是否收齐所有 pod。")
    elif len(snapshot.pods) != snapshot.nodes:
        logger.warning("日志覆盖不完整：%d/%d pods", len(snapshot.pods), snapshot.nodes)
    for pod in snapshot.pods:
        for line, message in pod.events:
            logger.info("%s:%d: %s", pod.path.name, line, message)
    last = snapshot.last_rank
    logger.info("统计日志：%s", last.path)
    if not last.steps:
        logger.warning("最新启动段没有已完成 iteration；历史速度不能代表当前进度。")
        return
    step = last.steps[-1]
    logger.info(
        "最后记录：iter=%d/%d consumed_samples=%d，日志时间=%s（日志本地时区）",
        step.number,
        step.total,
        step.samples,
        step.stamp,
    )
    if last.last_timeout_line > step.line:
        logger.warning("此启动段最后一条 TIMEOUT 位于最后一条 iteration 之后；尚未见后续完成步。")
    try:
        stats = _statistics(last, warmup=warmup, window=window, min_steps=min_steps)
    except ValueError as exc:
        logger.warning("%s", exc)
        return
    logger.info("稳态窗口：iter %d–%d，%d 步", stats.first, stats.last, stats.count)
    logger.info(
        "秒/步：均值=%.2f 中位数=%.2f P95=%.2f 最大=%.2f",
        stats.mean,
        stats.median,
        stats.p95,
        stats.maximum,
    )
    logger.info("慢步（>1.5×中位数）=%d/%d；均值/中位数=%.2f", stats.slow, stats.count, stats.mean / stats.median)
    logger.info("已完成步计时吞吐=%.2f samples/s", stats.samples_per_second)
    logger.info("同窗口日志时间戳吞吐=%.2f samples/s", stats.wall_samples_per_second)
    logger.info("以上不含最后完成步之后的卡顿/停机。慢步也可能来自样本差异或存档，不能据此归因。")


def _report_progress(first: Snapshot, second: Snapshot) -> None:
    elapsed = second.observed_at - first.observed_at
    first_counts = {pod.path: pod.timeouts for pod in first.pods}
    if any(pod.timeouts < first_counts.get(pod.path, 0) for pod in second.pods):
        logger.warning("部分 TIMEOUT 行数减少，日志可能被截断/替换；不计算新增行数。")
    else:
        added = sum(pod.timeouts - first_counts.get(pod.path, 0) for pod in second.pods)
        logger.info("两次观测间新增 TIMEOUT 打印行数=%d（可能是同一次故障的重复打印）", added)
    if elapsed <= 0 or not first.last_rank.steps or not second.last_rank.steps:
        logger.warning("两次观测未取得有效完成步，无法计算净推进；请检查当前 pods 和启动日志。")
        return
    before, after = first.last_rank.steps[-1], second.last_rank.steps[-1]
    if first.last_rank.path != second.last_rank.path or before.total != after.total:
        logger.warning("两次观测的日志或总步数不同，不合并计算吞吐。")
        return
    delta = after.samples - before.samples
    logger.info(
        "观测 %.1fs：iter %d→%d；净 consumed samples=%+d；净推进=%.2f samples/s",
        elapsed,
        before.number,
        after.number,
        delta,
        delta / elapsed,
    )
    logger.info("两次完成步的日志时间：%s → %s", before.stamp, after.stamp)
    if delta < 0:
        logger.warning("净推进为负：记录显示样本进度回退，不是 GPU 负吞吐；请核查重启/恢复。")
    elif delta == 0:
        logger.info("观察期间没有新的完成步记录；可能在长步、初始化、存档、停机，或已结束。")
    logger.info("净推进含观察区间内的停顿/回退；单个短窗口有整步量化误差，不验证训练数值正确性。")


def main() -> int:
    """Run a log snapshot and optionally a second read for wall-clock progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Exact workload name, without a pod suffix")
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "train_logs")
    parser.add_argument("--warmup", type=int, default=50, help="Drop first N completed records after each launch (50)")
    parser.add_argument("--window", type=int, default=100, help="Maximum completed steps in the timing window (100)")
    parser.add_argument("--min-steps", type=int, default=20, help="Minimum steady-state intervals required (20)")
    parser.add_argument(
        "--interval", type=float, default=0, help="Seconds between two progress snapshots; 0 reads once"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.min_steps < 1 or args.window < args.min_steps:
        parser.error("require warmup >= 0 and window >= min-steps >= 1")
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("interval must be finite and nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        snapshot = _collect(args.log_dir, args.job)
        _report(snapshot, warmup=args.warmup, window=args.window, min_steps=args.min_steps)
        if args.interval:
            logger.info("等待 %.1fs 后复查净推进；仅观察日志，不访问 GPU。", args.interval)
            time.sleep(args.interval)
            _report_progress(snapshot, _collect(args.log_dir, args.job))
    except (OSError, ValueError) as exc:
        logger.error("日志检查失败：%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("观察已停止。")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
