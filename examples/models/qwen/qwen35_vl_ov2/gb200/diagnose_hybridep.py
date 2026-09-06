# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read Qwen3.5 TP1/EP8 production logs and optional TensorBoard scalars.

Counter-to-step inference assumes 40 decoder MoEs + 1 MTP MoE, each evaluated
once in forward and once in selective MoE recompute (82 allgathers/microbatch).
It is NOT a CUDA trace or a measurement of avoidable communication overhead.
See production-log-audit.md. No model, torch, or distributed imports are used.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import logging
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analyze_prod_logs import _ANSI, Snapshot, _collect


logger = logging.getLogger(__name__)
_TIMEOUT = re.compile(r"HYBRID-EP ALLGATHER TIMEOUT:.*?expecting\s+(\d+)\s+got\s+(\d+)")
_PHASE = re.compile(
    r"\[PHASETIMER r(\d+)\] fwd#(\d+) prefix\(vision\+adapter\+fuse\)=(\d+)ms "
    r"llm=(\d+)ms .*?patches=(\d+)"
)


@dataclass(frozen=True)
class Event:
    """One printed timeout; repeated SM/rank messages are not independent failures."""

    group: int
    expected: int
    got: int
    pod: str
    line: int


@dataclass(frozen=True)
class Phase:
    """A sampled forward; timing includes waits and excludes backward."""

    rank: int
    forward: int
    prefix_ms: int
    llm_ms: int
    patches: int


def _evidence(snapshot: Snapshot) -> tuple[list[Event], list[Phase], Path | None, bool]:
    events: list[Event] = []
    phases: list[Phase] = []
    saves: set[Path] = set()
    supported = True
    for pod in snapshot.pods:
        wrapper: dict[str, str] = {}
        base: dict[str, str] = {}
        node = 0 if pod.pod == "master-0" else int(pod.pod.split("-")[1]) + 1
        with pod.path.open(errors="replace") as stream:
            for lineno, raw in enumerate(stream, 1):
                if lineno < pod.segment_start or not raw.endswith("\n"):
                    continue
                line = _ANSI.sub("", raw)
                if "[qwen35-s15-prod] tp=" in line:
                    wrapper = dict(re.findall(r"\b(\w+)=(\S+)", line))
                    if "save" in wrapper:
                        saves.add(Path(wrapper["save"]))
                if "in-container |" in line:
                    base = dict(re.findall(r"\b(\w+)=(\S+)", line))
                match = _TIMEOUT.search(line)
                if match:
                    events.append(Event(node // 2, int(match[1]), int(match[2]), pod.pod, lineno))
                match = _PHASE.search(line)
                if match:
                    phases.append(Phase(*(int(value) for value in match.groups())))
        # No cross-attempt or TP2/full-recompute counter mapping. The remaining
        # 40+1 layers, no extra eval forwards, and no buffer reset are assumptions.
        supported &= pod.starts == 1 and pod.regressions == 0
        supported &= all(base.get(key) == value for key, value in {"tp": "1", "nproc": "4", "accel": "2"}.items())
        supported &= wrapper.get("recompute_full") == "0" and wrapper.get("recompute_moe") == "1"
    supported &= bool(snapshot.last_rank.steps) and snapshot.last_rank.steps[0].number == 1
    supported &= snapshot.nodes is not None and len(snapshot.pods) == snapshot.nodes
    if len(saves) > 1:
        raise ValueError("各 pod 的 SAVE 不一致；不合并 TensorBoard")
    return events, phases, next(iter(saves), None), supported


def _position(expected: int, microbatches: int) -> tuple[int, int, int, int] | None:
    if expected <= 0 or expected % 8 or microbatches < 1:
        return None
    call = expected // 8
    forward, offset = divmod(call - 1, 82)
    iteration, microbatch = divmod(forward, microbatches)
    return iteration + 1, microbatch + 1, offset + 1, forward + 1


def _report_events(snapshot: Snapshot, events: list[Event], phases: list[Phase], supported: bool) -> set[int]:
    logger.info("TIMEOUT 打印行=%d；先按候选 EP 组+expected 合并，不能把行数当故障数。", len(events))
    if not supported:
        logger.warning("缺少完整首次启动/TP1/4-GPU-pod/ACCEL2/selective-MoE 证据；不推算超时 iteration。")
        return set()
    batch_sizes = {row.batch_size for row in snapshot.last_rank.steps}
    world = (snapshot.nodes or 0) * 4
    if len(batch_sizes) != 1 or not world or next(iter(batch_sizes)) % world:
        logger.warning("GBS/DP 不一致；不推算计数位置。")
        return set()
    microbatches = next(iter(batch_sizes)) // world
    logger.info("条件推算：EP8；40+1 个 MoE×前向/重算=82 次/mb；每步 %d mb。", microbatches)
    logger.info("仍须确认无额外 eval forward、buffer 重建或未记录重启；这是定位假设，不是实测时间戳。")
    grouped: dict[tuple[int, int], list[Event]] = defaultdict(list)
    sampled: dict[tuple[int, int], dict[int, Phase]] = defaultdict(dict)
    for event in events:
        grouped[(event.group, event.expected)].append(event)
    for phase in phases:
        sampled[(phase.rank // 8, phase.forward)][phase.rank] = phase
    steps = {row.number: row for row in snapshot.last_rank.steps}
    inferred: set[int] = set()
    logger.info("候选组 expected  行数  推算iter/mb/调用序号  完成步秒")
    for (group, expected), messages in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        position = _position(expected, microbatches)
        if position is None:
            logger.warning("EP%d expected=%d 不符合 EP8 计数，跳过推算", group, expected)
            continue
        iteration, microbatch, offset, forward = position
        inferred.add(iteration)
        step = steps.get(iteration)
        logger.info(
            "EP%d %8d %4d   %d/%d/%d   %s  (%s:%d)",
            group,
            expected,
            len(messages),
            iteration,
            microbatch,
            offset,
            f"{step.seconds:.2f}" if step else "未见完成步",
            messages[0].pod,
            messages[0].line,
        )
        local = next((event for event in messages if event.pod == snapshot.last_rank.pod), None)
        if local:
            before = [row.number for row in steps.values() if row.line < local.line]
            after = [row.number for row in steps.values() if row.line > local.line]
            logger.info(
                "  最后rank pod日志位置：完成iter %s 与 %s 之间（CUDA printf可能延迟输出）。",
                max(before) if before else "NA",
                min(after) if after else "NA",
            )
        rows = list(sampled[(group, forward)].values())
        if rows:
            logger.info(
                "  同一fwd#%d采样 %d/8 ranks：prefix %.2f–%.2fs；LLM %.2f–%.2fs；patches %d–%d",
                forward,
                len(rows),
                min(row.prefix_ms for row in rows) / 1000,
                max(row.prefix_ms for row in rows) / 1000,
                min(row.llm_ms for row in rows) / 1000,
                max(row.llm_ms for row in rows) / 1000,
                min(row.patches for row in rows),
                max(row.patches for row in rows),
            )
        else:
            logger.info("  同一 fwd#%d 未采样 PHASETIMER；不用相邻 microbatch 代替。", forward)
    logger.info("调用序号1指向 microbatch 首次 MoE；prefix 不含上一 microbatch 的视觉 backward。")
    return inferred


def _compare(values: dict[int, float], event_steps: set[int], label: str) -> None:
    bad = sum(not math.isfinite(value) for value in values.values())
    finite = {step: value for step, value in values.items() if math.isfinite(value)}
    hit = [value for step, value in finite.items() if step in event_steps]
    other = [value for step, value in finite.items() if step not in event_steps]
    logger.info(
        "%s：候选超时步 n=%d mean=%s；其余步 n=%d mean=%s；非有限值=%d",
        label,
        len(hit),
        f"{statistics.mean(hit):.6g}" if hit else "NA",
        len(other),
        f"{statistics.mean(other):.6g}" if other else "NA",
        bad,
    )
    if hit and other and label == "日志秒/步":
        logger.info(
            "  两组均值差=%+.3fs/步（相关性；样本难度、首步JIT等未控制，不能当可恢复速度）。",
            statistics.mean(hit) - statistics.mean(other),
        )


def _tensorboard(path: Path, event_steps: set[int], first: int, last: int) -> None:
    # TensorBoard may find an installed TensorFlow. Hide GPUs in THIS diagnostic
    # process before that optional import; never import training modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("当前诊断容器没有 TensorBoard；日志诊断已完成，未自动安装依赖。")
        return
    directories = sorted({file.parent for file in path.rglob("events.out.tfevents.*") if file.is_file()})
    if not directories:
        logger.warning("TensorBoard 目录未找到 event 文件：%s", path)
        return
    for directory in directories:
        # TensorBoard's public event-processing API is untyped.
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})  # type: ignore[no-untyped-call]
        accumulator.Reload()  # type: ignore[no-untyped-call]
        tags = accumulator.Tags()  # type: ignore[no-untyped-call]
        relevant = [
            tag
            for tag in tags.get("scalars", [])
            if re.search(
                r"loss|grad.norm|iteration.time|forward.backward|batch.generator|optimizer|all.?gather|all.?to.?all",
                tag,
                re.I,
            )
            and "vs samples" not in tag
        ]
        logger.info("TensorBoard: %s；原始scalar（未平滑），窗口iter %d–%d", directory, first, last)
        for tag in relevant:
            # Do not silently overwrite duplicate steps from restarts/events.
            rows = [row for row in accumulator.Scalars(tag) if first <= row.step <= last]  # type: ignore[no-untyped-call]
            values = {int(row.step): float(row.value) for row in rows}
            if len(values) != len(rows):
                logger.warning("%s 有重复step，可能混入另一启动段；不合并", tag)
                continue
            if values:
                _compare(values, event_steps, tag)
        if not relevant:
            logger.info("没有匹配的scalar；可用scalar tags=%s", tags.get("scalars", []))
        if tags.get("tensors"):
            logger.info("另有 %d 个tensor-format tags，本工具未解码，不能当作缺失数据。", len(tags["tensors"]))
    logger.info("TensorBoard 聚合值不能证明通信正确，也不能分离同一 kernel 中的传输与等待。")


def _runtime() -> None:
    logger.info(
        "运行本脚本的容器 hostname=%s python=%s；code-sync 容器不一定等于训练容器。",
        os.uname().nodename,
        sys.executable,
    )
    for name in ("deep_ep", "hybrid_ep_cpp"):
        spec = importlib.machinery.PathFinder.find_spec(name)
        logger.info("%s import候选=%s（未导入模块）", name, spec.origin if spec else "未找到")
        if name == "deep_ep" and spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                source = Path(location) / "hybrid_ep_buffer.py"
                if source.is_file():
                    for lineno, line in enumerate(source.read_text().splitlines(), 1):
                        if "enable_custom_allgather" in line:
                            logger.info("%s:%d %s", source, lineno, line.strip())
    source_root = Path("/opt/DeepEP")
    if not source_root.is_dir():
        logger.warning("/opt/DeepEP 不存在；无法核验镜像内保留源码。")
        return
    result = subprocess.run(
        ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    logger.info("/opt/DeepEP HEAD=%s", result.stdout.strip() if result.returncode == 0 else "不可读")
    source = source_root / "csrc/hybrid_ep/extension/allgather.cu"
    if source.is_file():
        lines = source.read_text().splitlines()
        selected: set[int] = set()
        for index, line in enumerate(lines):
            if "#define TIMEOUT" in line or "clock64() - s" in line:
                selected.update(range(index, min(index + 7, len(lines))))
        for index in sorted(selected):
            logger.info("%s:%d %s", source, index + 1, lines[index].strip())
    logger.info("保留源码不能单独证明当前训练进程加载的二进制；结合镜像digest与模块路径核验。")


def main() -> int:
    """Report counter evidence and optional scalar correlations without changing training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "train_logs")
    parser.add_argument("--tensorboard-dir", type=Path, help="Default: SAVE/tensorboard from this job's wrapper log")
    parser.add_argument(
        "--runtime", action="store_true", help="Also inspect this container's DeepEP source without imports"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()
    if args.warmup < 0 or args.window < 1:
        parser.error("warmup >= 0 and window >= 1 required")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        if args.runtime:
            _runtime()
        snapshot = _collect(args.log_dir, args.job)
        events, phases, save, supported = _evidence(snapshot)
        inferred = _report_events(snapshot, events, phases, supported)
        rows = snapshot.last_rank.steps[args.warmup :][-args.window :]
        if rows and supported:
            logger.info("比较窗口：iter %d–%d；候选超时步不是随机分组。", rows[0].number, rows[-1].number)
            _compare({row.number: row.seconds for row in rows}, inferred, "日志秒/步")
            tb_path = args.tensorboard_dir or (save / "tensorboard" if save else None)
            if tb_path:
                _tensorboard(tb_path, inferred, rows[0].number, rows[-1].number)
            else:
                logger.warning("未识别 SAVE；用 --tensorboard-dir 指定实际目录。")
        else:
            logger.warning("缺少可用窗口/计数映射证据；不做 TensorBoard 超时归因。")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.error("诊断未完成：%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
