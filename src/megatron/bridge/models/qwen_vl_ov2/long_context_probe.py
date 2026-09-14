# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in runtime and CPU batch evidence for long-video smoke/resume experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from megatron.bridge.models.qwen_vl_ov2.ablation_inputs import batch_digest


_COUNTS: dict[str, int] = {}
_SEEN: set[str] = set()


def record_batch(batch: Mapping[str, Any]) -> None:
    """Record metadata, useful length and visual workload before the CUDA transfer.

    Pixel values are excluded from the digest, as in the existing 16-GPU audit.
    Geometry and token order equality is not full numerical parity.
    """
    root = Path(os.environ["OV2_LONG_CONTEXT_PROBE_DIR"])
    path = root / f"inputs-rank{int(os.environ['RANK']):03d}.jsonl"
    key = str(path)
    count = _COUNTS.get(key, 0) + 1
    tokens = batch.get("tokens", batch.get("input_ids"))
    cu = batch.get("cu_seqlens")
    pixels = batch.get("pixel_values")
    mask = batch.get("loss_mask")
    positions = batch.get("patch_positions")
    length = int(tokens.shape[-1])
    item = {
        "index": count,
        "digest": batch_digest(batch),
        "tokens": length,
        "payload_tokens": int(cu.flatten()[-1]) if cu is not None else length,
        "loss_tokens": int(torch.count_nonzero(mask)) if mask is not None else None,
        "patches": int(pixels.shape[0]) if pixels is not None else 0,
        "temporal_max": int(positions[:, 0].max()) if positions is not None and positions.numel() else 0,
    }
    with path.open("x" if count == 1 else "a") as stream:
        stream.write(json.dumps(item) + "\n")
    _COUNTS[key] = count


def runtime_snapshot(state: Any, model: Any, parallel: Any) -> dict[str, Any]:
    """Read both the built inner LLM and initialized groups, before the first step."""
    while hasattr(model, "module"):
        model = model.module
    inner = model.language_model.config
    cfg = state.cfg
    return {
        "world": torch.distributed.get_world_size(),
        "tp": parallel.get_tensor_model_parallel_world_size(),
        "etp": parallel.get_expert_tensor_parallel_world_size(),
        "ep": parallel.get_expert_model_parallel_world_size(),
        "dp": parallel.get_data_parallel_world_size(),
        "expert_dp": parallel.get_expert_data_parallel_world_size(),
        "inner_tp": inner.tensor_model_parallel_size,
        "inner_etp": inner.expert_tensor_parallel_size,
        "inner_ep": inner.expert_model_parallel_size,
        "dispatcher": inner.moe_token_dispatcher_type,
        "recompute": inner.recompute_granularity,
        "seq": cfg.model.seq_length,
        "gbs": cfg.train.global_batch_size,
        "mbs": cfg.train.micro_batch_size,
        "step": int(state.train_state.step),
        "samples": int(state.train_state.consumed_train_samples),
        "finetune": cfg.checkpoint.finetune,
        "load_optim": cfg.checkpoint.load_optim,
        "load_rng": cfg.checkpoint.load_rng,
        "load": cfg.checkpoint.load,
        "optimizer": cfg.optimizer.optimizer,
        "scheduler_from_checkpoint": cfg.scheduler.use_checkpoint_opt_param_scheduler,
        "trainable": {
            name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
            for name in ("language_model", "vision_model", "adapter")
        },
    }


def record_runtime(state: Any, model: Any) -> None:
    """Persist one snapshot per process and fail collectively on a contract mismatch."""
    root = os.environ["OV2_LONG_CONTEXT_PROBE_DIR"]
    if root in _SEEN:
        return
    from megatron.core import parallel_state

    item = runtime_snapshot(state, model, parallel_state)
    expected = json.loads(os.environ["OV2_LONG_CONTEXT_EXPECT"])
    bad = {key: [item.get(key), value] for key, value in expected.items() if item.get(key) != value}
    if not all(item["trainable"].values()):
        bad["trainable"] = item["trainable"]
    item["mismatches"] = bad
    rank = torch.distributed.get_rank()
    with (Path(root) / f"runtime-rank{rank:03d}.json").open("x") as stream:
        json.dump(item, stream, indent=2)
    flag = torch.tensor([bool(bad)], dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    if flag.item():
        raise RuntimeError(f"long-context runtime contract mismatch; inspect {root}/runtime-rank*.json")
    _SEEN.add(root)
