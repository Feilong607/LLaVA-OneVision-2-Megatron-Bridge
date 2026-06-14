# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bridge-native forward step for OV2.1 (LlavaOnevision2), used via run_recipe.py --step_func ov2_step.

LlavaOnevision2.forward(images, image_grid_thw, input_ids, labels=...) returns the LLM's
per-token loss [b, s] (mcore GPTModel.compute_language_model_loss, which does NOT shift labels).
Labels are therefore expected PRE-SHIFTED by the data task encoder (roll -1), exactly the
QwenVLTaskEncoder convention. This step does NOT shift labels and does NOT double-count.

Assumes CP = 1 (LlavaOnevision2.forward asserts CP==1). Offline-packed OV2 samples use
THD block-diagonal attention via cu_seqlens; unlike qwen3_vl_step there is no CP split here.
"""
import logging
import os
from typing import Iterable

import torch
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.losses import create_masked_next_token_loss_function as _create_loss_function
from megatron.bridge.training.state import GlobalState

logger = logging.getLogger(__name__)


def _cuda(x):
    return x.cuda(non_blocking=True) if torch.is_tensor(x) else x


def get_batch(data_iterator: Iterable):
    """Pull one batch from the (energon) iterator and move tensors to CUDA. No label shift."""
    batch = next(data_iterator)
    tokens = _cuda(batch.get("tokens", batch.get("input_ids")))
    labels = _cuda(batch.get("labels"))
    loss_mask = _cuda(batch.get("loss_mask"))
    attention_mask = _cuda(batch.get("attention_mask"))
    pixel_values = _cuda(batch.get("pixel_values"))
    image_grid_thw = _cuda(batch.get("image_grid_thw"))
    cu_seqlens = _cuda(batch.get("cu_seqlens"))
    patch_positions = _cuda(batch.get("patch_positions"))   # video: [P,3] int64 (t,h,w); do NOT bf16-cast
    if pixel_values is not None and pixel_values.dtype == torch.float32:
        pixel_values = pixel_values.to(torch.bfloat16)  # match bf16 weights (vision tower)
    return tokens, labels, loss_mask, attention_mask, pixel_values, image_grid_thw, cu_seqlens, patch_positions


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: MegatronModule,
    return_schedule_plan: bool = False,
):
    """OV2.1 forward training step.

    Returns:
        (output_tensor, loss_function) where output_tensor is the per-token loss [b, s]
        returned by the model (labels provided) and loss_function reduces it with loss_mask.
    """
    timers = state.timers
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, pixel_values, image_grid_thw, cu_seqlens, patch_positions = get_batch(data_iterator)
    timers("batch-generator").stop()

    # THD block-diagonal packed attention when the task encoder emitted segment boundaries
    # (offline-packed data, e.g. PackedCaptioningSample). OV2_PACK_FULL_CAUSAL=1 instead runs ONE
    # full-causal sequence over the whole pack (later sub-samples see earlier ones).
    packed_seq_params = None
    if cu_seqlens is not None and os.environ.get("OV2_PACK_FULL_CAUSAL", "0") != "1":
        from megatron.core.packed_seq_params import PackedSeqParams
        cu = cu_seqlens.to(torch.int32).flatten()
        seq_len = int(tokens.shape[1])
        cu_last = int(cu[-1].item())
        if cu_last > seq_len:
            raise RuntimeError(
                f"OV2 packed cu_seqlens ends at {cu_last}, beyond token sequence length {seq_len}; "
                f"cu_seqlens={cu.tolist()}"
            )
        cu_padded = None
        if cu_last < seq_len:
            # Energon / SP padding can make the actual hidden sequence longer than the packed
            # sample payload. Represent the padded tail as a zero-length real segment plus
            # a nonzero padded segment, matching MCore's padded/unpadded THD convention.
            labels[:, cu_last:] = -100
            loss_mask[:, cu_last:] = 0
            cu_unpadded = torch.cat([cu, cu.new_tensor([cu_last])])
            cu_padded = torch.cat([cu, cu.new_tensor([seq_len])])
        else:
            cu_unpadded = cu
        cu_for_msl = cu_padded if cu_padded is not None else cu_unpadded
        seglens_for_msl = cu_for_msl[1:] - cu_for_msl[:-1]
        msl = int(seglens_for_msl.max().item()) if seglens_for_msl.numel() > 0 else int(tokens.shape[1])
        _packed_kwargs = dict(
            qkv_format="thd",
            cu_seqlens_q=cu_unpadded,
            cu_seqlens_kv=cu_unpadded,
            max_seqlen_q=msl,
            max_seqlen_kv=msl,
        )
        if cu_padded is not None:
            _packed_kwargs.update(cu_seqlens_q_padded=cu_padded, cu_seqlens_kv_padded=cu_padded)
        packed_seq_params = PackedSeqParams(**_packed_kwargs)

    # FLOPS metadata. Packed attention is O(Sum Li^2) over per-sub-sample lengths (block-diagonal),
    # NOT O((Sum Li)^2) -> feed the REAL Sum Li / Sum Li^2 so MODEL_TFLOP/s and the MFU line are not
    # inflated. Non-packed falls back to mbs * padded seq_len.
    if tokens is not None:
        mbs = tokens.shape[0]
        seq_len = tokens.shape[1]
        if cu_seqlens is not None:
            _cu_stats = cu_seqlens.flatten()
            _sl = (_cu_stats[1:] - _cu_stats[:-1]).to(torch.float64)
            state._flops_seqlen_sum = getattr(state, "_flops_seqlen_sum", 0) + int(_sl.sum().item())
            state._flops_seqlen_sq_sum = getattr(state, "_flops_seqlen_sq_sum", 0) + int((_sl * _sl).sum().item())
        else:
            state._flops_seqlen_sum = getattr(state, "_flops_seqlen_sum", 0) + mbs * seq_len
            state._flops_seqlen_sq_sum = getattr(state, "_flops_seqlen_sq_sum", 0) + mbs * seq_len**2
    if image_grid_thw is not None and image_grid_thw.numel() > 0:
        state._flops_vision_patches = getattr(state, "_flops_vision_patches", 0) + int(
            image_grid_thw.prod(dim=-1).sum().item()
        )

    output_tensor = model(
        images=pixel_values,
        image_grid_thw=image_grid_thw,
        input_ids=tokens,
        position_ids=None,        # LlavaOnevision2 / Qwen3 LLM compute positions internally
        attention_mask=attention_mask,
        labels=labels,            # PRE-SHIFTED by the task encoder; mcore does not shift
        patch_positions=patch_positions,       # video: temporal (t,h,w) RoPE; None for image (self-derived)
        packed_seq_params=packed_seq_params,   # THD block-diagonal when offline-packed (else None)
    )

    check_nan = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_spiky = state.cfg.rerun_state_machine.check_for_spiky_loss
    loss_function = _create_loss_function(loss_mask, check_nan, check_spiky)
    return output_tensor, loss_function
