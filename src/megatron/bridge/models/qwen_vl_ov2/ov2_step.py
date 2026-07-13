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
import math
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


def _ov2_fp8_pad_mult(state) -> int:
    """Token-dim alignment the packed sequence needs so TE fp8/MXFP8 GEMMs stay engaged.

    Returns 1 for bf16 runs (ACCEL=0/2) so the pad block is byte-identical to the TP-only pad.
    Returns 32 (override: OV2_FP8_PAD_MULT) for fp8/fp4 runs (ACCEL=1, mixed_precision.fp8 set):
    the TE crash assert_dim_for_fp8_exec only needs the token dim %8 (last dim %16), but MXFP8 uses
    1x32 block scaling and in the weight-gradient GEMM the token dim is the contraction dim, so
    MXFP8Quantizer.is_quantizable requires the token dim %32 or the GEMM silently falls back to bf16.
    OV2_FP8_SEQ_PAD=1/0 force-enables/disables regardless of the resolved precision.
    """
    _ovr = os.environ.get("OV2_FP8_SEQ_PAD")
    if _ovr is not None:
        _on = _ovr == "1"
    else:
        mp = getattr(getattr(state, "cfg", None), "mixed_precision", None)
        _lp = getattr(mp, "fp8", None) or getattr(mp, "fp4", None)  # resolved MixedPrecisionConfig
        if _lp is not None:
            _on = bool(_lp)
        elif isinstance(mp, str):  # defensive: pre-resolution registry key (e.g. MIMO path)
            _on = ("fp8" in mp) or ("fp4" in mp)
        else:
            _on = False
    if not _on:
        return 1
    try:
        return max(1, int(os.environ.get("OV2_FP8_PAD_MULT", "32")))
    except ValueError:
        return 32


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

    # SP needs the token dim to be a multiple of TP. seed85m packs to VARIABLE (often ODD) lengths,
    # so pad tokens/labels/loss_mask to a TP multiple HERE -> the in-model SP scatter (llava_ov2 step 3)
    # becomes a no-op and logits/labels/loss_mask stay length-consistent. Without this, SP lengthens the
    # hidden states (e.g. 8147->8148) but NOT labels/loss_mask, so fused vocab-parallel cross-entropy
    # hits an off-by-one (logits N+1 vs labels N). The pad tail is masked just below by the existing
    # `cu_last < seq_len` branch (labels=-100, loss_mask=0) + the cu_*_padded twins, so the loss is
    # identical to the unpadded sequence. Dormant when TP=1 or packs are already a TP multiple.
    _unpadded_seq_len = tokens.shape[1] if tokens is not None else 0   # capture BEFORE SP-pad so the FLOP non-packed branch uses the TRUE len
    if tokens is not None:
        from megatron.core import parallel_state
        _tp = parallel_state.get_tensor_model_parallel_world_size()
        # Base alignment: SP scatter needs a TP multiple. FP8/MXFP8 (ACCEL=1) ALSO needs the packed
        # token dim aligned or TE's fp8 GEMMs fail assert_dim_for_fp8_exec (crash); and for MXFP8's
        # 1x32 block scaling the wgrad GEMM (token dim = contraction) silently drops to bf16 unless
        # the token dim is a multiple of 32. _ov2_fp8_pad_mult(state) returns 1 for bf16 (ACCEL=0/2)
        # -> lcm(_tp, 1) == _tp -> byte-identical to the old TP-only pad; 32 for fp8/fp4 (ACCEL=1).
        _pad_mult = math.lcm(_tp, _ov2_fp8_pad_mult(state))
        if _pad_mult > 1 and tokens.shape[1] % _pad_mult:
            _pad = _pad_mult - tokens.shape[1] % _pad_mult
            tokens = torch.nn.functional.pad(tokens, (0, _pad), value=0)
            if labels is not None:    labels = torch.nn.functional.pad(labels, (0, _pad), value=-100)
            if loss_mask is not None: loss_mask = torch.nn.functional.pad(loss_mask, (0, _pad), value=0)

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
            # sample payload. FOLD the pad into the LAST real segment padded span (cu_padded last =
            # seq_len); keep cu_unpadded = real boundaries. Same N segs, last seg actual<=padded, NO
            # zero-length segment (a 0-actual THD seg -> flash-attn backward 0/0 -> NaN grad; only bites
            # when the LLM trains). Pad tokens TE-masked + label-masked -> loss identical.
            if labels is not None:    labels[:, cu_last:] = -100
            if loss_mask is not None: loss_mask[:, cu_last:] = 0
            cu_unpadded = cu
            cu_padded = torch.cat([cu[:-1], cu.new_tensor([seq_len])])
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
            # use the PRE-SP-pad length (SP-pad inflated tokens.shape[1]); matches the packed branch's unpadded cu_seqlens
            state._flops_seqlen_sum = getattr(state, "_flops_seqlen_sum", 0) + mbs * _unpadded_seq_len
            state._flops_seqlen_sq_sum = getattr(state, "_flops_seqlen_sq_sum", 0) + mbs * _unpadded_seq_len**2
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
        loss_mask=loss_mask,      # threaded so the MTP head masks aux loss like the main loss (midtrain)
        patch_positions=patch_positions,       # video: temporal (t,h,w) RoPE; None for image (self-derived)
        packed_seq_params=packed_seq_params,   # THD block-diagonal when offline-packed (else None)
    )

    check_nan = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_spiky = state.cfg.rerun_state_machine.check_for_spiky_loss
    loss_function = _create_loss_function(loss_mask, check_nan, check_spiky)
    return output_tensor, loss_function
