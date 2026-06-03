# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bridge-native forward step for OV2.1 (LlavaOnevision2), used via run_recipe.py --step_func ov2_step.

LlavaOnevision2.forward(images, image_grid_thw, input_ids, labels=...) returns the LLM's
per-token loss [b, s] (mcore GPTModel.compute_language_model_loss, which does NOT shift labels).
Labels are therefore expected PRE-SHIFTED by the data task encoder (roll -1), exactly the
QwenVLTaskEncoder convention. This step does NOT shift labels and does NOT double-count.

Assumes TP/PP/CP = 1 (LlavaOnevision2.forward asserts CP==1) and no sequence packing — so unlike
qwen3_vl_step there is no pack_or_pad / cp-split here.
"""
import logging
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
    if pixel_values is not None and pixel_values.dtype == torch.float32:
        pixel_values = pixel_values.to(torch.bfloat16)  # match bf16 weights (vision tower)
    return tokens, labels, loss_mask, attention_mask, pixel_values, image_grid_thw


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
    tokens, labels, loss_mask, attention_mask, pixel_values, image_grid_thw = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(
        images=pixel_values,
        image_grid_thw=image_grid_thw,
        input_ids=tokens,
        position_ids=None,        # LlavaOnevision2 / Qwen3 LLM compute positions internally
        attention_mask=attention_mask,
        labels=labels,            # PRE-SHIFTED by the task encoder; mcore does not shift
    )

    check_nan = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_spiky = state.cfg.rerun_state_machine.check_for_spiky_loss
    loss_function = _create_loss_function(loss_mask, check_nan, check_spiky)
    return output_tensor, loss_function
