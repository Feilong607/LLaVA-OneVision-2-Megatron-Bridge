# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OV2 recipes for the LITERAL Qwen3.5-35B-A3B (model_type ``qwen3_5_moe_text``: GatedDeltaNet
hybrid + 256-expert MoE + MTP).

SEPARATE from ``ov2.py``'s Qwen3-30B-A3B stack — the misnamed ``ov2_35b_a3b_*`` recipes there are a
128-expert Qwen3-30B-A3B, NOT this 256-expert base. This module is additive: it registers the
``qwen3.5-35b-a3b`` backbone into ``ov2._OV2_BACKBONES`` at import and exposes its own recipe
functions; it does NOT modify any qwen3 entry.

Reuses the LLM-agnostic ``build_llava_ov2`` + ``_ov2_common``. Verified gates (A100-18):
  * LLM build via AutoBridge: GPTModel 35.51B, GatedDeltaNet=30 + MTP intact
  * OneVisionEncoder p16m33 stitch: vision 303M + adapter 104M (out dim 2048) onto the qwen3.5 LLM

LLM weights come from the TEXT-only HF dir produced by ``qwen35_vl_ov2/tools/extract_qwen35_text.py``
(``architectures=Qwen3_5MoeForCausalLM`` -> AutoBridge routes to the ``qwen3_5_moe_text`` bridge, so
the OneVision tower replaces Qwen3.5's native vision tower). Run that tool with ``--weights`` before
real training.
"""
import os

from megatron.bridge.training.config import ConfigContainer

from .ov2 import _OV2_BACKBONES, _OV2_PRETRAIN_ROOT, _ov2_common

_QWEN35_BACKBONE = "qwen3.5-35b-a3b"

# --- register the qwen3.5-35b-a3b p16m33 backbone ADDITIVELY (ov2.py / qwen3 entries untouched) ---
_OV2_BACKBONES.setdefault(
    _QWEN35_BACKBONE,
    {
        "is_moe": True,
        # Qwen3.5 <|image_pad|> = 248056. NOT 151655 (that is a NORMAL text token in Qwen3.5;
        # using it would scatter image features onto real text -> mismatch/corruption).
        "image_token_id": 248056,
        "adapter_init_scale": 0.0184,   # fc2 init rescale: Qwen3.5 emb L2 ~0.48 vs adapter ~26 (54x); 0.477/25.9. 4B/30B omit -> 1.0
        # M-RoPE: Qwen3.5 was pretrained with multimodal RoPE (2D-grid t/h/w positions for vision). The
        # text bridge forces 1D "rope", so the frozen LLM can't localize visual tokens -> stage-1 plateau.
        # Build the LLM as mrope with this section so get_rope_index's [3,b,s] positions take effect.
        # 4B/30B omit -> stay 1D rope (their native pretraining). Matches native qwen35_vl_provider.
        "mrope_section": [11, 11, 10],
        # text-only HF dir extracted from the Qwen3.5-35B-A3B VLM (run extract_qwen35_text.py --weights).
        "llm_hf": os.environ.get("OV2_LLM_HF_QWEN35", f"{_OV2_PRETRAIN_ROOT}/Qwen3.5-35B-A3B-text"),
        # stage_0 combined base = qwen3.5 text LLM + OneVision p16m33 tower + fresh merge3 adapter
        # (torch_dist, EP8). Built via the qwen35_vl_ov2 convert step; no aligned OV2 ckpt -> stage-1.
        "mcore_ckpt": os.environ.get(
            "OV2_MCORE_QWEN35_P16M33",
            f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8",
        ),
        # p16m33 processor with the Qwen3.5 tokenizer. Its <|image_pad|>=248056 MUST match the model's
        # image_token_id=248056 (set above) so the dataloader scatters image features at the right positions.
        "hf_proc": os.environ.get(
            "OV2_HF_PROC_QWEN35_P16M33",
            f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model",
        ),
        # p16m33 OneVisionEncoder (same tower class as qwen3-30b-a3b-p16m33). adapter merges 1024*3^2=9216 -> 2048.
        "vision_patch_size": 16,
        "vision_spatial_merge_size": 3,
        "vision_hidden_size": 1024,
        "vision_num_layers": 24,
        "vision_model_name": None,
        "stage1_ckpt": None,
    },
)


def ov2_qwen35_35b_a3b_stage1() -> ConfigContainer:
    """Qwen3.5-35B-A3B OV2 stage-1 adapter-only alignment (LLM + vision FROZEN, AdamW, EP8, p16m33)."""
    return _ov2_common(_QWEN35_BACKBONE, "stage1", expert_model_parallel_size=8, sequence_parallel=False)


def ov2_qwen35_35b_a3b_stage2() -> ConfigContainer:
    """Qwen3.5-35B-A3B OV2 stage-2 vit+adapter SFT (EP8). Chain from a trained stage-1 via CLI."""
    return _ov2_common(_QWEN35_BACKBONE, "stage2", expert_model_parallel_size=8, sequence_parallel=False)


def ov2_qwen35_35b_a3b_midtrain() -> ConfigContainer:
    """Qwen3.5-35B-A3B OV2 mid-train full-model SFT (EP8). MTP kept (mtp_loss 0.1 via qwen35_bridge)."""
    return _ov2_common(_QWEN35_BACKBONE, "midtrain", expert_model_parallel_size=8, sequence_parallel=False)


__all__ = [
    "ov2_qwen35_35b_a3b_stage1",
    "ov2_qwen35_35b_a3b_stage2",
    "ov2_qwen35_35b_a3b_midtrain",
]
