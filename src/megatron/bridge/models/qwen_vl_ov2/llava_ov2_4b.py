# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""LLaVA-OneVision-2.1 4B (p16m33) for Megatron-Bridge.

Faithful 3-sibling port of AIAK's ``LlavaOnevision2`` so the assembled AIAK
mcore checkpoint loads with near-identity naming:

    model.language_model  -- Qwen3-4B GPTModel  (built by Bridge AutoBridge)
    model.vision_model    -- OV2.1 OneVisionEncoderModel (patch16 / merge3)
    model.adapter         -- OV2.1 m33 Adapter (1024 -> 9216 -> 2560)

The build + checkpoint-stitch recipe in this file was VERIFIED to load
``llava_onevision2_4b_p16m33_mcore_tp1_pp1`` with 588/588 params, 0 missing,
0 unexpected (modulo ``._extra_state`` which TE rebuilds, and the
``patch_embed.proj`` Linear->Conv2d reshape).

Must run against the repo's ``3rdparty/Megatron-LM`` mcore (the OV2 vision
``layer_spec`` needs ``SelfAttentionSubmodules.apply_rotary_fn``).
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Optional

import torch
from megatron.core import tensor_parallel, parallel_state
from megatron.core.transformer.module import MegatronModule

logger = logging.getLogger(__name__)

# OV2.1 p16m33 vision-tower constants (from lmms-lab/LLaVA-OneVision-2-4B-p16m33).
OV2_PATCH_SIZE = 16
OV2_VISION_HIDDEN = 1024
OV2_VISION_LAYERS = 24
OV2_SPATIAL_MERGE = 3
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


def _fill_init(cfg, *, perform_init: bool = True):
    """Fill init_method helpers (AutoBridge(load_weights=False) leaves them None)."""
    from megatron.core.utils import init_method_normal, scaled_init_method_normal

    std = getattr(cfg, "init_method_std", 0.02) or 0.02
    if getattr(cfg, "init_method", None) is None:
        cfg.init_method = init_method_normal(std)
    if getattr(cfg, "output_layer_init_method", None) is None:
        cfg.output_layer_init_method = scaled_init_method_normal(std, cfg.num_layers)
    if hasattr(cfg, "perform_initialization"):
        cfg.perform_initialization = perform_init
    return cfg


def _vision_config_from(llm_cfg):
    """Build the OV2.1 p16m33 vision TransformerConfig by overlaying VisionConfig on the LLM cfg."""
    from megatron.bridge.models.qwen_vl_ov2 import get_vision_config

    vc = deepcopy(llm_cfg)
    for k, v in asdict(get_vision_config()).items():
        setattr(vc, k, v)
    vc.pipeline_model_parallel_size = 1
    vc.context_parallel_size = 1
    for f, d in (
        ("first_pipeline_num_layers", None),
        ("last_pipeline_num_layers", None),
        ("tp_comm_overlap", False),
        ("num_moe_experts", None),
        ("moe_router_topk", None),
        ("qk_layernorm", False),
        ("attention_output_gate", False),
    ):
        if hasattr(vc, f):
            setattr(vc, f, d)
    vc.patch_size = OV2_PATCH_SIZE
    vc.hidden_size = OV2_VISION_HIDDEN
    vc.num_layers = OV2_VISION_LAYERS
    return vc


def _adapter_config_from(llm_cfg):
    """Build the m33 adapter TransformerConfig: LayerNorm (weight+bias) + biased linears."""
    ac = deepcopy(llm_cfg)
    ac.normalization = "LayerNorm"
    ac.add_bias_linear = True
    # official/HF PatchMerger uses GELU (modeling_llava_onevision2.py:276-280); the deep-copied
    # Qwen3 cfg would otherwise leave SiLU + SwiGLU here. Force plain GELU, no gated/fused path.
    ac.activation_func = torch.nn.functional.gelu
    ac.gated_linear_unit = False
    ac.bias_activation_fusion = False
    return ac


class LlavaOnevision2(MegatronModule):
    """3-sibling OV2.1 multimodal model (language_model + vision_model + adapter).

    Forward ported from AIAK ``LlavaOnevision2`` (LLaVA-style masked_scatter merge
    of image embeddings into the text embedding stream at ``image_token_id``).
    Assumes PP=1 (pre_process and post_process both on this rank).
    """

    def __init__(self, language_model, vision_model, adapter, *, image_token_id: int = IMAGE_TOKEN_ID):
        super().__init__(config=language_model.config)
        self.language_model = language_model
        self.vision_model = vision_model
        self.adapter = adapter
        self.image_token_id = image_token_id
        self.share_embeddings_and_output_weights = getattr(
            language_model, "share_embeddings_and_output_weights", False
        )

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        # PP=1: the language model owns the embedding; nothing to wire from a prior stage.
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, *, freeze_language_model: bool, freeze_vision_model: bool, freeze_adapter: bool):
        mods = []
        if freeze_language_model:
            mods.append(self.language_model)
        if freeze_vision_model:
            mods.append(self.vision_model)
        if freeze_adapter:
            mods.append(self.adapter)
        for m in mods:
            for p in m.parameters():
                p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        packed_seq_params=None,
        patch_positions=None,
        **kwargs,
    ) -> torch.Tensor:
        # 1) vision -> adapter
        image_embeddings = None
        if images is not None:
            ve = self.vision_model(images, grid_thw=image_grid_thw, patch_positions=patch_positions)
            image_embeddings = self.adapter(ve)
            n_tok = (input_ids == self.image_token_id).sum().item()
            n_feat = image_embeddings.shape[0]
            if n_feat > n_tok:  # trim ViT SP-padding overflow
                image_embeddings = image_embeddings[:n_tok]
            elif n_feat < n_tok:
                raise ValueError(f"image features {n_feat} < image tokens {n_tok}")

        # 2) text embeddings + masked_scatter fuse
        language_embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None)
        if image_embeddings is None or self.image_token_id not in input_ids:
            combined = language_embeddings
        else:
            mask = (
                (input_ids == self.image_token_id)
                .transpose(0, 1)
                .unsqueeze(-1)
                .expand_as(language_embeddings)
                .to(language_embeddings.device)
            )
            image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
            combined = language_embeddings.masked_scatter(mask, image_embeddings)

        # 3) sequence-parallel scatter (pad to TP multiple) if enabled
        if getattr(self.config, "sequence_parallel", False):
            tp = parallel_state.get_tensor_model_parallel_world_size()
            rem = combined.size(0) % tp
            if rem:
                pad = combined.new_zeros((tp - rem,) + tuple(combined.shape[1:]))
                combined = torch.cat((combined, pad), dim=0)
            combined = tensor_parallel.scatter_to_sequence_parallel_region(combined)

        # 4) language model
        return self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined,
            labels=labels,
            packed_seq_params=packed_seq_params,
        )


def build_llava_ov2_4b(
    llm_hf_path: str = "/ov2/pretrain_models/Qwen3-4B-Instruct-2507",
    *,
    pre_process: bool = True,
    post_process: bool = True,
    perform_init: bool = True,
    use_cpu_init: bool = False,
    grad_accum_fusion: Optional[bool] = None,
    recompute: bool = False,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> LlavaOnevision2:
    """Build the OV2.1-4B model (untied Qwen3-4B LLM + OV2 p16m33 vision + m33 adapter).

    ``perform_init=False`` + ``use_cpu_init=True`` is the cheap structure-only build used
    for checkpoint-stitch validation (weights get overwritten by the ckpt load anyway).
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.qwen_vl_ov2 import (
        Adapter,
        OneVisionEncoderModel,
        get_adapter_layer_spec,
        get_vision_layer_spec,
    )

    prov = AutoBridge.from_hf_pretrained(llm_hf_path).to_megatron_provider(load_weights=False)
    prov.tensor_model_parallel_size = tensor_model_parallel_size
    prov.pipeline_model_parallel_size = pipeline_model_parallel_size
    prov.share_embeddings_and_output_weights = False  # OV2 ckpt has a separate output_layer.weight
    if hasattr(prov, "use_cpu_initialization"):
        prov.use_cpu_initialization = use_cpu_init
    # grad_accum_fusion=False is needed when running OUTSIDE the Megatron DDP wrapper
    # (e.g. structure/forward smokes) — TE wgrad fusion expects param.main_grad otherwise.
    # Leave None in real training so the framework default applies.
    if grad_accum_fusion is not None and hasattr(prov, "gradient_accumulation_fusion"):
        prov.gradient_accumulation_fusion = grad_accum_fusion
    if recompute:  # activation recompute for long-seq stage-2 (full/uniform/1)
        prov.recompute_granularity = "full"; prov.recompute_method = "uniform"; prov.recompute_num_layers = 1
    _fill_init(prov, perform_init=perform_init)
    language_model = prov.provide(pre_process=pre_process, post_process=post_process)
    llm_cfg = language_model.config

    vis_cfg = _fill_init(_vision_config_from(llm_cfg), perform_init=perform_init)
    adp_cfg = _fill_init(_adapter_config_from(llm_cfg), perform_init=perform_init)
    # Recompute applies to the LLM ONLY (seq 32000 × 36 layers dominates memory). vis_cfg
    # deepcopies llm_cfg, so explicitly DISABLE recompute on the vision encoder — its ported
    # _checkpointed_forward passes attn_mask_type, incompatible with this mcore's TransformerLayer
    # (the encoder is small: 24L / bounded patches, so it needn't recompute anyway).
    for _a in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
        if hasattr(vis_cfg, _a):
            setattr(vis_cfg, _a, None)
    if grad_accum_fusion is not None:
        for c in (vis_cfg, adp_cfg):
            if hasattr(c, "gradient_accumulation_fusion"):
                c.gradient_accumulation_fusion = grad_accum_fusion

    vision_model = OneVisionEncoderModel(vis_cfg, get_vision_layer_spec(), spatial_merge_size=OV2_SPATIAL_MERGE)
    adapter = Adapter(
        adp_cfg,
        get_adapter_layer_spec(),
        input_size=OV2_VISION_HIDDEN,
        output_size=llm_cfg.hidden_size,
        spatial_merge_size=OV2_SPATIAL_MERGE,
    )
    return LlavaOnevision2(language_model, vision_model, adapter)


def convert_hf_onevision_to_mcore(hf_sd, *, num_query_groups: int = 16, head_dim: int = 64, hidden: int = 1024) -> dict:
    """Convert a HF OneVisionEncoder state_dict -> Bridge mcore tower (vision_model submodule) keys.

    Mirrors AIAK's vision converter: per-group interleaved q/k/v fusion, layernorm fusion into
    the TE linears, patch_embed Conv2d kept 4-D. Head (probe pooling) is dropped (the tower uses
    patch features). VERIFIED key/shape compatible with the Bridge OneVisionEncoderModel.
    """
    ng, hd, D = num_query_groups, head_dim, hidden
    out = {"patch_embed.proj.weight": hf_sd["embeddings.patch_embedding.weight"],
           "pre_layernorm.weight": hf_sd["layernorm_pre.weight"],
           "pre_layernorm.bias": hf_sd["layernorm_pre.bias"]}
    if "layernorm_post.weight" in hf_sd:
        out["decoder.final_layernorm.weight"] = hf_sd["layernorm_post.weight"]
        out["decoder.final_layernorm.bias"] = hf_sd["layernorm_post.bias"]
    nl = len({k.split(".")[2] for k in hf_sd if k.startswith("encoder.layers.")})
    for i in range(nl):
        p, m = f"encoder.layers.{i}.", f"decoder.layers.{i}."
        qw = hf_sd[p + "self_attn.q_proj.weight"].view(ng, hd, D)
        kw = hf_sd[p + "self_attn.k_proj.weight"].view(ng, hd, D)
        vw = hf_sd[p + "self_attn.v_proj.weight"].view(ng, hd, D)
        out[m + "self_attention.linear_qkv.weight"] = torch.cat([qw, kw, vw], dim=1).reshape(3 * ng * hd, D)
        qb = hf_sd[p + "self_attn.q_proj.bias"].view(ng, hd)
        kb = hf_sd[p + "self_attn.k_proj.bias"].view(ng, hd)
        vb = hf_sd[p + "self_attn.v_proj.bias"].view(ng, hd)
        out[m + "self_attention.linear_qkv.bias"] = torch.cat([qb, kb, vb], dim=1).reshape(3 * ng * hd)
        out[m + "self_attention.linear_proj.weight"] = hf_sd[p + "self_attn.out_proj.weight"]
        out[m + "self_attention.linear_proj.bias"] = hf_sd[p + "self_attn.out_proj.bias"]
        out[m + "self_attention.linear_qkv.layer_norm_weight"] = hf_sd[p + "layer_norm1.weight"]
        out[m + "self_attention.linear_qkv.layer_norm_bias"] = hf_sd[p + "layer_norm1.bias"]
        out[m + "mlp.linear_fc1.layer_norm_weight"] = hf_sd[p + "layer_norm2.weight"]
        out[m + "mlp.linear_fc1.layer_norm_bias"] = hf_sd[p + "layer_norm2.bias"]
        out[m + "mlp.linear_fc1.weight"] = hf_sd[p + "mlp.fc1.weight"]
        out[m + "mlp.linear_fc1.bias"] = hf_sd[p + "mlp.fc1.bias"]
        out[m + "mlp.linear_fc2.weight"] = hf_sd[p + "mlp.fc2.weight"]
        out[m + "mlp.linear_fc2.bias"] = hf_sd[p + "mlp.fc2.bias"]
    return out


def load_hf_encoder_into_tower(tower, hf_path: str) -> dict:
    """Load a HF OneVisionEncoder (safetensors) into the Bridge vision tower (mcore)."""
    import glob, os
    from safetensors.torch import load_file
    f = sorted(glob.glob(os.path.join(hf_path, "*.safetensors")))[0]
    conv = convert_hf_onevision_to_mcore(load_file(f))
    missing, unexpected = tower.load_state_dict(conv, strict=False)
    missing = [k for k in missing if not k.endswith("._extra_state")]
    unexpected = [k for k in unexpected if not k.endswith("._extra_state")]
    return {"loaded": len(conv), "missing": missing, "unexpected": unexpected}


def _resolve_ckpt_file(p: str) -> str:
    import os
    if os.path.isfile(p):
        return p
    for cand in (os.path.join(p, "release", "mp_rank_00", "model_optim_rng.pt"),):
        if os.path.exists(cand):
            return cand
    latest = os.path.join(p, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest):
        with open(latest) as fh:
            it = fh.read().strip()
        cand = os.path.join(p, f"iter_{int(it):07d}", "model_optim_rng.pt")
        if os.path.exists(cand):
            return cand
    return p


def load_ov2_4b_mcore_checkpoint(model: LlavaOnevision2, ckpt_path: str, *, load_adapter: bool = True,
                                 load_vision: bool = True) -> dict:
    """Stitch-load the assembled AIAK mcore ckpt into the model.

    Args:
        ckpt_path: dir containing ``release/mp_rank_00/model_optim_rng.pt`` (or that file).
        load_adapter: if False, leave the adapter at its (random) init — stage-1 trains it fresh.

    VERIFIED: 588/588 params, 0 missing / 0 unexpected for the p16m33 4B ckpt.
    """
    import os

    p = ckpt_path
    if os.path.isdir(p):
        cand = os.path.join(p, "release", "mp_rank_00", "model_optim_rng.pt")
        p = cand if os.path.exists(cand) else p
    blob = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    sd = {k: v for k, v in blob["model"].items() if not k.endswith("._extra_state")}

    # patch_embed: ckpt Linear [out, 3*P*P] -> model Conv2d [out, 3, P, P]
    msd = model.state_dict()
    pe = "vision_model.patch_embed.proj.weight"
    if pe in sd and pe in msd:
        dst = tuple(msd[pe].shape)
        if tuple(sd[pe].shape) != dst and sd[pe].numel() == int(torch.tensor(list(dst)).prod()):
            sd[pe] = sd[pe].reshape(dst)

    if not load_adapter:
        sd = {k: v for k, v in sd.items() if not k.startswith("adapter.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [
        k for k in missing
        if not k.endswith("._extra_state") and not (not load_adapter and k.startswith("adapter."))
    ]
    real_unexpected = [k for k in unexpected if not k.endswith("._extra_state")]
    summary = {
        "loaded": len(sd),
        "missing": real_missing,
        "unexpected": real_unexpected,
        "adapter_loaded": load_adapter,
    }
    logger.info(
        "[ov2_4b load] loaded=%d missing=%d unexpected=%d adapter_loaded=%s",
        len(sd), len(real_missing), len(real_unexpected), load_adapter,
    )
    return summary
