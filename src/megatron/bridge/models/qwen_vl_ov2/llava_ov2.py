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
import torch.distributed as dist
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
    # Newer mcore separates embedding_init_method (and friends) from init_method; the
    # provider __post_init__ ran with init_method=None (load_weights=False) so they stayed
    # None. Default any remaining *_init_method field to init_method (output_layer keeps its
    # scaled variant set above). Without this, VocabParallelEmbedding gets init_method=None.
    for _name in dir(cfg):
        if _name.endswith("_init_method") and _name != "output_layer_init_method":
            if getattr(cfg, _name, None) is None:
                try:
                    setattr(cfg, _name, cfg.init_method)
                except Exception:
                    pass
    if hasattr(cfg, "perform_initialization"):
        cfg.perform_initialization = perform_init
    return cfg


def _vision_config_from(
    llm_cfg,
    *,
    patch_size: int = OV2_PATCH_SIZE,
    vision_hidden_size: int = OV2_VISION_HIDDEN,
    vision_num_layers: int = OV2_VISION_LAYERS,
    vision_model_name: Optional[str] = None,
):
    """Build the OV2 vision TransformerConfig by overlaying VisionConfig on the LLM cfg.

    Defaults reproduce the VERIFIED 4B p16m33 tower (patch16 / hidden1024 / 24L) exactly. Per-backbone
    callers pass patch_size / vision_hidden_size / vision_num_layers (and, for the larger 35B-A3B
    ``vision-2b`` tower, vision_model_name="...vision-2b...") so the tower geometry follows the
    backbone instead of being hardcoded to patch16.

    ``vision_model_name`` selects the named base/variant geometry inside get_vision_config (kv_channels,
    ffn_hidden_size, num_attention_heads). The explicit patch/hidden/layers args are then applied on
    top via get_vision_config's keyword overrides AND re-pinned below (kept for byte-identical 4B
    parity and to override any field the deepcopied llm_cfg would otherwise leak).
    """
    from megatron.bridge.models.qwen_vl_ov2 import get_vision_config

    # When vision_model_name is None, fall back to get_vision_config's own default model_name so the
    # 4B path's asdict(get_vision_config()) overlay is byte-identical to before.
    gv_kwargs = dict(
        patch_size=patch_size,
        hidden_size=vision_hidden_size,
        num_layers=vision_num_layers,
    )
    if vision_model_name is not None:
        gv_kwargs["model_name"] = vision_model_name
    vc = deepcopy(llm_cfg)
    for k, v in asdict(get_vision_config(**gv_kwargs)).items():
        # spatial_merge_size is a VisionConfig-only bookkeeping field (the tower's
        # TransformerConfig/TransformerBlock does not use it; merge is applied by the Adapter), so do
        # NOT stamp it onto the TransformerConfig — keeps the overlaid cfg identical to the original.
        if k == "spatial_merge_size":
            continue
        setattr(vc, k, v)
    vc.pipeline_model_parallel_size = 1
    vc.context_parallel_size = 1
    for f, d in (
        ("first_pipeline_num_layers", None),
        ("last_pipeline_num_layers", None),
        ("tp_comm_overlap", False),
        ("expert_model_parallel_size", 1),
        ("expert_tensor_parallel_size", 1),
        ("tensor_model_parallel_size", 1),
        ("sequence_parallel", False),
        ("num_moe_experts", None),
        ("moe_router_topk", None),
        ("qk_layernorm", False),
        ("attention_output_gate", False),
    ):
        if hasattr(vc, f):
            setattr(vc, f, d)
    vc.patch_size = patch_size
    vc.hidden_size = vision_hidden_size
    vc.num_layers = vision_num_layers
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
    # Dense PatchMerger MLP: do not let the deep-copied MoE-LLM cfg leak expert/parallel knobs onto it
    # (inert at TP1/EP8 today since the adapter spec is dense, but a landmine if TP/EP ever change).
    for _f, _d in (("num_moe_experts", None), ("moe_router_topk", None),
                   ("expert_model_parallel_size", 1), ("expert_tensor_parallel_size", 1),
                   ("tensor_model_parallel_size", 1), ("sequence_parallel", False),
                   ("pipeline_model_parallel_size", 1), ("context_parallel_size", 1)):
        if hasattr(ac, _f):
            setattr(ac, _f, _d)
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
        # This forward assumes context_parallel_size==1 (no get_inputs_on_this_cp_rank split).
        assert getattr(language_model.config, "context_parallel_size", 1) == 1, (
            "LlavaOnevision2.forward assumes context_parallel_size==1; add a CP split before enabling CP."
        )
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
        # Image-feature / image-token validity is checked COLLECTIVELY (see the all_reduce below) so a
        # malformed microbatch aborts the whole job in lockstep instead of deadlocking. A per-rank raise
        # inside this forward would hang every downstream rendezvous: first the MoE expert-parallel
        # all-to-all in language_model(), then the gradient all-reduce -- because the offending rank
        # never arrives. So we (a) never raise per-rank here, and (b) reduce over the WHOLE job, not just
        # the EP group (an EP-only abort would still hang the surviving DP replicas at the grad all-reduce).
        _feat_bad = 0
        _n_feat = _n_tok = -1
        _sp = bool(getattr(self.config, "sequence_parallel", False))
        if images is not None:
            ve = self.vision_model(images, grid_thw=image_grid_thw, patch_positions=patch_positions)
            image_embeddings = self.adapter(ve)
            _n_tok = (input_ids == self.image_token_id).sum().item()
            _n_feat = image_embeddings.shape[0]
            # SP off: no ViT SP-padding -> features must match tokens EXACTLY.
            # SP on : ViT SP-padding makes n_feat >= n_tok; overflow is trimmed, a shortfall is a bug.
            if not _sp:
                _feat_bad = 1 if _n_feat != _n_tok else 0
            else:
                _feat_bad = 1 if _n_feat < _n_tok else 0
                if _n_feat > _n_tok:  # trim ViT SP-padding overflow (only expected under sequence_parallel)
                    image_embeddings = image_embeddings[:_n_tok]
        # Collective abort: EVERY rank participates -- including text-only ranks where images is None and
        # _feat_bad stays 0 -- so the all_reduce itself can never desync. MAX => if ANY rank saw a
        # mismatch, ALL ranks raise the same RuntimeError at this identical program point.
        if dist.is_available() and dist.is_initialized():
            _flag = torch.tensor([_feat_bad], dtype=torch.int32, device=input_ids.device)
            dist.all_reduce(_flag, op=dist.ReduceOp.MAX)
            _feat_bad = int(_flag.item())
        if _feat_bad:
            _local_mismatch = "yes" if (_n_feat >= 0 and _n_feat != _n_tok) else "no"
            raise RuntimeError(
                "OV2 image feature/token mismatch (collective abort across all ranks). "
                f"This rank: n_feat={_n_feat} n_tok={_n_tok} sp={_sp} local_mismatch={_local_mismatch}. "
                "At least one rank's image features did not match its <image> placeholders; the whole job "
                "aborts in lockstep to avoid an expert-parallel all-to-all / gradient all-reduce deadlock. "
                "Inspect the processor/grid (image_grid_thw vs inserted image tokens) on the failing rank."
            )

        # 2) text embeddings + masked_scatter fuse
        language_embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None)
        if image_embeddings is None or self.image_token_id not in input_ids:
            combined = language_embeddings
        else:
            # masked_scatter fills True positions row-major over [seq, batch, hidden]; for batch>1 this
            # interleaves samples incorrectly (image_embeddings are concatenated per-sample). OV2 trains
            # at micro_batch_size==1 everywhere -> assert it so a future mbs>1 fails loud, not silent-wrong.
            assert language_embeddings.size(1) == 1, (
                f"OV2 masked_scatter fuse assumes micro_batch_size==1, got batch={language_embeddings.size(1)}; "
                "scatter per-sample before raising mbs."
            )
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


def build_llava_ov2(
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
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    sequence_parallel: bool = False,
    load_llm_weights: bool = False,
    # Per-backbone vision-tower geometry. Defaults == the VERIFIED 4B p16m33 tower, so an
    # un-parameterized call (and the 4B recipe) build the exact same vision_model/adapter as before.
    patch_size: int = OV2_PATCH_SIZE,
    spatial_merge_size: int = OV2_SPATIAL_MERGE,
    vision_hidden_size: int = OV2_VISION_HIDDEN,
    vision_num_layers: int = OV2_VISION_LAYERS,
    vision_model_name: Optional[str] = None,
) -> LlavaOnevision2:
    """Build the OV2 model (untied Qwen3-family LLM + OV2 vision tower + adapter).

    LLM-agnostic: the LLM config (dense vs. MoE, width, layers, hybrid GDN attention) flows
    entirely from the HF dir via ``AutoBridge`` — only ``llm_hf_path`` selects the backbone.
    The adapter ``output_size`` is set to ``llm_cfg.hidden_size``, so it auto-adapts to any LLM
    width. For an MoE backbone (e.g. Qwen3.5-35B-A3B, model_type ``qwen3_5_moe``) the bridge sets
    ``transformer_layer_spec`` to the hybrid GDN+MoE experimental-attention spec and the MoE
    config (num_experts / topk / shared-expert / grouped-gemm / dispatcher) on the provider, so
    ``prov.provide()`` builds the correct MoE language model. Expert-parallel / sequence-parallel
    knobs are propagated to the LLM provider below (they are no-ops for dense backbones).

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

    bridge = AutoBridge.from_hf_pretrained(llm_hf_path)
    # Always build the LLM with load_weights=False (the random/cpu-init path known to work
    # with the bare prov.provide() below). The load_weights=True provider path loads HF
    # weights only via a pre_wrap_hook fired by provide_distributed_model()/get_model() --
    # NOT by prov.provide() -- so HF LLM weights are streamed in explicitly after provide().
    prov = bridge.to_megatron_provider(load_weights=False)
    prov.tensor_model_parallel_size = tensor_model_parallel_size
    prov.pipeline_model_parallel_size = pipeline_model_parallel_size
    # Expert/sequence parallel for MoE backbones (no-op fields for dense). Only override
    # expert_tensor_parallel_size when explicitly requested; otherwise leave the bridge default.
    if hasattr(prov, "expert_model_parallel_size"):
        prov.expert_model_parallel_size = expert_model_parallel_size
    if expert_tensor_parallel_size is not None and hasattr(prov, "expert_tensor_parallel_size"):
        prov.expert_tensor_parallel_size = expert_tensor_parallel_size
    if hasattr(prov, "sequence_parallel"):
        prov.sequence_parallel = sequence_parallel
    # MoE router numerical stability: with many experts (Qwen3-30B-A3B has 128) the routing
    # softmax/top-k in bf16 can overflow to NaN (mcore warns: ">=32 experts without fp32 routing").
    # Observed: OV2-30B-A3B stage-1 NaN'd at iter 2 on all ranks. fp32 routing is the fix.
    if getattr(prov, "num_moe_experts", None):
        prov.moe_router_dtype = "fp32"
    prov.share_embeddings_and_output_weights = False  # OV2 ckpt has a separate output_layer.weight
    # AIAK sets scatter_embedding_sequence_parallel=False: the VLM does its own SP scatter on the
    # fused (text+image) embeddings (see forward step 3), so the LLM embedding must NOT pre-scatter.
    # No-op at TP=1; prevents double-scatter / layout corruption when TP>1.
    if hasattr(prov, "scatter_embedding_sequence_parallel"):
        prov.scatter_embedding_sequence_parallel = False
    if hasattr(prov, "use_cpu_initialization"):
        prov.use_cpu_initialization = use_cpu_init
    # grad_accum_fusion=False is needed when running OUTSIDE the Megatron DDP wrapper
    # (e.g. structure/forward smokes) — TE wgrad fusion expects param.main_grad otherwise.
    # Leave None in real training so the framework default applies.
    if grad_accum_fusion is not None and hasattr(prov, "gradient_accumulation_fusion"):
        prov.gradient_accumulation_fusion = grad_accum_fusion
    if recompute:  # activation recompute. stage-2 (LLM/experts FROZEN) -> SELECTIVE core_attn:
        # recomputing the whole frozen MoE block (the old full/uniform/1) is wasted compute, so only
        # recompute the memory-heavy/compute-light attention. midtrain (LLM UNFROZEN, tight memory at
        # seq 32000) can keep full/uniform/1 via OV2_RECOMPUTE_FULL=1. NB: selective REQUIRES
        # recompute_method/recompute_num_layers = None (TransformerConfig.__post_init__ raises otherwise).
        import os
        if os.environ.get("OV2_RECOMPUTE_FULL", "0") == "1":
            prov.recompute_granularity = "full"; prov.recompute_method = "uniform"; prov.recompute_num_layers = 1
        else:
            prov.recompute_granularity = "selective"; prov.recompute_modules = ["core_attn"]
            prov.recompute_method = None; prov.recompute_num_layers = None
    # moe_permute_fusion: force OFF on the BUILT LLM config — set on `prov` BEFORE prov.provide()
    # (the reliable spot, exactly like recompute above). The post-build model.config force-set in
    # provide() does NOT reach the built MoE layers (verified: it still dumps True). Disables the TE
    # Triton-JIT fused-permute kernel that intermittently wedges OV2-30B-A3B. Re-enable with
    # OV2_MOE_PERMUTE_FUSION=1.
    if hasattr(prov, "moe_permute_fusion"):
        import os
        prov.moe_permute_fusion = os.environ.get("OV2_MOE_PERMUTE_FUSION", "0") == "1"
    _fill_init(prov, perform_init=perform_init)
    language_model = prov.provide(pre_process=pre_process, post_process=post_process)
    # moe_permute_fusion override (MUST be post-provide): the qwen3_(5_)moe bridge's provider_bridge()
    # re-sets moe_permute_fusion=True DURING prov.provide(), clobbering any pre-provide set. Override on
    # the BUILT config here — language_model.config is the runtime TransformerConfig the MoE token
    # dispatcher reads. Default OFF (fixes the TE Triton-JIT fused-permute wedge on OV2-30B/35B-A3B);
    # set OV2_MOE_PERMUTE_FUSION=1 to re-enable for A/B. Bridge-agnostic + unconditional (no num_experts
    # guard) so it always lands; no-op on dense backbones (field just flips False->False).
    if hasattr(language_model.config, "moe_permute_fusion"):
        import os as _os
        language_model.config.moe_permute_fusion = _os.environ.get("OV2_MOE_PERMUTE_FUSION", "0") == "1"
    if load_llm_weights:  # stream HF Qwen3-4B weights into the freshly-built LLM (cpu params)
        bridge.load_hf_weights([language_model])
        # Qwen3-4B-Instruct ties input/output embeddings (no lm_head.weight in HF), but the
        # OV2 mcore model is untied (share_embeddings_and_output_weights=False), so the
        # output_layer was left at random init by load_hf_weights. Mirror AIAK: duplicate the
        # input embedding into output_layer so the stitched ckpt is byte-identical to AIAK's.
        _hf_cfg = getattr(getattr(bridge, "hf_pretrained", None), "config", None)
        if getattr(_hf_cfg, "tie_word_embeddings", False):
            _emb = getattr(getattr(language_model, "embedding", None), "word_embeddings", None)
            _out = getattr(language_model, "output_layer", None)
            if _emb is not None and _out is not None:
                with torch.no_grad():
                    _out.weight.copy_(_emb.weight)
    llm_cfg = language_model.config

    vis_cfg = _fill_init(
        _vision_config_from(
            llm_cfg,
            patch_size=patch_size,
            vision_hidden_size=vision_hidden_size,
            vision_num_layers=vision_num_layers,
            vision_model_name=vision_model_name,
        ),
        perform_init=perform_init,
    )
    adp_cfg = _fill_init(_adapter_config_from(llm_cfg), perform_init=perform_init)
    # Recompute applies to the LLM ONLY (seq 32000 × 36 layers dominates memory). vis_cfg
    # deepcopies llm_cfg, so explicitly DISABLE recompute on the vision encoder — its ported
    # _checkpointed_forward passes attn_mask_type, incompatible with this mcore's TransformerLayer
    # (the encoder is small: 24L / bounded patches, so it needn't recompute anyway).
    for _a in ("recompute_granularity", "recompute_method", "recompute_num_layers", "recompute_modules"):
        if hasattr(vis_cfg, _a):
            setattr(vis_cfg, _a, None)
    if grad_accum_fusion is not None:
        for c in (vis_cfg, adp_cfg):
            if hasattr(c, "gradient_accumulation_fusion"):
                c.gradient_accumulation_fusion = grad_accum_fusion

    vision_model = OneVisionEncoderModel(vis_cfg, get_vision_layer_spec(), spatial_merge_size=spatial_merge_size)
    adapter = Adapter(
        adp_cfg,
        get_adapter_layer_spec(),
        input_size=vision_hidden_size,           # adapter merges input_size * merge^2 (4B 1024*9=9216; 8B 1024*4=4096)
        output_size=llm_cfg.hidden_size,         # auto-sizes to ANY LLM width (4B=2560, 8B=4096, 35B-A3B=2048)
        spatial_merge_size=spatial_merge_size,
    )
    return LlavaOnevision2(language_model, vision_model, adapter)


# Back-compat alias: the 4B-specific name kept so older callers / running scripts still import it.
# The builder has always been LLM-agnostic (adapter output_size = llm_cfg.hidden_size); only the
# default llm_hf_path points at 4B. New code should call build_llava_ov2.
build_llava_ov2_4b = build_llava_ov2


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


def load_ov2_mcore_checkpoint(model: LlavaOnevision2, ckpt_path: str, *, load_adapter: bool = True,
                              load_vision: bool = True) -> dict:
    """Stitch-load the assembled AIAK mcore ckpt into the model.

    Args:
        ckpt_path: dir containing ``release/mp_rank_00/model_optim_rng.pt`` (or that file).
        load_adapter: if False, leave the adapter at its (random) init — stage-1 trains it fresh.

    Backbone-agnostic loader: ``strict=False`` matches whatever sibling-prefixed keys
    (``language_model.`` / ``vision_model.`` / ``adapter.``) exist in the ckpt. VERIFIED 588/588
    params, 0 missing / 0 unexpected for the p16m33 4B ckpt; 8B / 35B-A3B ckpts have a different
    language_model param count and are NOT yet verified here (flagged in the report).
    """
    import os, glob

    p = ckpt_path
    if os.path.isdir(p):
        rel = os.path.join(p, "release")
        # Single-chunk (4B / 8B): release/mp_rank_00/model_optim_rng.pt
        single = os.path.join(rel, "mp_rank_00", "model_optim_rng.pt")
        # EP-sharded (35B-A3B MoE): release/mp_rank_00_{ep:03d}/model_optim_rng.pt — one shard per
        # expert-parallel rank. Verified layout (stage_0_tp1_pp1_ep8): every shard replicates the
        # non-expert weights (attention/router/embed/norm) + vision_model (291) + adapter (6); each
        # shard carries ONLY its own 16 local_experts (indexed locally 0..15). The model built with
        # expert_model_parallel_size=8 also exposes local_experts.0..15 per rank, so THIS rank's
        # experts match by index. (The ckpt is PER-EXPERT, SequentialMLP-style keys, while the built
        # model is TE-grouped; the keys are remapped per-expert -> grouped below, before load.)
        ep_shards = sorted(glob.glob(os.path.join(rel, "mp_rank_00_[0-9][0-9][0-9]")))
        if os.path.exists(single):
            p = single
        elif ep_shards:
            try:
                from megatron.core import parallel_state as _ps
                ep_rank = _ps.get_expert_model_parallel_rank()
                ep_size = _ps.get_expert_model_parallel_world_size()
            except Exception:
                ep_rank, ep_size = 0, len(ep_shards)
            if len(ep_shards) != ep_size:
                # FATAL (was a warning): index-based ep_rank->shard mapping would load the WRONG experts
                # / drop half of them at EP!=8 (silent corruption -> loss/grad blow up). The AIAK base is
                # EP8; the only safe path is EP8 (>=2 GB200 nodes), or from_base at EP8 then `reshard`.
                raise SystemExit(
                    "[ov2 load] EP-sharded base has {} shards but EP world size is {} -> each rank would "
                    "load the WRONG experts (silent corruption). Build/run at expert_model_parallel_size={} "
                    "(EP8 on >=2 GB200 nodes), or from_base at EP8 then reshard to the target EP.".format(
                        len(ep_shards), ep_size, len(ep_shards)))
            cand = os.path.join(rel, f"mp_rank_00_{ep_rank:03d}", "model_optim_rng.pt")
            p = cand if os.path.exists(cand) else (ep_shards[ep_rank] if ep_rank < len(ep_shards) else single)
            logger.info("[ov2 load] EP-sharded ckpt: ep_rank=%d/%d -> %s", ep_rank, ep_size, p)
    blob = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    sd = {k: v for k, v in blob["model"].items() if not k.endswith("._extra_state")}

    # patch_embed: ckpt Linear [out, 3*P*P] -> model Conv2d [out, 3, P, P]
    msd = model.state_dict()
    pe = "vision_model.patch_embed.proj.weight"
    if pe in sd and pe in msd:
        dst = tuple(msd[pe].shape)
        if tuple(sd[pe].shape) != dst and sd[pe].numel() == int(torch.tensor(list(dst)).prod()):
            sd[pe] = sd[pe].reshape(dst)

    # MoE expert-key reconciliation. The legacy EP ckpt stores experts PER-EXPERT (SequentialMLP):
    #   ...mlp.experts.local_experts.{i}.linear_fc{1,2}.weight
    # A TE-grouped model (moe_grouped_gemm=True — the build_llava_ov2/HF default that actually gets
    # built; the provider field does not propagate through the HF rebuild) instead names them:
    #   ...mlp.experts.linear_fc{1,2}.weight{i}
    # Remap per-expert -> grouped ONLY when the built model is grouped, so each EP rank's 16 local
    # experts load key-for-key. Without this they stay random (loss ~20, grad norm ~1800). Keeping
    # the model grouped preserves the faster grouped-GEMM path (no SequentialMLP fallback needed).
    import re as _re
    if any(_re.search(r"\.experts\.linear_fc[12]\.weight\d+$", k) for k in msd):
        _pe_re = _re.compile(r"^(.*\.experts)\.local_experts\.(\d+)\.(linear_fc[12])\.weight$")
        _remapped, _n = {}, 0
        for k, v in sd.items():
            m = _pe_re.match(k)
            if m:
                _remapped[f"{m.group(1)}.{m.group(3)}.weight{m.group(2)}"] = v
                _n += 1
            else:
                _remapped[k] = v
        if _n:
            logger.info("[ov2 load] remapped %d per-expert ckpt keys -> TE-grouped "
                        "(experts.linear_fcX.weightI)", _n)
        sd = _remapped

    if not load_adapter:
        sd = {k: v for k, v in sd.items() if not k.startswith("adapter.")}
    # load_vision=False: drop vision_model.* from the primary stitch so a separate p16m33 vision
    # source (load_hf_encoder_into_tower) is authoritative (was a dead param; now honored).
    if not load_vision:
        sd = {k: v for k, v in sd.items() if not k.startswith("vision_model.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [
        k for k in missing
        if not k.endswith("._extra_state")
        and not (not load_adapter and k.startswith("adapter."))
        and not (not load_vision and k.startswith("vision_model."))
    ]
    real_unexpected = [k for k in unexpected if not k.endswith("._extra_state")]
    summary = {
        "loaded": len(sd),
        "missing": real_missing,
        "unexpected": real_unexpected,
        "adapter_loaded": load_adapter,
    }
    logger.info(
        "[ov2 load] loaded=%d missing=%d unexpected=%d adapter_loaded=%s",
        len(sd), len(real_missing), len(real_unexpected), load_adapter,
    )
    # Fail-fast on a partial load. perform_init is OFF here, so any param NOT found in the checkpoint
    # stays as torch.empty (uninitialized garbage) and training would silently learn from random
    # weights -- e.g. if the MoE per-expert -> TE-grouped key remap above ever stops matching, every
    # expert would be left random with only an INFO line. Mirror the EP-shard SystemExit above.
    if real_missing:
        import os
        _preview = real_missing[:20]
        _msg = (
            "[ov2 load] {} model parameter(s) were NOT found in the checkpoint and remain "
            "UNINITIALIZED (perform_init off -> torch.empty). The checkpoint format likely drifted "
            "(e.g. the MoE expert-key remap no longer matches the built model). First {}: {}{}".format(
                len(real_missing), len(_preview), _preview,
                "" if len(real_missing) <= len(_preview) else " ...",
            )
        )
        if os.environ.get("OV2_ALLOW_PARTIAL_LOAD", "0") == "1":
            logger.error("%s -- OV2_ALLOW_PARTIAL_LOAD=1 set, continuing anyway.", _msg)
        else:
            raise SystemExit(
                _msg + " If these params are intentionally initialized elsewhere, re-run with "
                "OV2_ALLOW_PARTIAL_LOAD=1 to bypass this guard."
            )
    return summary


# Back-compat alias for the old 4B-specific name (running scripts / provider import it).
load_ov2_4b_mcore_checkpoint = load_ov2_mcore_checkpoint
