# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bridge ModelProvider for OV2 (p16m33), backbone-agnostic.

`LlavaOnevision2Provider` subclasses `GPTModelProvider` (TransformerConfig + ModelProviderMixin),
so setup()/get_model() can consume it like any Bridge provider (it carries the parallelism /
precision / DDP config fields). `.provide()` returns ONE `LlavaOnevision2` (built by the verified
`build_llava_ov2`), and the legacy 3-sibling mcore stitch-load runs as a `pre_wrap_hook` — the
same slot Bridge's HF loader uses (auto_bridge.py:1519) — so Bridge's torch_dist loader is never
needed for the OV2 ckpt.

Construct via `LlavaOnevision2Provider.from_llm(llm_hf_path, ...)`, which copies the populated
TransformerConfig fields from the LLM's AutoBridge provider (GPTModelProvider requires
num_layers/hidden_size/... at construction). Because the OV2 vision encoder + m33 adapter + step
function are LLM-agnostic and the adapter auto-sizes to `llm_cfg.hidden_size`, the same provider
serves all three backbones:

  * Qwen3-4B-Instruct-2507  (dense, model_type qwen3,        36L, hidden 2560)
  * Qwen3-8B                (dense, model_type qwen3,        36L, hidden 4096)
  * Qwen3.5-35B-A3B         (MoE,   model_type qwen3_5_moe,  40L, hidden 2048; GDN hybrid attn)

For the MoE backbone use `LlavaOnevision2MoEProvider` (or `from_llm(..., is_moe=True)`): it adds
expert-parallel / sequence-parallel knobs that `provide()` forwards to `build_llava_ov2`. The MoE
*architecture* itself (experts/topk/grouped-gemm/dispatcher/GDN spec) is set by the qwen3_5_moe
bridge on the copied provider, so no separate model class is needed — only the parallel wiring.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import List, Optional

from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider

logger = logging.getLogger(__name__)


def _init_ov2_adapter(adapter) -> None:
    """Initialize the m33 adapter (build used perform_init=False, leaving it uninitialized).

    Matches the AIAK-aligned standalone init: fc1 -> init_method (std 0.02), fc2 -> the scaled
    output_layer_init_method (~std/sqrt(2*num_layers)), layernorm weight=1, biases=0.
    """
    import torch

    cfg = getattr(adapter, "config", None)
    init_fn = getattr(cfg, "init_method", None)
    out_fn = getattr(cfg, "output_layer_init_method", None)
    with torch.no_grad():
        for n, p in adapter.named_parameters():
            if "layernorm" in n and "weight" in n:
                p.fill_(1.0)
            elif p.dim() >= 2:
                if "linear_fc2" in n and callable(out_fn):
                    out_fn(p)
                elif callable(init_fn):
                    init_fn(p)
                else:
                    p.normal_(0.0, 0.02)
            else:
                p.zero_()


@dataclass
class LlavaOnevision2Provider(GPTModelProvider):
    """Provider for the 3-sibling OV2 model (language_model + vision_model + adapter).

    Backbone-agnostic for dense Qwen3-family LLMs (4B / 8B). For the MoE backbone
    (Qwen3-30B-A3B) use the `LlavaOnevision2MoEProvider` subclass, which carries the
    expert/sequence-parallel knobs forwarded to the model builder.
    """

    # --- OV2-specific config (recipe sets these) ---
    llm_hf_path: str = "/ov2/pretrain_models/Qwen3-4B-Instruct-2507"
    ov2_mcore_ckpt_path: Optional[str] = None     # dir containing release/mp_rank_00/model_optim_rng.pt
    load_ov2_adapter: bool = False                # stage-1: train the m33 adapter FRESH
    load_ov2_vision: bool = True
    load_llm_weights: bool = False                # LLM weights come from the stitch ckpt, not HF
    image_token_id: int = 151655
    recompute_activations: bool = False           # LLM activation recompute (full/uniform/1) — cuts
                                                  # peak memory (~71GB->~35GB for 30B-A3B) at ~30% speed
                                                  # cost; set via recipe/CLI to fit busy/coexisting GPUs

    # --- Per-backbone vision-tower geometry (recipe sets these from _OV2_BACKBONES) ---
    # Defaults == the VERIFIED 4B p16m33 tower; the 4B recipe leaves them at these values so the
    # built vision_model/adapter are byte-identical to before. 8B/35B-A3B override via from_llm.
    vision_patch_size: int = 16
    vision_spatial_merge_size: int = 3
    vision_hidden_size: int = 1024
    vision_num_layers: int = 24
    vision_model_name: Optional[str] = None       # selects a named vision variant; currently unused (all backbones use the base 1024/24 tower)

    # OV2 ckpt is untied; the VLM does its own SP scatter on the fused embeddings.
    share_embeddings_and_output_weights: bool = False
    scatter_embedding_sequence_parallel: bool = False

    # Freeze policy (stage-1 = train adapter only). NB: LlavaOnevision2.freeze takes freeze_adapter.
    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_adapter: bool = False

    _ov2_hook_registered: bool = False

    # ------------------------------------------------------------------ #
    @classmethod
    def from_llm(cls, llm_hf_path: str, *, is_moe: Optional[bool] = None, **ov2_kwargs) -> "LlavaOnevision2Provider":
        """Build a provider whose TransformerConfig fields are copied from the LLM's
        AutoBridge provider (so all required GPTModelProvider fields are populated).

        Works for any Qwen3-family backbone — the copied provider carries the right
        num_layers/hidden_size/MoE/GDN config from the HF dir. ``is_moe`` selects which
        provider class is instantiated:
          * is_moe=None  -> auto-detect from the base provider (num_moe_experts set).
          * is_moe=True  -> force ``LlavaOnevision2MoEProvider`` (EP/SP wiring).
          * is_moe=False -> force the dense ``LlavaOnevision2Provider``.
        Calling on the subclass directly (``LlavaOnevision2MoEProvider.from_llm(...)``) keeps
        that subclass regardless of is_moe.
        """
        from megatron.bridge import AutoBridge

        base = AutoBridge.from_hf_pretrained(llm_hf_path).to_megatron_provider(load_weights=False)

        # Pick the concrete provider class. Only redirect when called on the base class so an
        # explicit subclass call is respected; auto-detect MoE from the copied base provider.
        target_cls = cls
        if cls is LlavaOnevision2Provider:
            detected_moe = bool(getattr(base, "num_moe_experts", None))
            use_moe = detected_moe if is_moe is None else is_moe
            if use_moe:
                target_cls = LlavaOnevision2MoEProvider

        own = {f.name for f in dataclasses.fields(target_cls)}
        init = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name in own}
        self = target_cls(**init)
        self.llm_hf_path = llm_hf_path
        self.share_embeddings_and_output_weights = False
        if hasattr(self, "scatter_embedding_sequence_parallel"):
            self.scatter_embedding_sequence_parallel = False
        for k, v in ov2_kwargs.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------ #
    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MegatronModule:
        from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2

        pre = True if pre_process is None else pre_process
        post = True if post_process is None else post_process
        model = build_llava_ov2(
            self.llm_hf_path,
            pre_process=pre,
            post_process=post,
            perform_init=False,        # weights are overwritten by the stitch ckpt
            use_cpu_init=False,        # build on GPU: setup honors use_cpu_initialization and won't
                                       # move a CPU-init model to cuda; the stitch load_state_dict
                                       # copies the CPU ckpt tensors into the GPU params fine.
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            # MoE parallel knobs (no-op for dense backbones; the MoE subclass sets EP>1 / SP).
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            expert_tensor_parallel_size=getattr(self, "expert_tensor_parallel_size", None),
            sequence_parallel=bool(getattr(self, "sequence_parallel", False)),
            moe_expert_capacity_factor=getattr(self, "moe_expert_capacity_factor", None),
            moe_pad_expert_input_to_capacity=bool(getattr(self, "moe_pad_expert_input_to_capacity", False)),
            load_llm_weights=self.load_llm_weights,
            recompute=bool(getattr(self, "recompute_activations", False)),
            # Megatron-FSDP needs gradient_accumulation_fusion OFF unless TE>=2.10 (mcore_fsdp_adapter
            # asserts is_te_min_version("2.10") only when GAF is True). Env-gated so NON-FSDP runs are
            # untouched: set OV2_DISABLE_GAF=1 to pass grad_accum_fusion=False into build_llava_ov2.
            grad_accum_fusion=(False if __import__("os").environ.get("OV2_DISABLE_GAF") == "1" else None),
            # Per-backbone vision-tower geometry (4B defaults reproduce the verified p16m33 tower).
            patch_size=getattr(self, "vision_patch_size", 16),
            spatial_merge_size=getattr(self, "vision_spatial_merge_size", 3),
            vision_hidden_size=getattr(self, "vision_hidden_size", 1024),
            vision_num_layers=getattr(self, "vision_num_layers", 24),
            vision_model_name=getattr(self, "vision_model_name", None),
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_adapter:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_adapter=self.freeze_adapter,
            )

        # --- CRITICAL multi-node grad-normalization fix ---
        # LlavaOnevision2.__init__ does super().__init__(config=language_model.config), so the
        # RUNTIME config the schedule reads (get_model_config(model) == model.config) is the LLM's
        # config object -- NOT this provider (cfg.model). setup.py's _update_model_config_funcs sets
        # calculate_per_token_loss / finalize_model_grads_func on cfg.model, which therefore never
        # reach the runtime config for this custom multimodal model. Result: at runtime
        # calc_ptl=False + finalize_func=None -> the recipe's token-weighted GLOBAL normalization is
        # bypassed; a fallback divides loss by the LOCAL per-rank token count, which is correct
        # intra-node (single-node aligns ~3.07) but mis-scales the gradient ~10x across 2 nodes
        # (grad norm ~11 vs AIAK/single ~1.3). Set them on model.config HERE (before DDP wrap, so the
        # DDP also picks up calc_ptl -> gradient_scaling_factor=1.0). pg_collection=None makes
        # finalize_model_grads all-reduce num_tokens over the FULL data-parallel group (all ranks,
        # both nodes) -> correct global normalization (matches AIAK).
        from functools import partial as _partial
        from megatron.core.distributed.finalize_model_grads import finalize_model_grads as _finalize_model_grads
        model.config.calculate_per_token_loss = bool(getattr(self, "calculate_per_token_loss", False))
        model.config.finalize_model_grads_func = _partial(_finalize_model_grads, pg_collection=None)
        # --- fp8 / MXFP8 (Phase-2): build_llava_ov2 rebuilds the LLM from HF, so the fp8 fields that
        # cfg.mixed_precision.setup() wrote onto THIS provider (cfg.model == self) never reached the
        # RUNTIME LLM config -> MXFP8 was a SILENT NO-OP (ran bf16). Copy them across (same dead-field
        # issue as calc_ptl / moe_router_dtype). Gated on self.fp8 (None in bf16 Phase-1 -> no-op).
        if getattr(self, "fp8", None):
            # Runtime-compute fp8 fields ONLY: get_fp8_context(config) reads these at FORWARD time
            # (mcore fp8_utils), so a post-build set engages MXFP8 GEMMs (the main speedup). We do NOT
            # copy fp8 PARAM-STORAGE flags (fp8_param / fp8_param_gather / reuse_grad_buf_for_mxfp8_*):
            # those need a build-time fp8_model_init the OV2 HF-rebuild path doesn't do -> params stay
            # bf16 (standard MXFP8 = fp8 compute + bf16 master weights), which is correct + safe.
            for _f in ("fp8", "fp8_recipe", "fp8_margin", "fp8_interval", "fp8_amax_history_len",
                       "fp8_amax_compute_algo", "fp8_wgrad", "fp8_output_proj",
                       "fp8_dot_product_attention", "fp8_multi_head_attention",
                       "first_last_layers_bf16", "num_layers_at_start_in_bf16", "num_layers_at_end_in_bf16"):
                if hasattr(self, _f) and hasattr(model.config, _f):
                    setattr(model.config, _f, getattr(self, _f))
            logger.info("[ov2 provider] fp8 wired onto runtime LLM config: fp8=%s recipe=%s param_gather=%s",
                        getattr(model.config, "fp8", None), getattr(model.config, "fp8_recipe", None),
                        getattr(model.config, "fp8_param_gather", getattr(model.config, "fp8_param", None)))
        # MoE router fp32 (stability): set on the RUNTIME config too — the build-time set on the LLM
        # provider doesn't always propagate (the bridge rebuilds the config), same as the calc_ptl fix.
        if getattr(model.config, "num_moe_experts", None):
            model.config.moe_router_dtype = "fp32"
            # moe_permute_fusion: the TE Triton-JIT fused token-permute kernel intermittently wedges
            # OV2-30B-A3B (one EP rank stalls -> NCCL collective timeout). NVIDIA perf docs confirm
            # disabling it for this exact Qwen3-30B-A3B MoE due to the Triton/TE fused-permutation
            # failure. Like calc_ptl / moe_router_dtype, the cfg.model field does NOT propagate
            # (build_llava_ov2 rebuilds the LLM from HF and never sets it), so force it on the RUNTIME
            # LLM config here. Default OFF; re-enable with OV2_MOE_PERMUTE_FUSION=1 for A/B testing.
            import os
            model.config.moe_permute_fusion = os.environ.get("OV2_MOE_PERMUTE_FUSION", "0") == "1"
            # --- HybridEP / flex dispatcher (Phase-2b, GB200 NVL72): same dead-field issue. The
            # recipe/launch set moe_token_dispatcher_type on cfg.model, which never reaches the
            # rebuilt LLM (stays the bridge default 'alltoall'). Wire it on the RUNTIME config via the
            # helper (sets type='flex' + backend + disables shared-expert-overlap; self-gates on GPU
            # arch & MoE). Env-gated: OV2_FLEX_BACKEND unset -> no-op (keeps verified alltoall).
            _flex = os.environ.get("OV2_FLEX_BACKEND") or None
            if _flex:
                # MXFP8 + HybridEP fp8-dispatch is unsupported (mcore asserts in fused_a2a); fail loud
                # here instead of crashing deep in the first MoE layer. They are SEPARATE Phase-2 modes.
                if getattr(model.config, "fp8", None) and _flex == "hybridep":
                    raise SystemExit(
                        "[ov2 provider] MXFP8 + HybridEP fp8-dispatch is unsupported. Pick ONE: MXFP8 "
                        "with alltoall (unset OV2_FLEX_BACKEND), OR HybridEP with bf16 (unset fp8 / use "
                        "MIXED_PRECISION=bf16). See ax_ov2_30b_a3b_gb200.sh ACCEL=1 vs ACCEL=2.")
                from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
                apply_flex_dispatcher_backend(model.config, _flex)
                if getattr(model.config, "moe_token_dispatcher_type", None) != "flex":
                    raise SystemExit(
                        "[ov2 provider] OV2_FLEX_BACKEND={!r} did NOT engage -- apply_flex_dispatcher_backend "
                        "silently no-op'd (GPU arch not in [8,9,10], or non-MoE). Refusing to run with a "
                        "misleading 'alltoall' config; unset OV2_FLEX_BACKEND to use alltoall.".format(_flex))
                logger.info("[ov2 provider] flex dispatcher wired: type=%s backend=%s shared_overlap=%s",
                            model.config.moe_token_dispatcher_type,
                            getattr(model.config, "moe_flex_dispatcher_backend", None),
                            getattr(model.config, "moe_shared_expert_overlap", None))
        return model

    # ------------------------------------------------------------------ #
    def register_ov2_ckpt_hook(self, *, skip_if_resumable_load: Optional[str] = None) -> None:
        """Register the legacy stitch-load as a pre_wrap_hook. Call once at recipe-build time.

        On a warm resume (a real torch_dist ckpt exists in `skip_if_resumable_load`) the hook
        no-ops so the resumed weights win.
        """
        if self._ov2_hook_registered or not self.ov2_mcore_ckpt_path:
            return
        self._ov2_hook_registered = True
        ckpt = self.ov2_mcore_ckpt_path
        load_adapter, load_vision = self.load_ov2_adapter, self.load_ov2_vision
        resume_dir = skip_if_resumable_load

        def _ov2_load_hook(model_list):
            if resume_dir:
                try:
                    from megatron.bridge.training.checkpointing import checkpoint_exists

                    if checkpoint_exists(resume_dir):
                        logger.info("[ov2 provider] resume ckpt present at %s; skipping base stitch", resume_dir)
                        return model_list
                except Exception:
                    pass
            assert len(model_list) == 1, "OV2 stitch-load assumes PP=1 single chunk"
            from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import load_ov2_mcore_checkpoint

            summary = load_ov2_mcore_checkpoint(
                model_list[0], ckpt, load_adapter=load_adapter, load_vision=load_vision
            )
            logger.info("[ov2 provider] stitch-load: %s", summary)
            if not load_adapter:
                _init_ov2_adapter(model_list[0].adapter)  # build left adapter uninitialized
                logger.info("[ov2 provider] adapter freshly initialized (fc1 std, fc2 scaled)")
            return model_list

        self.register_pre_wrap_hook(_ov2_load_hook)


@dataclass
class LlavaOnevision2MoEProvider(LlavaOnevision2Provider):
    """OV2 provider for an MoE LLM backbone (Qwen3.5-35B-A3B, model_type ``qwen3_5_moe``).

    Inherits ALL OV2 behavior from ``LlavaOnevision2Provider`` (``from_llm`` field-copy, the
    3-sibling ``build_llava_ov2`` build, the stitch ``pre_wrap_hook``, and the CRITICAL
    calc_ptl/finalize fix in ``provide()``). The MoE *architecture* (num_experts / topk /
    shared-expert / grouped-gemm / dispatcher / GDN hybrid ``transformer_layer_spec``) is set by
    the qwen3_5_moe bridge on the copied provider in ``from_llm`` — this subclass only adds the
    expert-parallel / sequence-parallel wiring (mirrors qwen35_vl_provider's MoE provider) that
    ``provide()`` forwards to the builder.

    NOTE: ``num_query_groups``, ``num_moe_experts``, ``moe_router_topk``, GDN head dims, rope, etc.
    are NOT redeclared here on purpose — they come from the HF config via the bridge, so this stays
    correct if Qwen3.5-35B-A3B's exact MoE/GDN config differs from the 397B reference defaults.
    """

    # Expert / sequence parallel (recipe sets these for 35B-A3B). expert_model_parallel_size is a
    # TransformerConfig field; declare expert_tensor_parallel_size default here for clarity.
    expert_tensor_parallel_size: Optional[int] = 1

    # MoE kernel defaults (match qwen35_vl MoE recipe). The bridge already sets grouped_gemm /
    # dispatcher / permute_fusion; these are harmless re-affirmations + the SP-driven router fusion.
    moe_token_dispatcher_type: str = "alltoall"
    moe_grouped_gemm: bool = True
    # NOTE: this cfg.model field is DEAD at runtime — build_llava_ov2 rebuilds the LLM from HF and
    # never propagates it. The EFFECTIVE knob is the env-gated force-set on model.config in provide()
    # (OV2_MOE_PERMUTE_FUSION; default OFF to avoid the Triton-JIT permute wedge on OV2-30B-A3B).
    # Default here mirrors that for consistency; do NOT rely on this field to change behavior.
    moe_permute_fusion: bool = False

    mlp_only_layers: List[int] = dataclasses.field(default_factory=list)

    def finalize(self) -> None:
        # Mirror qwen35_vl: SP requires TP>1; CP>1 forces global per-token loss. The OV2
        # token-weighted loss already sets calculate_per_token_loss via the recipe + the
        # provide() fix, so we only add the parallel-consistency guards here.
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        if (self.tensor_model_parallel_size or 1) > 1:
            self.sequence_parallel = True
        # NOTE on MoE expert format: the built model uses TE-grouped experts (moe_grouped_gemm=True,
        # the HF/bridge default — the perf-preferred path). The legacy EP-sharded OV2-30B-A3B ckpt
        # stores experts PER-EXPERT (SequentialMLP keys), so load_ov2_mcore_checkpoint remaps
        # per-expert -> grouped at load time. We deliberately KEEP grouped GEMM here (no SequentialMLP
        # override) since build_llava_ov2 rebuilds the LLM from HF and would ignore the field anyway.
        # expert_tensor_parallel_size > 1 is UNVALIDATED for OV2 (the verified MoE config is EP=8 /
        # ETP=1 on one 8-GPU node). Warn loudly rather than fail, so it stays usable for experiments.
        if (getattr(self, "expert_tensor_parallel_size", 1) or 1) > 1:
            logger.warning(
                "[ov2 MoE] expert_tensor_parallel_size=%s is UNVALIDATED for OV2; only EP=8/ETP=1 has "
                "been smoke-tested. Verify gradients/outputs before trusting such runs.",
                self.expert_tensor_parallel_size,
            )
        super().finalize()
