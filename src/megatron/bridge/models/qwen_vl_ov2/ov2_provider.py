# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bridge ModelProvider for OV2.1-4B (p16m33).

`LlavaOnevision2Provider` subclasses `GPTModelProvider` (TransformerConfig + ModelProviderMixin),
so setup()/get_model() can consume it like any Bridge provider (it carries the parallelism /
precision / DDP config fields). `.provide()` returns ONE `LlavaOnevision2` (built by the verified
`build_llava_ov2_4b`), and the legacy 3-sibling mcore stitch-load runs as a `pre_wrap_hook` — the
same slot Bridge's HF loader uses (auto_bridge.py:1519) — so Bridge's torch_dist loader is never
needed for the OV2 ckpt.

Construct via `LlavaOnevision2Provider.from_llm(llm_hf_path, ...)`, which copies the populated
TransformerConfig fields from the Qwen3-4B AutoBridge provider (GPTModelProvider requires
num_layers/hidden_size/... at construction).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

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
    """Provider for the 3-sibling OV2.1 model (language_model + vision_model + adapter)."""

    # --- OV2-specific config (recipe sets these) ---
    llm_hf_path: str = "/ov2/pretrain_models/Qwen3-4B-Instruct-2507"
    ov2_mcore_ckpt_path: Optional[str] = None     # dir containing release/mp_rank_00/model_optim_rng.pt
    load_ov2_adapter: bool = False                # stage-1: train the m33 adapter FRESH
    load_ov2_vision: bool = True
    load_llm_weights: bool = False                # LLM weights come from the stitch ckpt, not HF
    image_token_id: int = 151655

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
    def from_llm(cls, llm_hf_path: str, **ov2_kwargs) -> "LlavaOnevision2Provider":
        """Build a provider whose TransformerConfig fields are copied from the Qwen3-4B
        AutoBridge provider (so all required GPTModelProvider fields are populated)."""
        from megatron.bridge import AutoBridge

        base = AutoBridge.from_hf_pretrained(llm_hf_path).to_megatron_provider(load_weights=False)
        own = {f.name for f in dataclasses.fields(cls)}
        init = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name in own}
        self = cls(**init)
        self.llm_hf_path = llm_hf_path
        self.share_embeddings_and_output_weights = False
        if hasattr(self, "scatter_embedding_sequence_parallel"):
            self.scatter_embedding_sequence_parallel = False
        for k, v in ov2_kwargs.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------ #
    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MegatronModule:
        from megatron.bridge.models.qwen_vl_ov2.llava_ov2_4b import build_llava_ov2_4b

        pre = True if pre_process is None else pre_process
        post = True if post_process is None else post_process
        model = build_llava_ov2_4b(
            self.llm_hf_path,
            pre_process=pre,
            post_process=post,
            perform_init=False,        # weights are overwritten by the stitch ckpt
            use_cpu_init=False,        # build on GPU: setup honors use_cpu_initialization and won't
                                       # move a CPU-init model to cuda; the stitch load_state_dict
                                       # copies the CPU ckpt tensors into the GPU params fine.
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            load_llm_weights=self.load_llm_weights,
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
            from megatron.bridge.models.qwen_vl_ov2.llava_ov2_4b import load_ov2_4b_mcore_checkpoint

            summary = load_ov2_4b_mcore_checkpoint(
                model_list[0], ckpt, load_adapter=load_adapter, load_vision=load_vision
            )
            logger.info("[ov2 provider] stitch-load: %s", summary)
            if not load_adapter:
                _init_ov2_adapter(model_list[0].adapter)  # build left adapter uninitialized
                logger.info("[ov2 provider] adapter freshly initialized (fc1 std, fc2 scaled)")
            return model_list

        self.register_pre_wrap_hook(_ov2_load_hook)
