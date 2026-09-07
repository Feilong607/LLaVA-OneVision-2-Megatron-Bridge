# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Bridge for LLaVA-OneVision-2 (OV2) composite VLMs.

OV2 grafts a Megatron OneVision encoder + an m33 patch-merger adapter onto a Qwen3-family LLM.
Unlike the native qwen_vl bridges (which map Qwen's NATIVE vision tower), OV2's vision tower is the
custom OneVision encoder, so this bridge supplies its OWN vision + merger mappings while REUSING the
qwen3_moe LLM mapping verbatim (the OV2 inner LLM IS qwen3_moe with the `model.language_model.*` HF
prefix, identical to Qwen3-VL-MoE).

Composite HF model = `LlavaOnevision2ForConditionalGeneration` (trust_remote_code / auto_map-only,
model_type="llava_onevision2_moe"):
  model.language_model.*   qwen3_moe text LLM   -> mcore language_model.*
  model.visual.*           OneVision encoder    -> mcore vision_model.*
  model.visual.merger.*    patch merger         -> mcore adapter.*
  lm_head.weight                                -> mcore language_model.output_layer.weight

Backbones. The inner LLM is selected by ``text_config.model_type``:
  * ``qwen3_moe`` / ``qwen3`` (30B-A3B / 4B): the qwen3 mappings below (VALIDATED export + roundtrip).
  * ``qwen3_5_moe_text`` / ``qwen3_5_text`` (Qwen3.5-35B-A3B line): the LLM half is delegated to
    ``Qwen35MoEBridge._get_moe_lm_mappings`` (+ ``_get_moe_mtp_mappings``) exactly as the native
    Qwen3.5-VL bridge does, so GatedDeltaNet (in_proj/conv1d/A_log/dt_bias/out_norm/out_proj), the
    shared experts, the PACKED routed experts (experts.gate_up_proj / down_proj) and the MTP head are
    all mapped. Without this branch a Qwen3.5 export silently dropped 30/40 layers' mixers (the export
    loop only warns on an unmapped megatron param). provider_bridge likewise keeps the Qwen3.5
    model_type/architectures in the temp text dir so HF->mcore (roundtrip) builds the real hybrid.

⚠️ DRAFT — needs HF<->mcore round-trip verification (skill acceptance bar) before production use.
   Two architectural subtleties to confirm against the AIAK base ckpt during verification:
   (1) HF `model.visual.layernorm_post` (applied after the encoder, before the merger): the mcore
       vision tower is built post_process=False (NO decoder.final_layernorm). Confirm whether
       layernorm_post folds into adapter.layernorm or is genuinely dropped. Currently UNMAPPED here.
   (2) merger.ln_q vs layernorm_post: HF has two norms after the encoder; mcore adapter has one
       (adapter.layernorm). The ln_q->adapter.layernorm mapping below assumes ln_q is the surviving one.
"""

import os

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen.qwen35_bridge import (
    Qwen35Bridge,
    Qwen35MoEBridge,
    _apply_qwen35_common_config,
    _apply_qwen35_moe_config,
)
from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import LlavaOnevision2
from megatron.bridge.models.qwen_vl_ov2.ov2_provider import LlavaOnevision2Provider


# Qwen3.5 <|image_pad|>; 151655 is a NORMAL text token in the Qwen3.5 vocabulary (recipes/ov2/ov2_qwen35.py).
_QWEN35_IMAGE_TOKEN_ID = 248056
_QWEN3_IMAGE_TOKEN_ID = 151655


def _text_model_type(text_config) -> str:
    """model_type of the composite's inner text config ('' when unavailable, e.g. a bare Mock)."""
    mt = getattr(text_config, "model_type", None) if text_config is not None else None
    return mt if isinstance(mt, str) else ""


def _is_qwen35_text(text_config) -> bool:
    """True for the Qwen3.5 line (qwen3_5_moe_text / qwen3_5_text). OV2_BRIDGE_TEXT_MODEL_TYPE overrides
    when the dispatch system hands the bridge no usable config (export paths); unset -> pure detection."""
    mt = _text_model_type(text_config) or os.environ.get("OV2_BRIDGE_TEXT_MODEL_TYPE", "")
    return mt.startswith("qwen3_5")


# OV2 VISION encoder (OneVision) + m33 MERGER->ADAPTER 1:1 name pairs. Backbone-agnostic: the tower and
# adapter classes are the same for every OV2 line (geometry comes from vision_config at build time).
_OV2_VISION_ADAPTER_PARAM_MAPPINGS = {
    # === OV2 VISION encoder (OneVision) — model.visual.encoder.layers.* -> vision_model.decoder.layers.* ===
    # attention out-projection
    "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.encoder.layers.*.self_attn.proj.weight",
    "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.encoder.layers.*.self_attn.proj.bias",
    # MLP (HF fc1/fc2 -> TE linear_fc1/fc2)
    "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.visual.encoder.layers.*.mlp.fc1.weight",
    "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.visual.encoder.layers.*.mlp.fc1.bias",
    "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.encoder.layers.*.mlp.fc2.weight",
    "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.encoder.layers.*.mlp.fc2.bias",
    # layer norms (TE-fused into the following linear): layer_norm1->qkv LN, layer_norm2->fc1 LN
    "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.encoder.layers.*.layer_norm1.weight",
    "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.encoder.layers.*.layer_norm1.bias",
    "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.encoder.layers.*.layer_norm2.weight",
    "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.encoder.layers.*.layer_norm2.bias",
    # pre-encoder layernorm (separate TENorm module, NOT fused)
    "vision_model.pre_layernorm.weight": "model.visual.layernorm_pre.weight",
    "vision_model.pre_layernorm.bias": "model.visual.layernorm_pre.bias",
    # === MERGER -> ADAPTER (HF model.visual.merger.{ln_q,mlp.0,mlp.2} -> mcore adapter.{layernorm,linear_fc1,linear_fc2}) ===
    "adapter.layernorm.weight": "model.visual.merger.ln_q.weight",
    "adapter.layernorm.bias": "model.visual.merger.ln_q.bias",
    "adapter.linear_fc1.weight": "model.visual.merger.mlp.0.weight",
    "adapter.linear_fc1.bias": "model.visual.merger.mlp.0.bias",
    "adapter.linear_fc2.weight": "model.visual.merger.mlp.2.weight",
    "adapter.linear_fc2.bias": "model.visual.merger.mlp.2.bias",
    # NOTE: model.visual.layernorm_post is intentionally NOT mapped here (mcore tower is
    # post_process=False). Confirm fold/drop during round-trip verification (see module docstring).
}


def _ov2_vision_adapter_special_mappings() -> list:
    """Vision-tower mappings that need a transform (fused QKV re-layout, 4-D conv weight)."""
    return [
        # OV2 VISION QKV: HF stores a FUSED self_attn.qkv [3d,d] (modeling_*:585 self.qkv=nn.Linear(d,3d)),
        # like Qwen3-VL's vision tower -> ConcatenatedQKVMapping re-lays-out [q;k;v] into TE linear_qkv.
        ConcatenatedQKVMapping(
            megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
            hf_param="model.visual.encoder.layers.*.self_attn.qkv.weight",
        ),
        ConcatenatedQKVMapping(
            megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
            hf_param="model.visual.encoder.layers.*.self_attn.qkv.bias",
        ),
        # patch_embed: HF Conv2d -> mcore Conv2d (replicated, 4-D weight kept)
        ReplicatedMapping(
            megatron_param="vision_model.patch_embed.proj.**",
            hf_param="model.visual.embeddings.patch_embedding.**",
        ),
    ]


@MegatronModelBridge.register_bridge(
    source="LlavaOnevision2ForConditionalGeneration",  # string: auto_map/trust_remote_code class, no installed transformers class
    target=LlavaOnevision2,
    provider=LlavaOnevision2Provider,
    model_type="llava_onevision2_moe",
)
class LlavaOnevision2MoEBridge(MegatronModelBridge):
    """HF LlavaOnevision2ForConditionalGeneration (MoE qwen3_moe LLM + OneVision encoder) <-> mcore LlavaOnevision2.

    The LLM mappings are IDENTICAL to Qwen3-VL-MoE (same `model.language_model.*` qwen3_moe layout);
    the vision + merger mappings are OV2-specific (OneVision encoder, not Qwen's native ViT).
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> LlavaOnevision2Provider:
        hf_config = hf_pretrained.config
        # AutoBridge may hand us the INNER text (qwen3_moe) config for an auto_map/string-source
        # composite; recover the full composite (text_config + vision_config) from the source path.
        if not hasattr(hf_config, "vision_config"):
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(
                hf_pretrained.model_name_or_path, trust_remote_code=True
            )
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config
        is_qwen35 = _is_qwen35_text(text_config)

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = LlavaOnevision2Provider(**provider_kwargs)

        # --- qwen3-family LLM settings (mirror Qwen3VL{,MoE}Bridge.provider_bridge) ---
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)
        # tie_word_embeddings lives on the TOP-LEVEL config for VLMs (OV2 unties output_layer).
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False) or False
        # MoE (30B llava_onevision2_moe = qwen3_moe) vs dense (4B llava_onevision2 = qwen3):
        # the two share one HF class name, so branch on the text_config to pick experts vs dense MLP.
        is_moe = getattr(text_config, "num_experts", None) is not None
        if is_qwen35:
            # Qwen3.5 (GatedDeltaNet hybrid + shared-expert MoE + MTP): mirror Qwen35MoEBridge.provider_bridge
            # so the composite provider describes the real LLM (rotary_percent 0.25, GDN dims, shared experts,
            # global_aux_loss, MTP). The LLM is still BUILT from the temp text dir below, which keeps the
            # Qwen3.5 model_type so build_llava_ov2 dispatches to Qwen35MoEBridge -- previously it was
            # rewritten to qwen3_moe and a plain Qwen3-MoE(+mrope) was built with no error (roundtrip garbage).
            provider.mtp_num_layers = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
            if is_moe:
                _apply_qwen35_moe_config(provider, text_config)
            else:
                _apply_qwen35_common_config(provider, text_config)
        elif is_moe:
            provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
            provider.num_moe_experts = text_config.num_experts
            provider.moe_router_topk = text_config.num_experts_per_tok
            provider.decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)
            provider.mlp_only_layers = getattr(text_config, "mlp_only_layers", [])
            provider.moe_grouped_gemm = True
            provider.moe_router_load_balancing_type = "aux_loss"
            provider.moe_aux_loss_coeff = 1e-3
            provider.moe_router_pre_softmax = False
            provider.moe_token_dispatcher_type = "alltoall"
        provider.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

        # --- OV2 vision-tower geometry (drives build_llava_ov2 in provide()) ---
        provider.vision_patch_size = getattr(vision_config, "patch_size", 16)
        provider.vision_spatial_merge_size = getattr(vision_config, "spatial_merge_size", 3)
        provider.vision_hidden_size = getattr(vision_config, "hidden_size", 1024)
        provider.vision_num_layers = getattr(vision_config, "num_hidden_layers", 24)

        # --- OV2 token ids (composite config, top level). A skeleton that omits image_token_id gets the
        # backbone's <|image_pad|>: 248056 for Qwen3.5 (151655 is a normal text token there), 151655 otherwise.
        _img_tok = getattr(hf_config, "image_token_id", None)
        provider.image_token_id = (
            int(_img_tok) if _img_tok else (_QWEN35_IMAGE_TOKEN_ID if is_qwen35 else _QWEN3_IMAGE_TOKEN_ID)
        )
        # mrope only for the qwen3.5 backbone (section in text_config.rope_parameters/rope_scaling);
        # qwen3_moe 30B has none -> mrope_section stays None -> 1D rope, exactly as it trains today.
        rope_cfg = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
        ms = rope_cfg.get("mrope_section") if isinstance(rope_cfg, dict) else None
        if ms is not None:
            provider.mrope_section = ms

        # --- Resolve the inner-LLM build source from the composite's OWN text_config (issue: the
        # default llm_hf_path points at a 4B dir). build_llava_ov2 (called by provide()) does
        # AutoBridge.from_hf_pretrained(provider.llm_hf_path) to construct the LLM; write text_config
        # to a temp HF dir so that build dispatches to the qwen3_moe bridge and builds the CORRECT
        # (e.g. 48L/128-expert) LLM. No weights needed (load_weights=False structure build). ---
        try:
            import json as _json, tempfile
            _td = tempfile.mkdtemp(prefix="ov2_text_llm_")
            _raw = text_config.to_dict()
            _raw.pop("auto_map", None)  # CRITICAL: strip auto_map so build_llava_ov2's
            _raw.pop("_name_or_path", None)  # AutoBridge.from_hf_pretrained(this dir) routes to the
            # registered qwen3{,_moe} bridge, NOT back into THIS OV2 bridge (would re-enter provider_bridge).
            if is_qwen35:
                # Keep the Qwen3.5 text model_type (qwen3_5_moe_text / qwen3_5_text) so AutoBridge routes to
                # Qwen35MoEBridge / Qwen35Bridge and builds the GDN+MoE(+MTP) hybrid the checkpoint was trained as.
                _raw["model_type"] = _text_model_type(text_config)
                _raw["architectures"] = ["Qwen3_5MoeForCausalLM"] if is_moe else ["Qwen3_5ForCausalLM"]
            elif is_moe:
                _raw["model_type"] = "qwen3_moe"
                _raw["architectures"] = ["Qwen3MoeForCausalLM"]
            else:
                _raw["model_type"] = "qwen3"
                _raw["architectures"] = ["Qwen3ForCausalLM"]
            with open(_td + "/config.json", "w") as _f:
                _json.dump(_raw, _f)
            provider.llm_hf_path = _td
        except Exception:
            pass  # unit-test Mock text_config has no to_dict -> leave llm_hf_path default

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # MoE (30B) vs dense (4B): the LLM MLP mappings differ. self.hf_config is set by the dispatch
        # system before this is called; it may be the composite (has .text_config) or the inner LLM config.
        _hf = getattr(self, "hf_config", None)
        _tc = getattr(_hf, "text_config", _hf) if _hf is not None else None
        is_moe = (_tc is not None) and (getattr(_tc, "num_experts", None) is not None)
        is_qwen35 = _is_qwen35_text(_tc)

        if is_qwen35:
            return self._qwen35_mapping_registry(_tc, is_moe)

        # ---- 1:1 (AutoMapping) name pairs : { megatron_param : hf_param } ----
        param_mappings = {
            # === LLM (qwen3_moe) — VERBATIM from Qwen3VLMoEBridge (same model.language_model.* layout) ===
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            # NOTE: post_attention_layernorm maps differently dense-vs-MoE (fused linear_fc1 LN vs separate
            # pre_mlp_layernorm) — appended in the is_moe branch below, NOT here.
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # (LLM MLP — MoE experts+router vs dense gated MLP — appended conditionally below.)
            # (OV2 vision tower + adapter pairs are appended from _OV2_VISION_ADAPTER_PARAM_MAPPINGS below.)
        }
        param_mappings.update(_OV2_VISION_ADAPTER_PARAM_MAPPINGS)

        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]

        mapping_list.append(
            # LLM QKV: separate q/k/v -> fused linear_qkv (qwen3_moe)
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            )
        )
        # (LLM MLP mappings — MoE experts+router or dense gated MLP — appended after this list.)
        mapping_list.extend(_ov2_vision_adapter_special_mappings())

        # ---- LLM MLP: MoE (30B) experts+router, or dense (4B) gated MLP ----
        if is_moe:
            mapping_list += [
                # qwen3_moe: post-attention LN is a SEPARATE pre_mlp_layernorm (not fused).
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.pre_mlp_layernorm.weight",
                    hf_param="model.language_model.layers.*.post_attention_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.router.weight",
                    hf_param="model.language_model.layers.*.mlp.gate.weight",
                ),
                # qwen3_moe PER-EXPERT gate/up/down -> TEGroupedMLP (experts.linear_fc*) + SequentialMLP
                # (experts.local_experts.*.linear_fc*); mirror qwen/qwen3_moe_bridge.py.
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        else:
            # dense qwen3 4B: single gated MLP per layer (no experts, no router); the post-attention
            # LN is FUSED into the TE LayerNormMLP -> mlp.linear_fc1.layer_norm_weight.
            mapping_list += [
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                    hf_param="model.language_model.layers.*.post_attention_layernorm.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc2.weight",
                    hf_param="model.language_model.layers.*.mlp.down_proj.weight",
                ),
            ]

        return MegatronMappingRegistry(*mapping_list)

    @staticmethod
    def _qwen35_mapping_registry(text_config, is_moe: bool) -> MegatronMappingRegistry:
        """Qwen3.5 backbone: delegate the LLM half to the native Qwen3.5 bridges (same prefixes as Qwen3.5-VL).

        Covers GatedDeltaNet layers (in_proj / conv1d / A_log / dt_bias / out_norm / out_proj), gated attention
        layers, PACKED routed experts (experts.gate_up_proj / down_proj), shared experts (+gate), embeddings,
        final norm, lm_head, and -- when text_config.mtp_num_hidden_layers > 0 -- the MTP head as top-level
        ``mtp.*`` HF keys (the Qwen3.5-native layout; HF's Qwen3.5 classes ignore ``^mtp.*`` on load, and the
        OV2 golden modeling does the same). OV2_EXPORT_MTP=0 drops the MTP mappings (weights stay out of the
        HF dir). MTP experts are written per-expert (Qwen3.5 layout, not the Qwen3.6 packed one).
        """
        mapping_list = []
        if is_moe:
            mapping_list.extend(
                Qwen35MoEBridge._get_moe_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="language_model.")
            )
            _mtp_layers = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
            if _mtp_layers > 0 and os.environ.get("OV2_EXPORT_MTP", "1") == "1":
                mapping_list.extend(
                    Qwen35MoEBridge._get_moe_mtp_mappings(megatron_prefix="language_model.", mtp_experts_packed=False)
                )
        else:
            mapping_list.extend(
                Qwen35Bridge._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="language_model.")
            )
        mapping_list.extend(AutoMapping(megatron_param=m, hf_param=h) for m, h in _OV2_VISION_ADAPTER_PARAM_MAPPINGS.items())
        mapping_list.extend(_ov2_vision_adapter_special_mappings())
        return MegatronMappingRegistry(*mapping_list)
