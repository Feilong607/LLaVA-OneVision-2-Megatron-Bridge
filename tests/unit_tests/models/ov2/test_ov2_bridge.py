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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the LLaVA-OneVision-2 (OV2) composite VLM bridge.

NOTE: importing ov2_bridge pulls llava_ov2 -> fla/triton, which initializes a Triton driver at import
time, so these "unit" tests currently require a CUDA device present (run with a GPU visible). The tests
themselves are config-only (no model build / no weights)."""

from unittest.mock import Mock

import pytest

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.qwen_vl_ov2.ov2_bridge import LlavaOnevision2MoEBridge
from megatron.bridge.models.qwen_vl_ov2.ov2_provider import LlavaOnevision2Provider

pytestmark = pytest.mark.unit  # config-only unit tests (GPU needed only for the fla import; see module docstring)


def _make_vision_config():
    vc = Mock(spec=[])
    vc.model_type = "llava_onevision2"
    vc.hidden_size = 1024
    vc.num_hidden_layers = 24
    vc.num_attention_heads = 16
    vc.patch_size = 16
    vc.image_size = 448
    vc.spatial_merge_size = 3
    vc.out_hidden_size = 2048
    return vc


def _make_text_config():
    tc = Mock(spec=[])
    tc.model_type = "qwen3_moe"
    tc.hidden_size = 2048
    tc.num_hidden_layers = 48
    tc.num_attention_heads = 32
    tc.num_key_value_heads = 4
    tc.head_dim = 128
    tc.intermediate_size = 6144
    tc.moe_intermediate_size = 768
    tc.num_experts = 128
    tc.num_experts_per_tok = 8
    tc.decoder_sparse_step = 1
    tc.mlp_only_layers = []
    tc.vocab_size = 151936
    tc.max_position_embeddings = 40960
    tc.rms_norm_eps = 1e-06
    tc.rope_theta = 10000000.0
    tc.attention_bias = False
    tc.torch_dtype = "bfloat16"
    tc.initializer_range = 0.02
    tc.tie_word_embeddings = False
    tc.rope_parameters = None
    tc.rope_scaling = None
    return tc


def _make_composite_config():
    cfg = Mock(spec=[])
    cfg.model_type = "llava_onevision2_moe"
    cfg.architectures = None  # auto_map only
    cfg.text_config = _make_text_config()
    cfg.vision_config = _make_vision_config()
    cfg.tie_word_embeddings = None
    cfg.image_token_id = 151655
    cfg.video_token_id = 151656
    cfg.vision_start_token_id = 151652
    cfg.vision_end_token_id = 151653
    return cfg


def _make_qwen35_text_config():
    """Qwen3.5-35B-A3B text config (qwen3_5_moe_text): GatedDeltaNet hybrid + 256 shared/routed experts + MTP."""
    tc = Mock(spec=[])
    tc.model_type = "qwen3_5_moe_text"
    tc.hidden_size = 2048
    tc.num_hidden_layers = 40
    tc.num_attention_heads = 16
    tc.num_key_value_heads = 2
    tc.head_dim = 256
    tc.moe_intermediate_size = 512
    tc.shared_expert_intermediate_size = 512
    tc.num_experts = 256
    tc.num_experts_per_tok = 8
    tc.decoder_sparse_step = 1
    tc.mlp_only_layers = []
    tc.vocab_size = 248320
    tc.max_position_embeddings = 262144
    tc.rms_norm_eps = 1e-06
    tc.attention_bias = False
    tc.torch_dtype = "bfloat16"
    tc.initializer_range = 0.02
    tc.tie_word_embeddings = False
    tc.layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 10
    tc.linear_conv_kernel_dim = 4
    tc.linear_key_head_dim = 128
    tc.linear_value_head_dim = 128
    tc.linear_num_key_heads = 16
    tc.linear_num_value_heads = 32
    tc.mtp_num_hidden_layers = 1
    tc.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }
    tc.rope_scaling = None
    tc.to_dict = lambda: {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_experts": 256,
        "auto_map": {"AutoConfig": "should_be_stripped"},
    }
    return tc


def _make_qwen35_composite_config(with_image_token: bool = True):
    cfg = Mock(spec=[])
    cfg.model_type = "llava_onevision2_moe"
    cfg.architectures = ["LlavaOnevision2ForConditionalGeneration"]
    cfg.text_config = _make_qwen35_text_config()
    cfg.vision_config = _make_vision_config()
    cfg.tie_word_embeddings = False
    if with_image_token:
        cfg.image_token_id = 248056
    cfg.video_token_id = 248057
    cfg.vision_start_token_id = 248053
    cfg.vision_end_token_id = 248054
    return cfg


@pytest.fixture
def mock_pretrained():
    p = Mock()
    p.config = _make_composite_config()
    return p


@pytest.fixture
def mock_qwen35_pretrained():
    p = Mock()
    p.config = _make_qwen35_composite_config()
    return p


class TestRegistration:
    def test_is_subclass(self):
        assert issubclass(LlavaOnevision2MoEBridge, MegatronModelBridge)

    def test_instantiation(self):
        assert LlavaOnevision2MoEBridge() is not None

    def test_registered_in_dispatch(self):
        from megatron.bridge.models.conversion.model_bridge import get_model_bridge

        reg = getattr(get_model_bridge, "_exact_types", {})
        assert any("LlavaOnevision2ForConditionalGeneration" in str(k) for k in reg), (
            "OV2 bridge not in get_model_bridge registry"
        )


class TestProviderBridge:
    def test_returns_ov2_provider(self, mock_pretrained):
        provider = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert isinstance(provider, LlavaOnevision2Provider)

    def test_llm_dimensions(self, mock_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert p.num_layers == 48
        assert p.hidden_size == 2048
        assert p.num_attention_heads == 32
        assert p.head_dim == 128

    def test_moe_config(self, mock_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert p.num_moe_experts == 128
        assert p.moe_router_topk == 8
        assert p.moe_ffn_hidden_size == 768

    def test_vision_geometry(self, mock_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert p.vision_patch_size == 16
        assert p.vision_spatial_merge_size == 3
        assert p.vision_num_layers == 24
        assert p.vision_hidden_size == 1024

    def test_token_ids_and_tie(self, mock_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert p.image_token_id == 151655
        assert p.share_embeddings_and_output_weights is False

    def test_no_mrope_for_30b(self, mock_pretrained):
        # 30B qwen3_moe has no mrope_section -> stays None (1D rope), byte-identical to training.
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        assert getattr(p, "mrope_section", None) is None


class TestMappingRegistry:
    def test_returns_registry(self):
        reg = LlavaOnevision2MoEBridge().mapping_registry()
        assert isinstance(reg, MegatronMappingRegistry)


class TestQwen35Backbone:
    """The Qwen3.5-35B-A3B line (qwen3_5_moe_text). Without these branches a Qwen3.5 export silently dropped every
    GDN / shared-expert / MTP tensor (the export loop only warns on an unmapped param) and HF->mcore built a plain
    Qwen3-MoE because provider_bridge rewrote the temp text config to qwen3_moe."""

    def test_provider_is_qwen35_hybrid(self, mock_qwen35_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_qwen35_pretrained)
        assert p.num_layers == 40
        assert p.num_moe_experts == 256
        assert p.moe_router_topk == 8
        assert p.moe_ffn_hidden_size == 512
        assert p.moe_shared_expert_intermediate_size == 512
        assert p.experimental_attention_variant == "gated_delta_net"
        assert p.linear_attention_freq == 4
        assert p.rotary_percent == 0.25
        assert p.attention_output_gate is True
        assert p.mtp_num_layers == 1
        assert p.head_dim == 256

    def test_provider_mrope_and_token_id(self, mock_qwen35_pretrained):
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_qwen35_pretrained)
        assert list(p.mrope_section) == [11, 11, 10]
        assert p.image_token_id == 248056

    def test_image_token_default_when_skeleton_omits_it(self):
        pre = Mock()
        pre.config = _make_qwen35_composite_config(with_image_token=False)
        p = LlavaOnevision2MoEBridge().provider_bridge(pre)
        assert p.image_token_id == 248056  # NOT 151655 (a normal text token in the Qwen3.5 vocab)

    def test_temp_text_dir_keeps_qwen35_model_type(self, mock_qwen35_pretrained):
        import json
        import os

        p = LlavaOnevision2MoEBridge().provider_bridge(mock_qwen35_pretrained)
        cfg_path = os.path.join(p.llm_hf_path, "config.json")
        assert os.path.isfile(cfg_path), "provider_bridge must write the inner-LLM build config"
        raw = json.load(open(cfg_path))
        assert raw["model_type"] == "qwen3_5_moe_text"
        assert raw["architectures"] == ["Qwen3_5MoeForCausalLM"]
        assert "auto_map" not in raw

    def test_temp_text_dir_still_qwen3_moe_for_30b(self, mock_pretrained):
        import json
        import os

        mock_pretrained.config.text_config.to_dict = lambda: {"model_type": "qwen3_moe", "num_experts": 128}
        p = LlavaOnevision2MoEBridge().provider_bridge(mock_pretrained)
        raw = json.load(open(os.path.join(p.llm_hf_path, "config.json")))
        assert raw["model_type"] == "qwen3_moe"
        assert raw["architectures"] == ["Qwen3MoeForCausalLM"]

    def test_registry_maps_gdn_shared_packed_and_mtp(self):
        bridge = LlavaOnevision2MoEBridge()
        bridge.hf_config = _make_qwen35_composite_config()
        reg = bridge.mapping_registry()
        # GDN layer tensors
        for name in (
            "language_model.decoder.layers.0.self_attention.in_proj.weight",
            "language_model.decoder.layers.0.self_attention.in_proj.layer_norm_weight",
            "language_model.decoder.layers.0.self_attention.conv1d.weight",
            "language_model.decoder.layers.0.self_attention.A_log",
            "language_model.decoder.layers.0.self_attention.dt_bias",
            "language_model.decoder.layers.0.self_attention.out_norm.weight",
            "language_model.decoder.layers.0.self_attention.out_proj.weight",
            # gated-attention layer
            "language_model.decoder.layers.3.self_attention.linear_qkv.weight",
            "language_model.decoder.layers.3.self_attention.q_layernorm.weight",
            # shared experts + router + packed routed experts
            "language_model.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            "language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.weight",
            "language_model.decoder.layers.0.mlp.shared_experts.gate_weight",
            "language_model.decoder.layers.0.mlp.router.weight",
            "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
            # MTP head
            "language_model.mtp.layers.0.eh_proj.weight",
            "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight",
            # embeddings / head / vision / adapter
            "language_model.embedding.word_embeddings.weight",
            "language_model.output_layer.weight",
            "vision_model.decoder.layers.0.self_attention.linear_qkv.weight",
            "adapter.linear_fc2.weight",
        ):
            assert reg.megatron_to_hf_lookup(name) is not None, f"unmapped Qwen3.5 param: {name}"
        # HF side must be the Qwen3.5 layout: packed routed experts, VL prefix, top-level mtp.*
        hf = reg.megatron_to_hf_lookup("language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0").hf_param
        assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in str(hf)
        hf = reg.megatron_to_hf_lookup("language_model.decoder.layers.0.self_attention.in_proj.weight")
        assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" in str(
            hf.hf_param if hasattr(hf, "hf_param") else hf.__dict__
        )
        mtp = reg.megatron_to_hf_lookup("language_model.mtp.layers.0.eh_proj.weight")
        assert str(mtp.hf_param) == "mtp.fc.weight"

    def test_registry_no_per_expert_keys_for_qwen35(self):
        bridge = LlavaOnevision2MoEBridge()
        bridge.hf_config = _make_qwen35_composite_config()
        reg = bridge.mapping_registry()
        assert reg.hf_to_megatron_lookup("model.language_model.layers.0.mlp.experts.0.gate_proj.weight") is None

    def test_registry_30b_unchanged(self):
        # The qwen3_moe registry keeps its per-expert layout and has no GDN entries.
        bridge = LlavaOnevision2MoEBridge()
        bridge.hf_config = _make_composite_config()
        reg = bridge.mapping_registry()
        assert reg.hf_to_megatron_lookup("model.language_model.layers.0.mlp.experts.0.gate_proj.weight") is not None
        assert reg.megatron_to_hf_lookup("language_model.decoder.layers.0.self_attention.A_log") is None
        assert reg.megatron_to_hf_lookup("adapter.linear_fc2.weight") is not None

    def test_export_mtp_opt_out(self, monkeypatch):
        monkeypatch.setenv("OV2_EXPORT_MTP", "0")
        bridge = LlavaOnevision2MoEBridge()
        bridge.hf_config = _make_qwen35_composite_config()
        reg = bridge.mapping_registry()
        assert reg.megatron_to_hf_lookup("language_model.mtp.layers.0.eh_proj.weight") is None
        assert reg.megatron_to_hf_lookup("language_model.decoder.layers.0.self_attention.A_log") is not None
