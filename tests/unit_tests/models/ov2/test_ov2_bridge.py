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


@pytest.fixture
def mock_pretrained():
    p = Mock()
    p.config = _make_composite_config()
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
