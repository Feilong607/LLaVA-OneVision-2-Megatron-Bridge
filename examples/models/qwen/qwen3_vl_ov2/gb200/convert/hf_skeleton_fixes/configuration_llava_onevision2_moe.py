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
"""Composite config for the OV2 (LLaVA-OneVision-2) HF export -- backbone-dispatching FALLBACK.

The auto_model skeletons on the cluster ship their own ``configuration_llava_onevision2_moe.py``. This
vendored copy exists for one reason: a Qwen3.5-backbone export needs ``text_config`` to deserialize as a
REAL ``Qwen3_5MoeTextConfig`` (model_type ``qwen3_5_moe_text``, ``layer_types`` for the GDN/attention
hybrid, ``rope_parameters.mrope_section``). A skeleton configuration class that hardcodes ``Qwen3MoeConfig``
for ``text_config`` silently coerces the dict to model_type ``qwen3_moe`` -> ``AutoModel.from_config``
builds a plain Qwen3-MoE -> every GDN / shared-expert tensor loads as "missing" (random) with a warning only.

``build_qwen35_hf_skeleton.py`` and ``convert.sh fixup`` install this file ONLY when the skeleton's own
configuration class fails that dispatch test (or is absent). It follows transformers' ``LlavaConfig``
pattern: ``text_config`` is dispatched by its ``model_type`` through ``CONFIG_MAPPING`` (works for
``qwen3_moe`` / ``qwen3`` / ``qwen3_5_moe_text`` / ``qwen3_5_text`` alike), ``vision_config`` is the OV2
OneVision-encoder config. Attribute names of the vision config are exactly the ones
``modeling_llava_onevision2_moe.py`` reads (``config.hidden_size``, ``num_hidden_layers``,
``num_attention_heads``, ``patch_size``, ``image_size``, ``spatial_merge_size``, ``out_hidden_size``,
``layer_norm_eps``, ``layer_norm_type``, ``num_channels``, ``attention_dropout``, ``rope_theta``,
``use_head``, ``intermediate_size`` / ``hidden_act`` via SiglipMLP, ``use_patch_position_encoding``).
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class LlavaOnevision2VisionConfig(PretrainedConfig):
    """OneVision (OV2.1) encoder + m33 patch-merger geometry. Defaults = the p16m33 tower every OV2 line uses."""

    model_type = "llava_onevision2"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=16,
        spatial_merge_size=3,
        out_hidden_size=2048,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        rope_theta=10000.0,
        use_head=False,
        use_patch_position_encoding=False,
        temporal_patch_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.use_head = use_head
        self.use_patch_position_encoding = use_patch_position_encoding
        self.temporal_patch_size = temporal_patch_size


class LlavaOnevision2MoeConfig(PretrainedConfig):
    """LlavaOnevision2ForConditionalGeneration config: OneVision tower + adapter + a Qwen3-family text model.

    ``text_config`` is dispatched by ``model_type`` (qwen3_moe / qwen3 / qwen3_5_moe_text / qwen3_5_text);
    the multimodal token ids live at the TOP level (the modeling reads ``config.image_token_id`` etc.), as
    does ``tie_word_embeddings`` (the composite ties lm_head to embed_tokens only when this is True).
    """

    model_type = "llava_onevision2_moe"
    sub_configs = {"vision_config": LlavaOnevision2VisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = LlavaOnevision2VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = LlavaOnevision2VisionConfig()
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_model_type = text_config.get("model_type", "qwen3_moe")
            text_kwargs = {k: v for k, v in text_config.items() if k != "model_type"}
            text_config = CONFIG_MAPPING[text_model_type](**text_kwargs)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen3_moe"]()
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["LlavaOnevision2MoeConfig", "LlavaOnevision2VisionConfig"]
