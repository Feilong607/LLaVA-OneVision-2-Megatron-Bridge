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

"""Vision tower configuration for the OV2.1 onevision encoder.

Extracted from
``aiak_training_llm.models.llava_onevision2.llava_onevision2_config`` keeping
ONLY the vision-side ``VisionConfig`` dataclass and the
``get_vision_config`` helper.  The language-model config
(``LlavaOnevision2Config``), the adapter config (``AdapterConfig``), the
``get_adapeter_config`` helper, and every ``@register_model_config`` LLM-side
registration have been intentionally dropped — they are LLM/training-side
concerns that the qwen_vl_ov2 vision-only port does not require.
"""

from dataclasses import dataclass

import torch


@dataclass
class VisionConfig:
    """configuration for vision model

    The fields need to be consistent with the definitions in args
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: tuple[int]
    image_size: tuple[int]
    kv_channels: int
    normalization: str
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    layernorm_epsilon: float = 1e-05
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    frame_windows_size: int = 4
    # spatial_merge_size is NOT consumed by the vision TransformerConfig itself (the tower's
    # patch merging is applied by the OV2 Adapter, which receives merge explicitly). It is carried
    # here only so a per-backbone get_vision_config() can return the merge alongside the rest of the
    # tower geometry, keeping vision-tower selection in one place. Default 3 == the 4B p16m33 value.
    spatial_merge_size: int = 3


def get_vision_config(
    model_family=None,
    model_name: str = "llava-onevision2-30b-a3b",
    *,
    patch_size: int = None,
    spatial_merge_size: int = None,
    hidden_size: int = None,
    num_layers: int = None,
):
    """Build the OV2.1 vision tower config.

    Mirrors ``aiak_training_llm.models.llava_onevision2.llava_onevision2_config.
    get_vision_config``.  ``model_family`` is accepted for signature
    compatibility but unused.  ``model_name`` selects between the default
    1024-d / 24-layer vision tower used by the 30B-A3B LLM and the larger
    1664-d / 48-layer ``vision-2b`` variant.

    The keyword overrides (``patch_size`` / ``spatial_merge_size`` / ``hidden_size`` /
    ``num_layers``) let a caller pin the exact per-backbone vision geometry without relying on a
    ``model_name`` string match.  Each override, when not ``None``, wins over both the base config
    and any ``model_name`` branch (applied last). When ALL overrides are ``None`` (and model_name is
    the default) the returned config is byte-identical to the original 30B-A3B base config — this is
    the path the 4B builder uses (it then re-pins patch16/hidden1024/24L itself), so 4B is unchanged.
    """
    config = VisionConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        patch_size=14,
        image_size=(1344, 1344),
        kv_channels=64,
        normalization="LayerNorm",
        swiglu=False,
        class_token_len=0,
        group_query_attention=False,
        attention_dropout=0,
        hidden_dropout=0,
        layernorm_epsilon=1e-5,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=False,
        gated_linear_unit=False,
        in_channels=3,
        num_query_groups=16,
        add_bias_linear=True,
        add_qkv_bias=True,
        position_embedding_type="rope",
    )
    # Named-variant geometry (kept identical to the original AIAK switch). NOTE: as of the current
    # _OV2_BACKBONES, NO backbone sets vision_model_name — all use the base 1024/24 tower plus explicit
    # patch/hidden/layer overrides (incl. 30B-A3B, whose vision IS 1024/24, NOT this larger variant).
    # The ``vision-2b`` tower (1664-d / 48L / kv 104, 16 heads) is retained for any future backbone
    # that needs it but is currently UNUSED.
    if model_name and "vision-2b" in model_name:
        config.num_layers = 48
        config.hidden_size = 1664
        config.ffn_hidden_size = 8192
        config.kv_channels = 104
    elif model_name == "llava-onevision2-layer1":
        config.num_layers = 1
    # Explicit per-backbone overrides win last (None => keep base/variant value).
    if patch_size is not None:
        config.patch_size = patch_size
    if hidden_size is not None:
        config.hidden_size = hidden_size
    if num_layers is not None:
        config.num_layers = num_layers
    if spatial_merge_size is not None:
        config.spatial_merge_size = spatial_merge_size
    return config
