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


def get_vision_config(model_family=None, model_name: str = "llava-onevision2-30b-a3b"):
    """Build the OV2.1 vision tower config.

    Mirrors ``aiak_training_llm.models.llava_onevision2.llava_onevision2_config.
    get_vision_config``.  ``model_family`` is accepted for signature
    compatibility but unused.  ``model_name`` selects between the default
    1024-d / 24-layer vision tower used by the 30B-A3B LLM and the larger
    1664-d / 48-layer ``vision-2b`` variant.
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
    if model_name and "vision-2b" in model_name:
        config.num_layers = 48
        config.hidden_size = 1664
        config.ffn_hidden_size = 8192
        config.kv_channels = 104
    elif model_name == "llava-onevision2-layer1":
        config.num_layers = 1
    return config
