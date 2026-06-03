"""OV2.1 onevision-encoder + adapter, ported into Megatron-Bridge for grafting onto Qwen3.5-35B-A3B."""
from .onevision_encoder_model import OneVisionEncoderModel
from .adapter import Adapter, AdapterSubmodules
from .vision_config import VisionConfig, get_vision_config
from .layer_spec import get_vision_layer_spec
from .adapter_layer_spec import get_adapter_layer_spec

__all__ = [
    "OneVisionEncoderModel",
    "Adapter",
    "AdapterSubmodules",
    "VisionConfig",
    "get_vision_config",
    "get_vision_layer_spec",
    "get_adapter_layer_spec",
]
