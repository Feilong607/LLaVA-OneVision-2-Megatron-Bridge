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

# Register the AutoBridge (HF LlavaOnevision2ForConditionalGeneration <-> mcore LlavaOnevision2).
# Importing here runs the @MegatronModelBridge.register_bridge decorator so AutoBridge can dispatch.
from .ov2_bridge import LlavaOnevision2MoEBridge  # noqa: E402,F401
__all__ += ["LlavaOnevision2MoEBridge"]
