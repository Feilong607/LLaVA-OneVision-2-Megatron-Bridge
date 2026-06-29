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

"""Adapter (vision-to-LLM projector) layer spec for the OV2.1 onevision encoder.

Ported from
``aiak_training_llm.models.qwen_vl.qwen2_vl_layer_spec.get_adapeter_layer_with_spec``
(note the typo "adapeter" in the source — normalised to "adapter" here).

The source uses ``LocalNorm`` (an apex-backed FusedLayerNorm / FusedRMSNorm
wrapper) for the layernorm sub-module. Megatron-Bridge runs on TE only, so we
substitute ``TENorm`` (which is what the OV2 ``OneVisionEncoderModel`` already
uses for its own ``pre_layernorm``). ``TENorm`` picks LayerNorm vs RMSNorm
based on ``config.normalization`` just like ``LocalNorm`` does — behaviorally
equivalent for our purposes.

The Adapter class itself (and its ``AdapterSubmodules`` dataclass) lives in
``adapter.py``; we re-export from there.
"""

from megatron.core.extensions.transformer_engine import TELinear, TENorm

from .adapter import AdapterSubmodules


def get_adapter_layer_spec() -> AdapterSubmodules:
    """Return AdapterSubmodules with TE-backed layernorm + two TELinear projections.

    This mirrors ``get_adapeter_layer_with_spec`` from the source aiak module,
    with ``LocalNorm`` replaced by ``TENorm`` (TE-only environment).
    """
    return AdapterSubmodules(
        layernorm=TENorm,
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )
