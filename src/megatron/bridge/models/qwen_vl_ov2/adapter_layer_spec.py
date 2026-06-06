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
