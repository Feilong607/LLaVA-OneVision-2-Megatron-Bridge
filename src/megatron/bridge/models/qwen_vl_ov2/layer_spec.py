"""Vision-side transformer layer spec for the OV2.1 onevision encoder.

Extracted from
``aiak_training_llm.models.llava_onevision2.llava_onevision2_layer_spec``
keeping ONLY:

* ``rotate_half``
* ``apply_rotary_pos_emb_vision``
* ``get_vision_layer_with_spec`` (re-exported as ``get_vision_layer_spec``
  so the package matches the qwen_vl_ov2 ``__init__`` naming)

The original aiak file also imported
``get_adapeter_layer_with_spec`` / ``get_qwen_layer_with_te_spec`` from the
v1_5 layer-spec module, but those helpers were only used by the
language-model side and were never referenced from the vision spec returned
here — they have been dropped.
"""

import torch
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def rotate_half(x):
    """
    Perform interleaved rotation for rotary position embedding.

    Rotates half of the dimensions in an interleaved pattern to match the source model's implementation.
    Transforms (x1, x2, x3, x4) -> (-x2, x1, -x4, x3).

    Args:
        x (torch.Tensor): Input tensor to rotate.

    Returns:
        torch.Tensor: Rotated tensor with the same shape as input.
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb_vision(t, freqs, config, cu_seqlens=None, rotary_interleaved=False, **kwargs):
    """
    Apply rotary position embedding to vision tokens.

    Args:
        t (torch.Tensor): Input tensor to apply rotary embedding.
        freqs (torch. Tensor): Frequency tensor for rotary embedding.
        config:  Model configuration.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for packed sequences.
        rotary_interleaved (bool, optional): Whether to use interleaved rotation pattern.

    Returns:
        torch.Tensor: Tensor with rotary position embedding applied.
    """
    orig_dtype = t.dtype
    t = t.float()
    if cu_seqlens is not None:
        freqs = freqs.squeeze(1)
        cos_ = freqs.cos().float().repeat(1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 2)
    else:
        cos_ = freqs.cos().float().repeat(1, 1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 1, 2)
    t = (t * cos_) + (rotate_half(t) * sin_)
    return t.to(orig_dtype)


def get_vision_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_vision,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Public alias matching the qwen_vl_ov2 package __init__ naming convention.
get_vision_layer_spec = get_vision_layer_with_spec
