"""Vision-to-LLM adapter (projector) for the OV2.1 onevision encoder.

Ported verbatim from ``aiak_training_llm.models.llava_onevision2.adapter`` —
the source file imports nothing from ``aiak_training_llm`` so no
transformations were needed beyond moving it under the qwen_vl_ov2 package.
"""

from dataclasses import dataclass
from typing import Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class Adapter(MegatronModule):
    """Adaptor"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AdapterSubmodules,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = input_size * (spatial_merge_size**2)

        self.use_patch_position_encoding = getattr(config, "use_patch_position_encoding", False)
        # Type of encoding: only supports "absolute"
        self.patch_position_encoding_type = getattr(config, "patch_position_encoding_type", "absolute")

        if self.use_patch_position_encoding:
            max_pos = getattr(config, "max_position_embeddings", 8192)

            if self.patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unsupported patch_position_encoding_type: {self.patch_position_encoding_type}. "
                    "Only 'absolute' is supported."
                )

            self.pos_emb_h = torch.nn.Embedding(max_pos, output_size)
            self.pos_emb_w = torch.nn.Embedding(max_pos, output_size)

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

        self.activation_func = config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

    def forward(self, x: torch.Tensor, patch_positions: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        x = self.layernorm(x).view(-1, self.hidden_size)
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)

        if self.use_patch_position_encoding and patch_positions is not None:
            # patch_positions is [num_patches, 3] (t, h, w)
            # Need to reshape and process for spatial merge
            pp = patch_positions.view(-1, self.spatial_merge_size**2, 3)
            # Take the position of the first patch in the merged group
            pp = pp[:, 0, :]
            # Downsample positions according to spatial merge
            pp = (pp // self.spatial_merge_size).long()

            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])

        return x
