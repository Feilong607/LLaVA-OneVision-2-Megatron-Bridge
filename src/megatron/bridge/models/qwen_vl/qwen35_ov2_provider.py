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

"""Provider for Qwen3.5-35B-A3B with OV2.1 onevision-encoder grafted as the vision tower.

Reuses :class:`Qwen35VLMoEModelProvider`'s LLM build and only replaces the
vision tower + projection with OV2's :class:`OneVisionEncoderModel` +
:class:`Adapter`. Bridge's :class:`Qwen3VLModel` expects::

    self.vision_model(hidden_states, grid_thw) -> (vision_embeds, deepstack_feature_lists)

so we wrap OV2 vision + adapter behind an :class:`OV2VisionTower` that returns
``(vision_embeds, [])`` (OV2 has no deepstack features; the 35B vision_config
also has ``deepstack_visual_indexes=[]`` so this matches).

LLM hidden = adapter output = 2048 (Qwen3.5-35B-A3B uses 2048-d LLM hidden).
OV2 vision hidden = 1024. ``Adapter.linear_fc2`` therefore projects 4096 ->
2048 (input_size * spatial_merge_size**2 = 1024*4 = 4096 -> output_size = 2048).

User-chosen policy:
  - Vision weights: loaded later from OV2.1 checkpoint (random init at build time)
  - Adapter weights: random init (no loader path).

Both are deferred to the checkpoint loader step; this file only assembles the
modules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider
from megatron.bridge.models.qwen_vl_ov2 import (
    Adapter,
    OneVisionEncoderModel,
    VisionConfig,
    get_adapter_layer_spec,
    get_vision_config,
    get_vision_layer_spec,
)


class OV2VisionTower(nn.Module):
    """Match the public surface of :class:`Qwen3VLVisionModel` so the rest of
    :class:`Qwen3VLModel` can call us unchanged.

    Bridge's :class:`Qwen3VLModel` interacts with ``self.vision_model`` via:

      * ``forward(hidden_states=vision_data, grid_thw=vision_grid_thw)`` returning
        ``(vision_embeds, deepstack_feature_lists)`` — see
        ``Qwen3VLModel.forward`` around line 427-447.
      * ``freeze()`` accessing ``self.vision_model.decoder.deepstack_merger_list``
        and ``self.vision_model.merger`` and iterating their ``.parameters()`` —
        see ``Qwen3VLModel.freeze`` around line 300-331.

    Our tower wires up:

      * ``self.vit`` (an :class:`OneVisionEncoderModel`) → patches -> 1024-d tokens
      * ``self.adapter`` (an :class:`Adapter`) → 1024-d tokens -> 2048-d LLM tokens

    and exposes ``self.merger`` (the adapter, the only "projection" we have) and
    ``self.decoder`` (a stub whose ``deepstack_merger_list`` is an empty
    ``ModuleList`` — OV2 has no deepstack features) to satisfy ``freeze()``.
    """

    def __init__(
        self,
        vision_config: VisionConfig,
        adapter_config,
        vision_layer_spec,
        adapter_layer_spec,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        # OV2 vision encoder. ``spatial_merge_size`` IS a kwarg at this level.
        self.vit = OneVisionEncoderModel(
            vision_config,
            vision_layer_spec,
            spatial_merge_size=spatial_merge_size,
        )
        # Vision -> LLM projector. Adapter's forward applies the spatial-merge
        # reshape inside, so its conceptual input dim per merged token is
        # ``input_size * spatial_merge_size**2`` (= 4096 for 1024 * 4).
        self.adapter = Adapter(
            adapter_config,
            adapter_layer_spec,
            input_size=vision_config.hidden_size,
            output_size=adapter_config.hidden_size,
            spatial_merge_size=spatial_merge_size,
        )
        # Empty stub list to satisfy Qwen3VLModel.freeze()'s
        # ``vision_model.decoder.deepstack_merger_list.parameters()`` access.
        # OV2 has no deepstack features, so iterating its parameters is a no-op.
        self._deepstack_merger_list = nn.ModuleList()

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        """Run OV2 vision + adapter and return (vision_embeds, deepstack_list).

        OV2 ViT signature is ``forward(x, grid_thw, patch_positions=None)``;
        OV2 Adapter signature is ``forward(x, patch_positions=None)``.
        Qwen3VLModel expects a 2-tuple where the second element is a (possibly
        empty) list of deepstack feature maps — OV2 produces none, so we return
        ``[]``.
        """
        x = self.vit(hidden_states, grid_thw=grid_thw)
        x = self.adapter(x, patch_positions=None)
        return x, []

    # -- shims to keep Qwen3VLModel.freeze() happy --------------------------
    # freeze() does:
    #   modules.append(self.vision_model.decoder.deepstack_merger_list)
    #   modules.append(self.vision_model.merger)
    #   for module in modules: for param in module.parameters(): ...
    # We just need attribute paths that yield something with ``.parameters()``.

    @property
    def decoder(self):
        # Return a lightweight object with the only attribute freeze() touches.
        return _DecoderShim(self._deepstack_merger_list)

    @property
    def merger(self):
        return self.adapter


class _DecoderShim:
    """Minimal stand-in for the original ``Qwen3VLVisionModel.decoder`` attribute.

    Only used by :meth:`Qwen3VLModel.freeze` to access
    ``decoder.deepstack_merger_list``. Holding the real ``ModuleList`` (rather
    than constructing a fresh empty one each property access) keeps any
    ``param.requires_grad`` assignments stable across calls.
    """

    def __init__(self, deepstack_merger_list: nn.ModuleList) -> None:
        self.deepstack_merger_list = deepstack_merger_list


@dataclass
class Qwen35OV2MoEModelProvider(Qwen35VLMoEModelProvider):
    """Qwen3.5-35B-A3B LLM + OV2.1 vision tower + OV2.1 adapter (random init at build time).

    The LLM (language_model + embedding) is built by the parent provider; we
    then *replace* ``model.vision_model`` with an :class:`OV2VisionTower`.

    Pair with :func:`megatron.bridge.recipes.qwen_vl.qwen35_ov2.
    qwen35_ov2_35b_a3b_sft_config` for a ready-made SFT config.
    """

    # OV2 vision-tower knobs (exposed for hydra-style override). Defaults match
    # the OV2.1 onevision encoder checkpoint that ships with the 30B-A3B
    # release: 24-layer / 1024-hidden / patch=14 / spatial merge 2.
    ov2_vision_patch_size: int = 14
    ov2_vision_hidden_size: int = 1024
    ov2_vision_num_layers: int = 24
    ov2_vision_spatial_merge_size: int = 2
    # Optional: path to OV2 reshard ckpt dir (with release/mp_rank_00_<r>/model_optim_rng.pt).
    # If set, provide() loads vision_model.* tensors into the tower right after build.
    ov2_vision_ckpt_path: Optional[str] = None

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> Qwen3VLModel:
        """Build the parent model, then swap in the OV2 vision tower."""
        model = super().provide(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Only the pre-process (embedding-holding) PP stage runs the vision
        # tower in Bridge's Qwen3VLModel. On non-pre-process ranks the parent's
        # ``model.vision_model`` is already ``None`` and we leave it alone.
        if getattr(model, "vision_model", None) is None:
            return model

        # Build OV2 vision config by deep-copying the LLM TransformerConfig
        # (``self``) and overlaying VisionConfig's fields on top. This is the
        # OV2 reference pattern (see
        # ``aiak_training_llm.models.llava_onevision2.llava_onevision2_provider.
        # llavaov_2_model_provider``). It is required because
        # ``OneVisionEncoderModel.__init__`` calls ``super().__init__(config)``
        # which is ``MegatronModule.__init__`` and asserts
        # ``isinstance(config, TransformerConfig)``. A plain ``VisionConfig``
        # dataclass fails that check.
        vision_config = deepcopy(self)
        for k, v in asdict(get_vision_config()).items():
            setattr(vision_config, k, v)

        # The vision tower lives entirely on the pre-process PP stage and must
        # never inherit the LLM's PP/CP knobs (otherwise its TransformerBlock
        # would try to split its (small) layer count across the language model's
        # PP layout).
        vision_config.pipeline_model_parallel_size = 1
        vision_config.first_pipeline_num_layers = None
        vision_config.last_pipeline_num_layers = None
        vision_config.tp_comm_overlap = False
        vision_config.context_parallel_size = 1

        # Scrub LLM-only enhancement flags inherited from the parent (Qwen3.5-VL) config.
        # These DON'T exist on plain VisionConfig so the asdict() overlay above cannot
        # clear them. The most damaging is attention_output_gate, which inflates
        # SelfAttention's linear_qkv_out_dim by `num_heads * kv_channels` (a gate column),
        # turning the ViT's qkv from (3072, 1024) into (4096, 1024) and breaking the
        # OV2 ckpt shape match. qk_layernorm would add absent q/k_layernorm weights.
        for _llm_flag, _vision_default in (
            ("attention_output_gate", False),
            ("qk_layernorm", False),
            ("qk_l2_norm", False),
            ("multi_latent_attention", False),
            ("num_moe_experts", None),
            ("moe_router_topk", None),
            ("moe_grouped_gemm", False),
            ("moe_use_legacy_grouped_gemm", False),
            ("moe_shared_expert_intermediate_size", None),
            ("moe_ffn_hidden_size", None),
            ("mtp_num_layers", None),
            ("mtp_loss_scaling_factor", None),
        ):
            if hasattr(vision_config, _llm_flag):
                setattr(vision_config, _llm_flag, _vision_default)

        # Fill init helpers if the parent left them as None (Bridge providers can defer
        # init_method binding to weight-load time; our single-tower build path doesn't
        # trigger that, so fill from init_method_std).
        from megatron.core.utils import init_method_normal, scaled_init_method_normal
        if vision_config.init_method is None:
            vision_config.init_method = init_method_normal(vision_config.init_method_std)
        if vision_config.output_layer_init_method is None:
            vision_config.output_layer_init_method = scaled_init_method_normal(
                vision_config.init_method_std, vision_config.num_layers
            )
        # Some providers also leave embedding_init_method None.
        if getattr(vision_config, "embedding_init_method", "MISSING") is None:
            vision_config.embedding_init_method = init_method_normal(
                vision_config.init_method_std
            )

        # Re-apply user-facing ov2_vision_* overrides after the overlay so they
        # win over both ``self`` and the OV2 VisionConfig defaults.
        vision_config.patch_size = self.ov2_vision_patch_size
        vision_config.hidden_size = self.ov2_vision_hidden_size
        vision_config.num_layers = self.ov2_vision_num_layers

        # Adapter config: also a deepcopy of ``self`` (a TransformerConfig). We
        # did not port an OV2 ``get_adapeter_config`` helper — the adapter is
        # random-init at build time and its only structural requirement is
        # ``adapter_config.hidden_size == 2048`` (matches the LLM hidden, which
        # is what Adapter.__init__'s output_size=adapter_config.hidden_size
        # consumes). ``self.hidden_size`` is exactly 2048 for 35B-A3B.
        adapter_config = deepcopy(self)

        v_spec = get_vision_layer_spec()
        a_spec = get_adapter_layer_spec()

        tower = OV2VisionTower(
            vision_config=vision_config,
            adapter_config=adapter_config,
            vision_layer_spec=v_spec,
            adapter_layer_spec=a_spec,
            spatial_merge_size=self.ov2_vision_spatial_merge_size,
        )
        tower.to(dtype=self.params_dtype)

        model.vision_model = tower

        # Optionally splice OV2.1's pretrained vision weights into the tower.
        # Adapter is intentionally NOT loaded (random init per design).
        if self.ov2_vision_ckpt_path:
            from megatron.bridge.models.qwen_vl.ov2_vision_weight_loader import (
                load_ov2_vision_into_tower,
            )
            summary = load_ov2_vision_into_tower(tower, self.ov2_vision_ckpt_path)
            print(
                f"[Qwen35OV2] OV2 vision load: loaded={summary['loaded_count']} "
                f"missing={len(summary['missing'])} unexpected={len(summary['unexpected'])} "
                f"path={self.ov2_vision_ckpt_path}"
            )

        # Re-apply freeze policy after the swap so the new tower picks up the
        # ``freeze_vision_model`` / ``freeze_vision_projection`` flags.
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model
