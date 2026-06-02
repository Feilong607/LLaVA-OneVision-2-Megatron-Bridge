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

"""Provider for Qwen3-VL-30B-A3B (MoE) with OV2.1 onevision-encoder grafted as the vision tower.

Mirror of :mod:`qwen3_ov2_provider` (which targets the 8B dense Qwen3-VL) but
built on top of Bridge's :class:`Qwen3VLMoEModelProvider` (the 30B-A3B /
235B-A22B MoE family). The LLM backbone is Qwen3-30B-A3B (HF
``Qwen3-30B-A3B-Instruct-2507`` — text-only, 48-layer, 128-expert, 2048
hidden, top-k=8).

We graft OV2.1's onevision encoder + adapter on top of that MoE LLM by:

  1. Letting the parent provider build the full Qwen3VLModel (MoE LLM +
     a placeholder Qwen3VLVisionModel on the pre-process PP stage).
  2. Replacing ``model.vision_model`` with an :class:`OV2VisionTower`
     (OV2.1's :class:`OneVisionEncoderModel` + :class:`Adapter`).
  3. Optionally splicing OV2's pretrained vision_model.* tensors into
     ``tower.vit`` (via ``ov2_vision_ckpt_path``).
  4. Optionally splicing OV2's pretrained adapter.* tensors into
     ``tower.adapter`` (via ``ov2_adapter_ckpt_path``). This is NEW vs the
     8B/35B providers — for OV2 stage_1 alignment training we resume from
     the stage_0 adapter rather than starting random.

LLM hidden = adapter output = 2048 (Qwen3-30B-A3B hidden_size).
OV2 vision hidden = 1024. ``Adapter.linear_fc2`` therefore projects
input_size*spatial_merge_size**2 = 1024*4 = 4096 -> output_size = 2048.
This matches the OV2 stage_0 ckpt's adapter.linear_fc2.weight shape
(2048, 4096) verified by ``/tmp/inspect_ov2_stage0.py``.

User-chosen policy:
  - Vision weights: loaded from OV2.1 stage_0 ckpt if ov2_vision_ckpt_path set
  - Adapter weights: loaded from OV2.1 stage_0 ckpt if ov2_adapter_ckpt_path set
    (essential for stage_1 alignment continuation), else random init.

LLM weights are NOT loaded by this provider — they come through Bridge's
standard ``checkpoint.pretrained_checkpoint`` path. (For the initial smoke
that path is left empty; LLM starts random — see launcher TODO.)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLMoEModelProvider
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

    See :mod:`qwen3_ov2_provider.OV2VisionTower` for the full contract; this
    class is functionally identical and exists here so the MoE 30B provider
    doesn't have to cross-import from the dense 8B provider module.

    LLM hidden = adapter output = 2048 for Qwen3-30B-A3B (matching OV2 stage_0).
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
        self.vit = OneVisionEncoderModel(
            vision_config,
            vision_layer_spec,
            spatial_merge_size=spatial_merge_size,
        )
        self.adapter = Adapter(
            adapter_config,
            adapter_layer_spec,
            input_size=vision_config.hidden_size,
            output_size=adapter_config.hidden_size,
            spatial_merge_size=spatial_merge_size,
        )
        # Empty stub list to satisfy Qwen3VLModel.freeze()'s
        # ``vision_model.decoder.deepstack_merger_list.parameters()`` access.
        self._deepstack_merger_list = nn.ModuleList()

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        """Run OV2 vision + adapter and return (vision_embeds, deepstack_list)."""
        x = self.vit(hidden_states, grid_thw=grid_thw)
        x = self.adapter(x, patch_positions=None)
        return x, []

    @property
    def decoder(self):
        return _DecoderShim(self._deepstack_merger_list)

    @property
    def merger(self):
        return self.adapter


class _DecoderShim:
    """Minimal stand-in for the original ``Qwen3VLVisionModel.decoder`` attribute.

    Only used by :meth:`Qwen3VLModel.freeze` to access
    ``decoder.deepstack_merger_list``.
    """

    def __init__(self, deepstack_merger_list: nn.ModuleList) -> None:
        self.deepstack_merger_list = deepstack_merger_list


@dataclass
class Qwen3OV230BModelProvider(Qwen3VLMoEModelProvider):
    """Qwen3-VL-30B-A3B (MoE) LLM + OV2.1 vision tower + OV2.1 adapter.

    The LLM (language_model + embedding) is built by the parent MoE provider;
    we then *replace* ``model.vision_model`` with an :class:`OV2VisionTower`
    and (optionally) splice OV2 stage_0 weights into both the vision tower
    and the adapter.

    Pair with :func:`megatron.bridge.recipes.qwen_vl.qwen3_ov2_30b.
    qwen3_ov2_30b_a3b_sft_config` for a ready-made SFT config.
    """

    # OV2 vision-tower knobs (exposed for hydra-style override). Defaults match
    # the OV2.1 onevision encoder checkpoint shipped with the 30B-A3B release:
    # 24-layer / 1024-hidden / patch=14 / spatial merge 2.
    ov2_vision_patch_size: int = 14
    ov2_vision_hidden_size: int = 1024
    ov2_vision_num_layers: int = 24
    ov2_vision_spatial_merge_size: int = 2
    # Optional: path to OV2 reshard ckpt dir (with release/mp_rank_00_<r>/model_optim_rng.pt).
    # If set, provide() loads vision_model.* tensors into the tower right after build.
    ov2_vision_ckpt_path: Optional[str] = None
    # NEW vs 8B/35B: optional path to load OV2's pretrained adapter.* weights.
    # When set, provide() loads adapter.* tensors into the tower right after build,
    # overriding random adapter init. Essential for resuming OV2 stage_0 -> stage_1.
    # Same on-disk layout as ov2_vision_ckpt_path (legacy mcore release/mp_rank_00_<r>/).
    # The adapter has only 6 weight tensors and is EP-replicated, so reading rank 0
    # is sufficient (verified by /tmp/inspect_ov2_stage0.py).
    ov2_adapter_ckpt_path: Optional[str] = None

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> Qwen3VLModel:
        """Build the parent MoE model, then swap in the OV2 vision tower."""
        model = super().provide(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Only the pre-process (embedding-holding) PP stage runs the vision
        # tower in Bridge's Qwen3VLModel. On non-pre-process PP ranks the
        # parent's ``model.vision_model`` is already ``None``.
        if getattr(model, "vision_model", None) is None:
            return model

        # Build OV2 vision config by deep-copying the LLM TransformerConfig
        # (``self``) and overlaying VisionConfig's fields on top. Required
        # because OneVisionEncoderModel.__init__ asserts isinstance(config,
        # TransformerConfig).
        vision_config = deepcopy(self)
        for k, v in asdict(get_vision_config()).items():
            setattr(vision_config, k, v)

        # The vision tower lives entirely on the pre-process PP stage and must
        # never inherit the LLM's PP/CP/EP knobs.
        vision_config.pipeline_model_parallel_size = 1
        vision_config.first_pipeline_num_layers = None
        vision_config.last_pipeline_num_layers = None
        vision_config.tp_comm_overlap = False
        vision_config.context_parallel_size = 1

        # Scrub LLM-only enhancement flags inherited from the parent
        # (Qwen3-VL MoE) config. The most damaging is ``attention_output_gate``
        # which inflates SelfAttention's qkv_out_dim and breaks OV2 ckpt
        # shapes. The MoE flags MUST also be cleared since the vision tower
        # is dense.
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
            ("moe_router_pre_softmax", False),
            ("moe_router_dtype", None),
            ("moe_router_score_function", None),
            ("moe_token_dispatcher_type", None),
            ("moe_permute_fusion", False),
            ("mtp_num_layers", None),
            ("mtp_loss_scaling_factor", None),
            ("expert_model_parallel_size", 1),
        ):
            if hasattr(vision_config, _llm_flag):
                setattr(vision_config, _llm_flag, _vision_default)

        # Fill init helpers if the parent left them as None.
        from megatron.core.utils import init_method_normal, scaled_init_method_normal
        if vision_config.init_method is None:
            vision_config.init_method = init_method_normal(vision_config.init_method_std)
        if vision_config.output_layer_init_method is None:
            vision_config.output_layer_init_method = scaled_init_method_normal(
                vision_config.init_method_std, vision_config.num_layers
            )
        if getattr(vision_config, "embedding_init_method", "MISSING") is None:
            vision_config.embedding_init_method = init_method_normal(
                vision_config.init_method_std
            )

        # Re-apply user-facing ov2_vision_* overrides after the overlay.
        vision_config.patch_size = self.ov2_vision_patch_size
        vision_config.hidden_size = self.ov2_vision_hidden_size
        vision_config.num_layers = self.ov2_vision_num_layers

        # Adapter config: deepcopy of ``self`` (a TransformerConfig). Adapter's
        # only structural requirement is ``adapter_config.hidden_size == LLM
        # hidden`` (= 2048 for Qwen3-30B-A3B). ``self.hidden_size`` is exactly
        # that for the MoE provider.
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

        # Optionally splice OV2.1's pretrained vision weights into tower.vit.
        if self.ov2_vision_ckpt_path:
            from megatron.bridge.models.qwen_vl.ov2_vision_weight_loader import (
                load_ov2_vision_into_tower,
            )
            summary = load_ov2_vision_into_tower(tower, self.ov2_vision_ckpt_path)
            print(
                f"[Qwen3OV230B] OV2 vision load: loaded={summary['loaded_count']} "
                f"missing={len(summary['missing'])} unexpected={len(summary['unexpected'])} "
                f"path={self.ov2_vision_ckpt_path}"
            )

        # Optionally splice OV2.1's pretrained adapter weights into tower.adapter.
        # This is NEW for the 30B MoE provider: stage_1 alignment continues
        # from the stage_0 adapter (a 6-tensor block).
        if self.ov2_adapter_ckpt_path:
            from megatron.bridge.models.qwen_vl.ov2_vision_weight_loader import (
                load_ov2_adapter_into_tower,
            )
            a_summary = load_ov2_adapter_into_tower(tower, self.ov2_adapter_ckpt_path)
            print(
                f"[Qwen3OV230B] OV2 adapter load: loaded={a_summary['loaded_count']} "
                f"missing={len(a_summary['missing'])} unexpected={len(a_summary['unexpected'])} "
                f"path={self.ov2_adapter_ckpt_path}"
            )

        # Re-apply freeze policy after the swap.
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model
