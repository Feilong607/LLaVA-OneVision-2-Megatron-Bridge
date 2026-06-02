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

"""SFT recipe for Qwen3-VL-30B-A3B (MoE) grafted with the OV2.1 onevision vision tower.

Wraps :func:`megatron.bridge.recipes.qwen_vl.qwen3_vl.qwen3_vl_30b_a3b_sft_config`
and swaps its :class:`Qwen3VLMoEModelProvider` for our
:class:`Qwen3OV230BModelProvider` (vision tower = OV2.1 onevision encoder +
adapter, with optional load of OV2 stage_0 weights for both).

Notes
-----
* Mirror of :mod:`qwen3_ov2` (8B dense) and :mod:`qwen35_ov2` (35B MoE).
* Default parallel layout TP=1 PP=1 EP=8 matches the OV2 stage_0 ckpt
  exactly (``stage_0_tp1_pp1_ep8``), so vision/adapter EP-replicated reads
  from rank-0 and LLM expert EP-shards line up 1:1 with the 8-way EP layout.
* ``cfg.dataset.seq_length = 8192`` is smaller than OV2 prod (32768) for
  headroom; oversize sample skipping in :class:`OV2PackingTaskEncoder`
  handles long records.
"""

from __future__ import annotations

import dataclasses

from megatron.bridge.models.qwen_vl.qwen3_ov2_30b_provider import Qwen3OV230BModelProvider
from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_30b_a3b_sft_config
from megatron.bridge.training.config import ConfigContainer


def _swap_provider_to_ov2(
    cfg: ConfigContainer,
) -> None:
    """Replace ``cfg.model`` with a :class:`Qwen3OV230BModelProvider` carrying the
    same fields as the original :class:`Qwen3VLMoEModelProvider`.

    We copy every dataclass field defined on the original (and inherited by
    our subclass) by name, then ``Qwen3OV230BModelProvider`` is constructed
    fresh — this avoids both ``dataclasses.asdict`` (deep-copy issues) and
    ``dataclasses.replace`` (can't change the type).
    """
    original = cfg.model
    target_field_names = {f.name for f in dataclasses.fields(Qwen3OV230BModelProvider)}
    init_field_names = {
        f.name for f in dataclasses.fields(Qwen3OV230BModelProvider) if f.init
    }
    kwargs = {}
    for f in dataclasses.fields(original):
        if f.name in init_field_names:
            kwargs[f.name] = getattr(original, f.name)
    new_provider = Qwen3OV230BModelProvider(**kwargs)
    # Carry over runtime-set attributes that aren't dataclass fields (e.g.
    # ``_pg_collection``). Skip dataclass-field names so the subclass's own
    # ``__post_init__`` normalization wins.
    for attr_name, val in vars(original).items():
        if attr_name in target_field_names:
            continue
        try:
            setattr(new_provider, attr_name, val)
        except (AttributeError, TypeError):
            pass

    cfg.model = new_provider


def qwen3_ov2_30b_a3b_sft_config(
    hf_path: str = "/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507",
) -> ConfigContainer:
    """Return a full SFT config for Qwen3-VL-30B-A3B (MoE) with OV2.1 vision grafted on.

    Build flow:

      1. Call upstream :func:`qwen3_vl_30b_a3b_sft_config` to get a fully
         populated ``ConfigContainer`` with ``cfg.model`` =
         :class:`Qwen3VLMoEModelProvider`.
      2. Swap ``cfg.model`` for our :class:`Qwen3OV230BModelProvider`,
         copying every dataclass field by name.
      3. Replace the default mock VLM dataset with EnergonProvider + OV2
         packing task encoder pointed at the OV2 packed WDS smoke dataset.
      4. Patch ``image_token_id`` onto the tokenizer (Qwen3-30B-A3B's HF
         tokenizer doesn't expose it as an attribute; OV2PackingTaskEncoder
         requires it).

    Args:
        hf_path: HuggingFace model ID or local path to the Qwen3-30B-A3B
            model directory. Used by ``AutoBridge`` to build the LLM-side
            provider and by the SFT dataset for ``hf_processor_path``.

    Defaults:
      - TP=1 PP=1 EP=8 (matches OV2 stage_0_tp1_pp1_ep8 exactly)
      - seq_length=8192 (smaller than OV2 prod 32768; task encoder skips oversize)
    """
    cfg = qwen3_vl_30b_a3b_sft_config()
    _swap_provider_to_ov2(cfg)

    # Bridge's upstream recipe uses the HF id "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # but we want to point AutoBridge at our local mirror of the Qwen3-30B-A3B
    # (text-only) HF model. We don't re-run AutoBridge here (we'd re-fetch
    # weights and rebuild the provider); instead we patch hf_processor_path on
    # the dataset side. The provider already carries the correct config from
    # the upstream call.
    # NOTE: cfg.model.pretrained_model_name remains the upstream Qwen3-VL id;
    # downstream consumers (HF processor lookups) use hf_processor_path below.

    # Shrink seq_length from upstream default to give headroom on single-node.
    cfg.dataset.seq_length = 8192
    cfg.model.seq_length = 8192

    # ---- OV2 packed WDS wiring (replaces the default mock dataset) ----
    from megatron.bridge.data.energon.energon_provider import EnergonProvider
    from megatron.bridge.data.qwen_vl_ov2 import OV2PackingTaskEncoder
    from transformers import AutoTokenizer, AutoImageProcessor

    _ov2_tokenizer = AutoTokenizer.from_pretrained(hf_path)
    # Qwen3-30B-A3B-Instruct-2507's tokenizer doesn't expose ``image_token_id``
    # as an attribute (same as the 8B Qwen3-VL family). Patch it on from the
    # matching token (<|image_pad|> = 151655, verified for OV2).
    if not hasattr(_ov2_tokenizer, "image_token_id") or _ov2_tokenizer.image_token_id is None:
        _ov2_tokenizer.image_token_id = _ov2_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    _ov2_image_processor = AutoImageProcessor.from_pretrained(
        "/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5"
    )
    _ov2_task_encoder = OV2PackingTaskEncoder(
        tokenizer=_ov2_tokenizer,
        image_processor=_ov2_image_processor,
        seq_length=cfg.dataset.seq_length,
    )
    cfg.dataset = EnergonProvider(
        path="/ov2/dataset_mid/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-Packed-384332",
        image_processor=_ov2_image_processor,
        seq_length=cfg.dataset.seq_length,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=2,
        dataloader_type="external",
        task_encoder=_ov2_task_encoder,
        pack_sequences_in_batch=True,
    )

    return cfg
