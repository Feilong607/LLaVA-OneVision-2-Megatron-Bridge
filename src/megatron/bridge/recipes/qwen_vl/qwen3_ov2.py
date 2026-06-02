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

"""SFT recipe for Qwen3-VL-8B (dense) grafted with the OV2.1 onevision vision tower.

Wraps :func:`megatron.bridge.recipes.qwen_vl.qwen3_vl.qwen3_vl_8b_sft_config`
and swaps its dense :class:`Qwen3VLModelProvider` for our
:class:`Qwen3OV2ModelProvider` (vision tower = OV2.1 onevision encoder +
adapter, random init at build time).

Notes
-----
* Mirror of :mod:`qwen35_ov2` for the 35B MoE sibling. 8B is dense, so the
  recipe defaults to TP=2 PP=1 (matching :func:`qwen3_vl_8b_sft_config`).
* As with the 35B recipe, the swap rebuilds the provider object from raw
  dataclass fields — vision weights are reloaded from the OV2.1 checkpoint
  via ``+model.ov2_vision_ckpt_path``, and LLM weights are reloaded via
  ``checkpoint.pretrained_checkpoint`` (or omitted entirely for a smoke
  run that doesn't care about converged loss).
"""

from __future__ import annotations

import dataclasses

from megatron.bridge.models.qwen_vl.qwen3_ov2_provider import Qwen3OV2ModelProvider
from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_8b_sft_config
from megatron.bridge.training.config import ConfigContainer


def _swap_provider_to_ov2(
    cfg: ConfigContainer,
) -> None:
    """Replace ``cfg.model`` with a :class:`Qwen3OV2ModelProvider` carrying the
    same fields as the original :class:`Qwen3VLModelProvider`.

    We copy every dataclass field defined on the original (and inherited by
    our subclass) by name, then ``Qwen3OV2ModelProvider`` is constructed
    fresh — this avoids both ``dataclasses.asdict`` (deep-copy issues) and
    ``dataclasses.replace`` (can't change the type).
    """
    original = cfg.model
    target_field_names = {f.name for f in dataclasses.fields(Qwen3OV2ModelProvider)}
    init_field_names = {
        f.name for f in dataclasses.fields(Qwen3OV2ModelProvider) if f.init
    }
    kwargs = {}
    for f in dataclasses.fields(original):
        if f.name in init_field_names:
            kwargs[f.name] = getattr(original, f.name)
    new_provider = Qwen3OV2ModelProvider(**kwargs)
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


def qwen3_ov2_8b_sft_config(
    hf_path: str = "/ov2/pretrain_models/Qwen3-VL-8B-Instruct",
) -> ConfigContainer:
    """Return a full SFT config for Qwen3-VL-8B (dense) with OV2.1 vision grafted on.

    Build flow:

      1. Call the upstream :func:`qwen3_vl_8b_sft_config` to get a fully
         populated ``ConfigContainer`` (model/parallel/training/optimizer/etc.)
         with ``cfg.model`` being a :class:`Qwen3VLModelProvider`.
      2. Swap ``cfg.model`` for our :class:`Qwen3OV2ModelProvider`, copying
         every dataclass field by name.
      3. Replace the default HF dataset with EnergonProvider + OV2 task
         encoder pointed at the OV2 packed WDS smoke dataset.

    Args:
        hf_path: HuggingFace model ID or local path to the Qwen3-VL-8B model
            directory. Used by ``AutoBridge`` to build the LLM-side provider
            and by the SFT dataset for ``hf_processor_path``.
    """
    # Upstream recipe hardcodes "Qwen/Qwen3-VL-8B-Instruct"; we override by
    # patching cfg.model.hf_processor_path after the swap.
    cfg = qwen3_vl_8b_sft_config()
    _swap_provider_to_ov2(cfg)

    # ---- OV2 packed WDS wiring (replaces the default HF SFT dataset) ----
    # The actual data path and ckpt paths are typically overridden via hydra
    # CLI; defaults below point at the smoke-test dataset on /ov2.
    from megatron.bridge.data.energon.energon_provider import EnergonProvider
    from megatron.bridge.data.qwen_vl_ov2 import OV2PackingTaskEncoder
    from transformers import AutoTokenizer, AutoImageProcessor

    _ov2_tokenizer = AutoTokenizer.from_pretrained(hf_path)
    # Qwen3-VL-8B's tokenizer doesn't expose ``image_token_id`` as an attribute
    # (unlike the Qwen3.5 family). OV2PackingTaskEncoder requires it, so we
    # patch it on from the matching token (<|image_pad|> = 151655 for Qwen3-VL).
    if not hasattr(_ov2_tokenizer, "image_token_id") or _ov2_tokenizer.image_token_id is None:
        _ov2_tokenizer.image_token_id = _ov2_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    _ov2_image_processor = AutoImageProcessor.from_pretrained(
        "/ov2/feilong/preprocessor_p16m33"
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
