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

"""SFT recipe for Qwen3.5-35B-A3B grafted with the OV2.1 onevision vision tower.

Wraps :func:`megatron.bridge.recipes.qwen_vl.qwen35_vl.qwen35_vl_35b_a3b_sft_config`
and swaps its ``Qwen35VLMoEModelProvider`` for our
:class:`Qwen35OV2MoEModelProvider` (vision tower = OV2.1 onevision encoder +
adapter, random init at build time).

Notes
-----
* AutoBridge loads weights bound to the HF Qwen3.5 model into the original
  provider; by swapping the provider we *drop* that binding for the vision
  tower. That's intentional — vision weights will be reloaded from the OV2.1
  checkpoint via a separate ``checkpoint.pretrained_checkpoint`` step. LLM
  weights still need to be wired through ``checkpoint.pretrained_checkpoint``
  as well, since the swap rebuilds the provider object from raw fields.
"""

from __future__ import annotations

import dataclasses

from megatron.bridge.models.qwen_vl.qwen35_ov2_provider import Qwen35OV2MoEModelProvider
from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_35b_a3b_sft_config
from megatron.bridge.training.config import ConfigContainer


def _swap_provider_to_ov2(
    cfg: ConfigContainer,
) -> None:
    """Replace ``cfg.model`` with a ``Qwen35OV2MoEModelProvider`` carrying the
    same fields as the original Qwen3.5 provider.

    We copy every dataclass field defined on the original (and inherited by
    our subclass) by name, then ``Qwen35OV2MoEModelProvider`` is constructed
    fresh — this avoids both ``dataclasses.asdict`` (which would deep-copy
    non-dataclass fields like ``vision_config``) and ``dataclasses.replace``
    (which can't change the type).
    """
    original = cfg.model
    # Collect every dataclass field of the original (it's a Qwen35VLMoEModelProvider)
    # that is also a field of our target class — i.e. all of them, since
    # Qwen35OV2MoEModelProvider extends Qwen35VLMoEModelProvider.
    target_field_names = {f.name for f in dataclasses.fields(Qwen35OV2MoEModelProvider)}
    init_field_names = {
        f.name for f in dataclasses.fields(Qwen35OV2MoEModelProvider) if f.init
    }
    kwargs = {}
    for f in dataclasses.fields(original):
        if f.name in init_field_names:
            kwargs[f.name] = getattr(original, f.name)
    new_provider = Qwen35OV2MoEModelProvider(**kwargs)
    # Carry over any non-init / runtime-set instance attributes (e.g.
    # ``_pg_collection``) that the parent ``__post_init__`` set directly on
    # the original instance's ``__dict__``. We deliberately do NOT walk
    # ``dir(original)`` because it would also touch ``@property`` descriptors
    # like ``meta_model`` whose getters trigger heavy work.
    for attr_name, val in vars(original).items():
        # Already-passed init fields were copied via ``kwargs`` and re-validated
        # by ``__post_init__``; overwriting them here would defeat the
        # subclass's own ``__post_init__`` normalization. We only carry over
        # *runtime-set* attributes that aren't dataclass fields (e.g.
        # ``_pg_collection``, anything stashed by AutoBridge).
        if attr_name in target_field_names:
            continue
        try:
            setattr(new_provider, attr_name, val)
        except (AttributeError, TypeError):
            pass

    cfg.model = new_provider


def qwen35_ov2_35b_a3b_sft_config(
    hf_path: str = "/ov2/pretrain_models/Qwen3.5-35B-A3B",
) -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-35B-A3B with OV2.1 vision grafted on.

    Build flow:

      1. Call the upstream :func:`qwen35_vl_35b_a3b_sft_config` to get a fully
         populated ``ConfigContainer`` (model/parallel/training/optimizer/etc.)
         with ``cfg.model`` being a :class:`Qwen35VLMoEModelProvider`.
      2. Swap ``cfg.model`` for our :class:`Qwen35OV2MoEModelProvider`, copying
         every dataclass field by name.

    Args:
        hf_path: HuggingFace model ID or local path to the Qwen3.5-35B-A3B
            model directory. Used by ``AutoBridge`` to build the LLM-side
            provider and by the SFT dataset for ``hf_processor_path``.
    """
    cfg = qwen35_vl_35b_a3b_sft_config(hf_path=hf_path)
    _swap_provider_to_ov2(cfg)

    # ---- OV2 packed WDS wiring (replaces the mock HFDatasetConversationProvider) ----
    # The actual path and ckpt paths are typically overridden via hydra CLI; defaults
    # below point at the smoke-test dataset on /ov2.
    from megatron.bridge.data.energon.energon_provider import EnergonProvider
    from megatron.bridge.data.qwen_vl_ov2 import OV2PackingTaskEncoder
    from transformers import AutoTokenizer, AutoImageProcessor

    _ov2_tokenizer = AutoTokenizer.from_pretrained(hf_path)
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
