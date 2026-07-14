# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    dataloader_load: Optional[str] = None
    task_encoder: Optional[Any] = None
    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False

    def build_datasets(self, context: DatasetBuildContext):
        assert self.path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"
        # Upstream #4342 (minimal port): a CLI/config seq_length override updates THIS provider but the
        # task encoder was already built with its import-time value -> re-sync here so the encoder pads/
        # packs to the length the model actually runs. No-op for the OV2 launchers (OV2_SEQ_LEN is the
        # single source for both), load-bearing only for dataset.seq_length= CLI overrides.
        if self.task_encoder is not None and hasattr(self.task_encoder, "seq_length"):
            self.task_encoder.seq_length = self.seq_length
        if (
            self.pack_sequences_in_batch
            and self.task_encoder is not None
            and hasattr(self.task_encoder, "pack_sequences")
        ):
            self.task_encoder.pack_sequences = True
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            pg_collection=context.pg_collection,
            dataloader_load=self.dataloader_load,
        )
        train_iter = iter(dataset.train_dataloader())
        # Datasets prepared with only a 'train' split (e.g. the original
        # blip_laion_cc_sbu_558k_wds) have an empty/missing 'val' split, which
        # would otherwise raise EmptyDatasetError here. Fall back to the train
        # dataloader for val/test so the run proceeds; set validation.eval_iters=0
        # to skip evaluation entirely.
        try:
            val_iter = iter(dataset.val_dataloader())
        except Exception as e:
            logger.warning(
                "EnergonProvider: no usable 'val' split (%s); reusing the train "
                "dataloader for val/test. Set validation.eval_iters=0 to skip eval.",
                e,
            )
            val_iter = iter(dataset.train_dataloader())
        return (train_iter, val_iter, val_iter)
