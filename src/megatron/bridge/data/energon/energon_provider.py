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

from dataclasses import dataclass
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


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
    task_encoder: Optional[Any] = None
    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False
    packing_buffer_size: Optional[int] = None
    shuffle_buffer_size: Optional[int] = None
    num_val_workers: Optional[int] = None

    def build_datasets(self, context: DatasetBuildContext):
        assert self.path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"

        # Sync seq_length from the (already hydra-resolved) provider config onto
        # the task encoder, which was constructed during recipe build before
        # hydra CLI overrides applied. Without this, a custom task encoder
        # carries the stale recipe-default seq_length.
        if hasattr(self.task_encoder, "seq_length") and self.task_encoder.seq_length != self.seq_length:
            print(f"[EnergonProvider] sync task_encoder.seq_length {self.task_encoder.seq_length} -> {self.seq_length}")
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
            **({'packing_buffer_size': self.packing_buffer_size} if self.packing_buffer_size is not None else {}),
            **({'shuffle_buffer_size': self.shuffle_buffer_size} if self.shuffle_buffer_size is not None else {}),
            **({'num_val_workers': self.num_val_workers} if self.num_val_workers is not None else {}),
        )
        train_iter = iter(dataset.train_dataloader())
        # Fall back to train if dataset has no val/test split (common for OV2 packed WDS).
        try:
            from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
        except Exception:
            EmptyDatasetError = Exception
        def _safe(maker):
            try:
                return iter(maker())
            except (EmptyDatasetError, KeyError, Exception) as _e:
                print(f"[EnergonProvider] val/test split unavailable ({type(_e).__name__}: {_e}) -> fallback to train dataloader")
                return iter(dataset.train_dataloader())
        val_iter = _safe(dataset.val_dataloader)
        test_iter = _safe(dataset.val_dataloader)
        return (train_iter, val_iter, test_iter)
