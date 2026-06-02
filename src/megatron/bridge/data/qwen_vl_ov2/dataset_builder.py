"""Thin Energon factory for the OV2 packed-captioning webdataset.

Loads the offline-packed WDS at ``data_path`` (any directory containing a
``.nv-meta/`` subdir with ``dataset.yaml`` + ``sample_loader.py``), with the
sample type rebound to our Bridge-side ``PackedCaptioningSample`` and the
``OV2PackingTaskEncoder`` plugged in as the task encoder.
"""

from __future__ import annotations

from typing import Optional


def build_ov2_packed_dataset(
    data_path: str,
    *,
    tokenizer,
    image_processor,
    seq_length: int,
    split: str = "train",
    worker_config=None,
    batch_size: Optional[int] = None,
    packing_buffer_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int] = None,
    repeat: bool = True,
    chat_template_name: str = "qwen2-vl",
    **kwargs,
):
    """Build an Energon training dataset over the OV2 packed WDS.

    Notes:
        * Returns an Energon ``SavableDataset`` of ``ImageTaskBatchPacked``
          objects — call ``OV2PackingTaskEncoder.encode_batch(batch)`` per step
          to convert to the Bridge dict format.
        * If energon raises because the WDS ``dataset.yaml`` points at the
          OV2-side ``aiak_training_llm.data.multimodal.PackedCaptioningSample``,
          patch the metadataset loader to rebind to our class (see
          ``cookers``/``crude_sample`` docs); for the verification harness we
          construct ``PackedCaptioningSample`` records directly so this isn't
          required.
    """
    from megatron.energon import (
        get_train_dataset,
        WorkerConfig,
    )

    from .task_encoder import OV2PackingTaskEncoder

    task_encoder = OV2PackingTaskEncoder(
        tokenizer=tokenizer,
        image_processor=image_processor,
        seq_length=seq_length,
        chat_template_name=chat_template_name,
    )

    wc = worker_config if worker_config is not None else WorkerConfig.default_worker_config()

    dataset = get_train_dataset(
        data_path,
        batch_size=batch_size or 1,
        task_encoder=task_encoder,
        worker_config=wc,
        packing_buffer_size=packing_buffer_size,
        shuffle_buffer_size=shuffle_buffer_size,
        repeat=repeat,
        **kwargs,
    )
    return dataset, task_encoder
