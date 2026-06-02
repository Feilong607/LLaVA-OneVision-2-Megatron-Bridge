"""OV2 packed-captioning data pipeline for Megatron-Bridge.

Public names:
    PackedCaptioningSample  -- the WDS sample dataclass (Energon ``Sample``)
    OV2ImageProcessor       -- HF Qwen2VLImageProcessor wrapper with smart_resize
    OV2PackingTaskEncoder   -- energon ``DefaultTaskEncoder`` that emits Bridge dicts
    ImageTaskSample         -- intermediate single sub-sample dataclass
    ImageTaskSamplePacked   -- post-pack, pre-batch dataclass
    ImageTaskBatchPacked    -- post-batch dataclass
    greedy_knapsack         -- ported OV2 knapsack helper
    pack_selected_samples   -- ported OV2 packing helper
    build_ov2_packed_dataset -- thin energon dataset factory
    IGNORE_INDEX            -- the -100 label sentinel
"""

from .image_processor import OV2ImageProcessor, smart_resize
from .packed_captioning import PackedCaptioningSample
from .task_encoder import (
    IGNORE_INDEX,
    ImageTaskBatchPacked,
    ImageTaskSample,
    ImageTaskSamplePacked,
    OV2PackingTaskEncoder,
    greedy_knapsack,
    pack_selected_samples,
    print_error_handler,
)

try:  # dataset builder pulls in megatron.energon.get_train_dataset; keep optional.
    from .dataset_builder import build_ov2_packed_dataset
except Exception:  # pragma: no cover - optional path
    build_ov2_packed_dataset = None  # type: ignore[assignment]

__all__ = [
    "PackedCaptioningSample",
    "OV2ImageProcessor",
    "smart_resize",
    "OV2PackingTaskEncoder",
    "ImageTaskSample",
    "ImageTaskSamplePacked",
    "ImageTaskBatchPacked",
    "greedy_knapsack",
    "pack_selected_samples",
    "print_error_handler",
    "build_ov2_packed_dataset",
    "IGNORE_INDEX",
]
