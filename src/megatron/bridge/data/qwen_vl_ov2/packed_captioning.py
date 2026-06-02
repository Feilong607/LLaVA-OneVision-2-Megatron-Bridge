"""PackedCaptioningSample for OV2-style packed captioning data.

Ported verbatim from the OV2 source at
``aiak_training_llm/data/multimodal/flavors/packed_captioning.py`` with the
``aiak`` import removed — energon's ``Sample`` base is used directly.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from megatron.energon.flavors.base_dataset import Sample


@dataclass
class PackedCaptioningSample(Sample):
    """Sample type emitted by the OV2 offline-packed WDS.

    Each record stores a *list* of (prompt, caption, image) triples that the
    offline packer chose to co-locate. The downstream task encoder is expected
    to encode each sub-sample independently and then concatenate them into one
    long packed sequence.

    Fields mirror the JSON written by
    ``tools/data_preprocess/offline_packing/convert_packedsample_to_wds.py``.
    """

    images: List[torch.Tensor]
    prompts: Optional[List[str]]
    captions: List[str]
    # patch_positions[sub_sample_idx][image_idx] -> np.ndarray of [n_patches, 3]
    patch_positions: Optional[List[List[np.ndarray]]] = None
    # fps[sub_sample_idx] -> fps or None (videos only)
    fps: Optional[List[Optional[Union[float, int]]]] = None
    # timestamp_decimal[sub_sample_idx] -> 1 or 2 or None (video timestamp granularity)
    timestamp_decimal: Optional[List[Optional[int]]] = None
