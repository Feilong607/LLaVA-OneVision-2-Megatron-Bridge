"""Shim: re-export the OV2 energon sample types by their AIAK dotted path so
energon's dataset_loader can resolve ``aiak_training_llm.data.multimodal.<Sample>``
referenced in the WDS ``.nv-meta/dataset.yaml``.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from megatron.energon.flavors.base_dataset import Sample

from megatron.bridge.data.qwen_vl_ov2.packed_captioning import PackedCaptioningSample


@dataclass
class MultiMixQASample(Sample):
    """Non-packed conversation sample (image/video QA). Mirrors AIAK's MultiMixQASample."""

    messages: List[dict]
    video: list = None
    image: List[torch.Tensor] = None
    system: Optional[str] = None
    patch_positions: Optional[List[np.ndarray]] = None
    fps: Optional[Union[float, int]] = None
    timestamp_decimal: Optional[int] = None


__all__ = ["PackedCaptioningSample", "MultiMixQASample"]
