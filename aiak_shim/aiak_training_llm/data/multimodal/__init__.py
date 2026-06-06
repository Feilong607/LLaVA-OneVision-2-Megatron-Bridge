"""Shim exposing ``aiak_training_llm.data.multimodal.MultiMixQASample``.

The original WebDataset ``/vlm/data/blip_laion_cc_sbu_558k_wds/.nv-meta/dataset.yaml``
references this dotted path as its ``sample_type``. Megatron-Energon must be able to
import the class to load the dataset. This standalone copy mirrors AIAK's
``MultiMixQASample`` (an Energon ``Sample``) and adds three read-only compatibility
properties (``conversation`` / ``imgs`` / ``videos``) so Bridge's ``QwenVLTaskEncoder``
— which expects ``ChatMLSample``-style attributes — consumes it unchanged.

Put ``<repo>/aiak_shim`` on PYTHONPATH (see examples/.../qwen3_vl/sft_4b_558k.sh).
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from megatron.energon.flavors.base_dataset import Sample


@dataclass
class MultiMixQASample(Sample):
    """Non-packed image/video QA conversation sample (mirrors AIAK's MultiMixQASample).

    Fields are populated by the dataset's ``.nv-meta/sample_loader.py`` (which emits
    ``messages`` / ``system`` / ``image``); the rest default and stay unused for the
    blip_laion_cc_sbu_558k caption data.
    """

    messages: List[dict]
    video: list = None
    image: List[torch.Tensor] = None
    system: Optional[str] = None
    patch_positions: Optional[List[np.ndarray]] = None
    fps: Optional[Union[float, int]] = None
    timestamp_decimal: Optional[int] = None

    # --- Compatibility with Bridge's QwenVLTaskEncoder (expects ChatMLSample attrs) ---
    @property
    def conversation(self) -> str:
        """JSON string of the conversation; the encoder calls cook_chatml_sample on it.

        A leading system turn (when present) is prepended so cook_chatml_sample's
        odd-length system handling picks it up.
        """
        msgs = list(self.messages)
        if self.system is not None:
            msgs = [{"role": "system", "content": self.system}, *msgs]
        return json.dumps(msgs)

    @property
    def imgs(self):
        return self.image

    @property
    def videos(self):
        return self.video


@dataclass
class PackedCaptioningSample(Sample):
    """Offline-packed multi-image captioning sample (mirrors AIAK PackedCaptioningSample).
    One energon sample = N sub-samples: images[sample_idx][img_idx], prompts[i], captions[i]."""

    images: List
    prompts: Optional[List] = None
    captions: Optional[List] = None
    patch_positions: Optional[List] = None
    fps: Optional[List] = None
    timestamp_decimal: Optional[List] = None


__all__ = ["MultiMixQASample", "PackedCaptioningSample"]
