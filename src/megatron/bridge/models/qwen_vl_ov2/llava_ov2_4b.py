# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Back-compat shim. This module was renamed to ``llava_ov2.py``.

The builder here handles 4B / 8B / 30B-A3B (backbone selected by HF path + per-backbone
config) — the original ``_4b`` name was historical (the 4B port was verified first). Kept as a
re-export so any external ``...qwen_vl_ov2.llava_ov2_4b import X`` still works.
"""
from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import *  # noqa: F401,F403
from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import (  # noqa: F401  explicit re-exports
    LlavaOnevision2,
    build_llava_ov2,
    build_llava_ov2_4b,
    load_ov2_mcore_checkpoint,
    load_ov2_4b_mcore_checkpoint,
    convert_hf_onevision_to_mcore,
    load_hf_encoder_into_tower,
)
