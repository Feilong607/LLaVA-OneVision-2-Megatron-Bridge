# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OV2 Bridge-native recipes (3 backbones: Qwen3-4B, Qwen3-8B, Qwen3.5-35B-A3B MoE)."""
from .ov2 import (
    OV2EnergonProvider,
    _ov2_backbone_paths,
    _ov2_common,
    # per-size recipe functions
    ov2_4b_stage1,
    ov2_4b_stage2,
    ov2_8b_stage1,
    ov2_8b_stage2,
    ov2_35b_a3b_stage1,
    ov2_35b_a3b_stage2,
    ov2_30b_a3b_p16m33_stage1,
    ov2_30b_a3b_p16m33_stage2,
    # mid-train (stage 1.5) — full-model SFT
    ov2_4b_midtrain,
    ov2_8b_midtrain,
    ov2_35b_a3b_midtrain,
    ov2_30b_a3b_p16m33_midtrain,
    # back-compat aliases (original 4B names)
    ov2_1_stage1_adapter_only_config,
    ov2_1_stage2_vit_adapter_muon_config,
)

__all__ = [
    "OV2EnergonProvider",
    "_ov2_backbone_paths",
    "_ov2_common",
    "ov2_4b_stage1",
    "ov2_4b_stage2",
    "ov2_8b_stage1",
    "ov2_8b_stage2",
    "ov2_35b_a3b_stage1",
    "ov2_35b_a3b_stage2",
    "ov2_30b_a3b_p16m33_stage1",
    "ov2_30b_a3b_p16m33_stage2",
    "ov2_4b_midtrain",
    "ov2_8b_midtrain",
    "ov2_35b_a3b_midtrain",
    "ov2_30b_a3b_p16m33_midtrain",
    "ov2_1_stage1_adapter_only_config",
    "ov2_1_stage2_vit_adapter_muon_config",
]
