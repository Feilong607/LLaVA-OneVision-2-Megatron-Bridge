# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, CPU-only input-order evidence for paired OV2 experiments."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

import torch


_COUNTS: dict[str, int] = {}


def batch_digest(batch: Mapping[str, object]) -> str:
    """Hash text/labels/geometry and image shape, without copying images or CUDA data.

    This detects different sample ordering and packing. It deliberately does not
    hash image contents and must not be presented as full-input equality proof.
    """
    digest = hashlib.sha256()
    for key in (
        "tokens",
        "input_ids",
        "labels",
        "loss_mask",
        "attention_mask",
        "cu_seqlens",
        "image_grid_thw",
        "patch_positions",
        "pixel_values",
    ):
        value = batch.get(key)
        digest.update(key.encode())
        if value is None:
            digest.update(b"none")
            continue
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"Ablation fingerprint expects a CPU tensor for {key}")
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        if key != "pixel_values":
            digest.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def record_batch(batch: Mapping[str, object]) -> None:
    """Append one metadata fingerprint before the existing CPU-to-CUDA transfer."""
    root = os.environ["OV2_AB_INPUT_DIR"]
    rank = int(os.environ["RANK"])
    path = Path(root) / f"rank{rank:03d}.txt"
    key = str(path)
    count = _COUNTS.get(key, 0) + 1
    _COUNTS[key] = count
    digest = batch_digest(batch)
    # Parent is prepared by the experiment controller. Reject a stale run at the
    # first batch, rather than appending a second incarnation to old evidence.
    with path.open("x" if count == 1 else "a") as stream:
        stream.write(f"{count} {digest}\n")
