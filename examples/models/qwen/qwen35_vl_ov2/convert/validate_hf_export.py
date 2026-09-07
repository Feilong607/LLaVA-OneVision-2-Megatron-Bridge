# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check every inference tensor against the exported HF model on the meta device.

No GPU and no weight allocation. This checks keys, shapes and shard/index
integrity, NOT tensor values or HF/MCore numerical parity. MTP is an auxiliary
training head ignored by the HF inference model; its presence is reported.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def tensor_inventory(root: Path) -> dict[str, tuple[int, ...]]:
    """Inspect actual safetensors shards and reject missing/index-only tensors."""
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    files = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    inventory = {}
    observed_map = {}
    for filename in files:
        path = root / filename
        if path.resolve().parent != root.resolve():
            raise ValueError(f"Invalid shard path: {filename}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in inventory:
                    raise ValueError(f"Duplicate tensor across shards: {key}")
                view = handle.get_slice(key)
                if view.get_dtype() not in ("BF16", "F16", "F32"):
                    raise ValueError(f"Unexpected tensor dtype: {key} {view.get_dtype()}")
                inventory[key] = tuple(view.get_shape())
                observed_map[key] = filename
    if weight_map is not None and observed_map != weight_map:
        raise ValueError("Safetensors index does not match actual shard contents")
    return inventory


def expected_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    """Instantiate the actual exported remote-code class with meta tensors only."""
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    if config.text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expected Qwen3.5 MoE text configuration")
    if config.image_token_id != 248056 or config.tie_word_embeddings:
        raise ValueError("Incorrect image token or tied embeddings")
    if not config.text_config.rope_parameters.get("mrope_section"):
        raise ValueError("Missing M-RoPE configuration")
    reference = "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration"
    model_class = get_class_from_dynamic_module(reference, root, local_files_only=True)
    with torch.device("meta"):
        model = model_class(config)
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def check_shapes(actual: dict, expected: dict, *, require_mtp: bool) -> dict:
    """Reject missing inference tensors, shape errors and legacy expert layouts."""
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(key for key in actual.keys() - expected.keys() if not key.startswith("mtp."))
    mismatched = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]]
    mtp = [key for key in actual if key.startswith("mtp.")]
    if missing or extra or mismatched or (require_mtp and not mtp):
        raise ValueError(
            f"missing={missing} unexpected={extra} wrong_shapes={mismatched} "
            f"mtp_keys={len(mtp)} require_mtp={require_mtp}"
        )
    return {"inference_tensors_checked": len(expected), "mtp_keys": len(mtp), "numerical_parity_verified": False}


def main() -> None:
    """Validate an export before allowing its .export_done marker to be written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-mtp", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = check_shapes(tensor_inventory(args.root), expected_shapes(args.root), require_mtp=args.require_mtp)
    (args.root / "export_structure_check.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("[q35-export-check] ALL inference keys/shapes/shards PASS: %s", report)
    logger.info("MTP values, EP resharding and HF/MCore logits still require numerical validation.")


if __name__ == "__main__":
    main()
