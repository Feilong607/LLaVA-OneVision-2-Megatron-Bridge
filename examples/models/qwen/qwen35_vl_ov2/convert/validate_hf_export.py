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


def check_export_config(root: Path, *, require_mtp: bool) -> None:
    """Reject incompatible skeleton settings together, before loading GPU weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    text = config.text_config
    vision = config.vision_config
    wanted = {
        "use_post_layernorm": True,
        "use_head": False,
        "zero_centered_gamma": True,
        "merger_zero_centered_gamma": True,
        "layer_norm_type": "layer_norm",
        "layer_norm_eps": 1e-5,
        "post_layernorm_eps": 1e-5,
        "pre_layernorm_eps": 1e-4,
        "merger_layernorm_eps": text.rms_norm_eps,
        "out_hidden_size": text.hidden_size,
    }
    errors = [
        f"vision.{key}: {getattr(vision, key, None)!r} != {value!r}"
        for key, value in wanted.items()
        if getattr(vision, key, None) != value
    ]
    if config.architectures != ["LlavaOnevision2ForConditionalGeneration"]:
        errors.append(f"architectures: {config.architectures!r}")
    mtp_layers = int(getattr(text, "mtp_num_hidden_layers", 0) or 0)
    if require_mtp and mtp_layers != 1:
        errors.append(f"MTP export supports one configured layer; found {mtp_layers}")
    if errors:
        raise ValueError("Incompatible Qwen3.5 export skeleton:\n" + "\n".join(errors))


def tensor_inventory(root: Path) -> dict[str, tuple[int, ...]]:
    """Inspect actual safetensors shards and reject missing/index-only tensors."""
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.exists() and (root / "model.safetensors").exists():
        raise ValueError(
            "Ambiguous HF weights: both model.safetensors and its shard index exist; isolate stale output first"
        )
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
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    if config.text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expected Qwen3.5 MoE text configuration")
    if config.image_token_id != 248056 or config.tie_word_embeddings:
        raise ValueError("Incorrect image token or tied embeddings")
    if not config.text_config.rope_parameters.get("mrope_section"):
        raise ValueError("Missing M-RoPE configuration")
    reference = "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration"
    if (getattr(config, "auto_map", None) or {}).get("AutoModelForCausalLM") != reference:
        raise ValueError("AutoModelForCausalLM must point to the exported local OV2 class; rebuild the skeleton")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
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
    parser.add_argument("--config-only", action="store_true", help="Check HF construction before GPU export")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_export_config(args.root, require_mtp=args.require_mtp)
    expected = expected_shapes(args.root)
    if args.config_only:
        logger.info("[q35-export-check] HF config/meta-model preflight PASS: %s inference tensors", len(expected))
        return
    report = check_shapes(tensor_inventory(args.root), expected, require_mtp=args.require_mtp)
    (args.root / "export_structure_check.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("[q35-export-check] ALL inference keys/shapes/shards PASS: %s", report)
    logger.info("MTP values, EP resharding and HF/MCore logits still require numerical validation.")


if __name__ == "__main__":
    main()
