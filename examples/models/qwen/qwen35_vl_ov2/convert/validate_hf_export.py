# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check every inference tensor against the exported HF model on the meta device.

No GPU and no weight allocation. This checks keys, shapes and shard/index
integrity. Tensor VALUES are only certified when ``verify_export_parity.py`` has
left an ``export_parity.json`` next to the weights (EP-layout invariance of the
whole export, see that script); ``ep_invariance_verified`` reports that evidence,
and ``--require-ep-invariance`` makes it mandatory. Full numerical parity
remains unverified until a comparison against the training SAVE is available. HF/MCore logit parity is still out of scope for both. MTP is an
auxiliary training head ignored by the HF inference model; its presence is
reported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

PARITY_REPORT = "export_parity.json"


def _parity_module():
    """Load the sibling parity tool by path (this file is executed as a script, not imported)."""
    path = Path(__file__).resolve().parent / "verify_export_parity.py"
    spec = importlib.util.spec_from_file_location("ov2_verify_export_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ep_invariance_evidence(root: Path) -> dict:
    """Summarise the EP-invariance evidence attached to this export, if any.

    Three things must hold before a report counts, because each of them has a silent-pass failure mode:
      * the recorded manifest digest still matches the weights on disk -- otherwise a report left behind by
        an earlier export of a DIFFERENT iteration (same shapes, same layout, same headers) would certify
        the new one;
      * the report's candidate provenance matches this export's own stamp -- a report can be copied in;
      * the two compared exports really ran at different expert-parallel sizes on the same checkpoint --
        verify_export_parity.py enforces this from the stamps, and it is re-checked here.

    Note what a PASS is worth: EP invariance only. Both compared exports ran the same mapping registry, so
    an EP-independent mapping error passes; this is never evidence of agreement with the training SAVE.
    """
    report_path = root / PARITY_REPORT
    if not report_path.exists():
        return {"verified": False, "status": "absent", "detail": f"no {PARITY_REPORT} (run verify_export_parity.py)"}
    try:
        report = json.loads(report_path.read_text())
    except ValueError as exc:
        return {"verified": False, "status": "unreadable", "detail": str(exc)}
    if report.get("verdict") != "PASS":
        return {"verified": False, "status": "failed", "detail": f"verdict={report.get('verdict')!r}"}
    ref_stamp = report.get("reference_provenance") or {}
    cand_stamp = report.get("candidate_provenance") or {}
    module = _parity_module()
    try:
        live_stamp = module.read_provenance(root)
    except Exception as exc:
        return {"verified": False, "status": "unstamped", "detail": f"{type(exc).__name__}: {exc}"}
    if cand_stamp != live_stamp:
        return {"verified": False, "status": "foreign", "detail": "report was produced for a different export"}
    ref_ep, cand_ep = ref_stamp.get("expert_parallel_size"), cand_stamp.get("expert_parallel_size")
    if not ref_ep or ref_ep == cand_ep:
        return {"verified": False, "status": "vacuous", "detail": f"both sides at EP={cand_ep}"}
    if ref_stamp.get("source_checkpoint") != cand_stamp.get("source_checkpoint"):
        return {"verified": False, "status": "mismatched_source", "detail": "compared exports differ in source"}
    try:
        live_digest = module.manifest_digest(module.manifest(root))
    except Exception as exc:  # unreadable shards are the shard check's job to report
        return {"verified": False, "status": "unverifiable", "detail": f"{type(exc).__name__}: {exc}"}
    if report.get("candidate_manifest_digest") != live_digest:
        return {"verified": False, "status": "stale", "detail": "parity report belongs to different weights"}
    return {
        "verified": True,
        "status": "ok",
        "check": report.get("check"),
        "compared_eps": [ref_ep, cand_ep],
        "iteration": cand_stamp.get("iteration"),
        "tensors_compared": report.get("tensors_compared"),
    }


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


GAPS = [
    "not compared against the training SAVE (EP invariance shares the export mapping on both sides)",
    "no HF-vs-mcore logits on the same input",
    "MTP values unchecked (HF inference never instantiates the MTP head)",
]


def check_shapes(actual: dict, expected: dict, *, require_mtp: bool, parity: dict | None = None) -> dict:
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
    parity = parity or {"verified": False, "status": "not_checked"}
    return {
        "inference_tensors_checked": len(expected),
        "mtp_keys": len(mtp),
        # Deliberately NOT called numerical_parity: EP invariance is one differential test, and a PASS is
        # not evidence that the weights match the SAVE. numerical_parity_verified stays false until a
        # check against the checkpoint (or same-input logits) exists; see GAPS.
        "ep_invariance_verified": bool(parity.get("verified")),
        "ep_invariance": parity,
        "numerical_parity_verified": False,
        "numerical_parity_gaps": GAPS,
    }


def main() -> None:
    """Validate an export before allowing its .export_done marker to be written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--config-only", action="store_true", help="Check HF construction before GPU export")
    parser.add_argument(
        "--require-ep-invariance",
        action="store_true",
        help="Fail unless a matching export_parity.json shows this export is EP-reshard invariant",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_export_config(args.root, require_mtp=args.require_mtp)
    expected = expected_shapes(args.root)
    if args.config_only:
        logger.info("[q35-export-check] HF config/meta-model preflight PASS: %s inference tensors", len(expected))
        return
    parity = ep_invariance_evidence(args.root)
    if args.require_ep_invariance and not parity["verified"]:
        raise ValueError(
            f"EP-invariance evidence required but {parity['status']}: {parity['detail']}. "
            "Run verify_export_parity.py against a second export of this iteration at a different OV2_EP."
        )
    report = check_shapes(tensor_inventory(args.root), expected, require_mtp=args.require_mtp, parity=parity)
    (args.root / "export_structure_check.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("[q35-export-check] ALL inference keys/shapes/shards PASS: %s", report)
    if report["ep_invariance_verified"]:
        logger.info("EP-reshard invariance holds (EP%s vs EP%s).", *parity["compared_eps"])
    else:
        logger.info("EP-reshard invariance NOT established (%s).", parity["status"])
    logger.info("Values are still unproven against the SAVE: %s", "; ".join(GAPS))


if __name__ == "__main__":
    main()
