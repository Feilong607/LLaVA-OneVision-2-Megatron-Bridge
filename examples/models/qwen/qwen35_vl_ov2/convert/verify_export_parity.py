# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EP-layout invariance check for a Qwen3.5 OV2 HF export. This is ONE differential test, not "the export
is numerically correct" -- read the scope below before quoting a PASS.

Exporting the SAME iteration twice at two different expert-parallel sizes must yield bit-identical
weights: resharding only moves values between ranks, and every mapping (QKV concat, packed
``experts.gate_up_proj``, GDN conv1d, zero-centered norms) is a deterministic index operation on them.

PROVES (a differing byte is a real defect):
  * the EP8 -> EP_N torch_dist reshard on 256 routed experts + shared experts + MTP, where the 30B
    EP-reshard evidence does not transfer;
  * any expert remap, rank-ownership or gather bug whose result depends on the EP layout;
  * that the two runs read the same source checkpoint (enforced from the export's own provenance stamp).

DOES NOT PROVE:
  * agreement with the training SAVE. Both sides run the SAME mapping registry and the SAME export code,
    so a mapping that is wrong at EVERY EP (wrong key pairing, wrong transform, a tensor silently dropped
    and randomly initialised) is identical on both sides and PASSES here. Only a comparison against the
    checkpoint -- or same-input HF-vs-mcore logits -- can close that;
  * HF-vs-mcore logits, M-RoPE at inference, or MTP semantics.

Compares RAW safetensors bytes: no torch, no dequantisation, no HF import, so it runs on any CPU pod and
does not re-execute the export code path it is checking. numpy is used only to describe a mismatch.

Usage (after both exports exist; EP and source are read from each dir's export_provenance.json):
    python3 verify_export_parity.py <reference_export> <candidate_export>

Writes ``export_parity.json`` into the candidate (production) export and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# safetensors: 8-byte little-endian header length, then the JSON header, then the tensor data block.
_HEADER_LEN_FIELD = 8
_MAX_HEADER_BYTES = 100_000_000
_CHUNK = 8 << 20

# Byte widths of the dtypes an OV2 export may legitimately contain (validate_hf_export.py rejects others).
_DTYPE_WIDTH = {"BF16": 2, "F16": 2, "F32": 4}


class ParityError(ValueError):
    """Raised when the two exports are not the same model."""


def _read_header(path: Path) -> tuple[dict, int]:
    """Return the safetensors JSON header and the absolute offset of the data block."""
    with path.open("rb") as handle:
        raw_len = handle.read(_HEADER_LEN_FIELD)
        if len(raw_len) != _HEADER_LEN_FIELD:
            raise ParityError(f"{path.name}: truncated safetensors header length")
        header_len = int.from_bytes(raw_len, "little")
        if not 0 < header_len <= _MAX_HEADER_BYTES:
            raise ParityError(f"{path.name}: implausible safetensors header length {header_len}")
        header = json.loads(handle.read(header_len))
    return header, _HEADER_LEN_FIELD + header_len


def shard_files(root: Path) -> list[str]:
    """Resolve the shard list the same way HF would, rejecting the ambiguous both-layouts case."""
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.exists() and single.exists():
        raise ParityError(f"{root}: both model.safetensors and its shard index exist; isolate stale output first")
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted(set(weight_map.values()))
    if single.exists():
        return ["model.safetensors"]
    raise ParityError(f"{root}: no safetensors weights found")


def manifest_digest(entries: dict[str, dict]) -> str:
    """Content identity of an export: one hash over every tensor's dtype, shape and byte hash.

    Binds a parity record to the exact weights that were compared. A layout-only fingerprint would not:
    re-exporting a DIFFERENT iteration into the same directory produces byte-identical headers (same
    dtypes, shapes and offsets), so only the values can tell the two apart.
    """
    digest = hashlib.sha256()
    for key in sorted(entries):
        entry = entries[key]
        digest.update(f"{key}|{entry['dtype']}|{tuple(entry['shape'])}|{entry['sha256']}\n".encode())
    return digest.hexdigest()


def manifest(root: Path) -> dict[str, dict]:
    """Per-tensor dtype, shape and sha256 of the raw bytes, read straight from the shard files."""
    entries: dict[str, dict] = {}
    for filename in shard_files(root):
        path = root / filename
        if path.resolve().parent != root.resolve():
            raise ParityError(f"{root}: shard escapes the export directory: {filename}")
        header, data_start = _read_header(path)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        # Hash in file order so each shard is read once, sequentially.
        ordered = sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])
        with path.open("rb") as handle:
            for key, meta in ordered:
                if key in entries:
                    raise ParityError(f"{root}: duplicate tensor across shards: {key}")
                start, end = meta["data_offsets"]
                if end < start:
                    raise ParityError(f"{root}: {key} has inverted data offsets")
                width = _DTYPE_WIDTH.get(meta["dtype"])
                if width is None:
                    raise ParityError(f"{root}: {key} has unexpected dtype {meta['dtype']}")
                expected = width
                for dim in meta["shape"]:
                    expected *= dim
                if end - start != expected:
                    raise ParityError(f"{root}: {key} byte range {end - start} != {expected} implied by dtype/shape")
                handle.seek(data_start + start)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = handle.read(min(_CHUNK, remaining))
                    if not block:
                        raise ParityError(f"{root}: {key} is truncated inside {filename}")
                    digest.update(block)
                    remaining -= len(block)
                entries[key] = {
                    "dtype": meta["dtype"],
                    "shape": list(meta["shape"]),
                    "sha256": digest.hexdigest(),
                    "nbytes": end - start,
                    "shard": filename,
                    "offset": data_start + start,
                }
    if not entries:
        raise ParityError(f"{root}: export contains no tensors")
    return entries


def _read_tensor(root: Path, entry: dict):
    """Decode one tensor to float32 for diagnosis (numpy only, bf16 widened by shifting into f32)."""
    import numpy as np

    with (root / entry["shard"]).open("rb") as handle:
        handle.seek(entry["offset"])
        raw = handle.read(entry["nbytes"])
    if entry["dtype"] == "BF16":
        widened = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        values = widened.view(np.float32)
    elif entry["dtype"] == "F16":
        values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        values = np.frombuffer(raw, dtype=np.float32)
    return values.reshape(entry["shape"]) if entry["shape"] else values


def describe_mismatch(reference: Path, candidate: Path, key: str, ref: dict, cand: dict) -> dict:
    """Explain one differing tensor: how far apart, where, and (for expert tensors) which expert."""
    detail = {"key": key, "reference": {k: ref[k] for k in ("dtype", "shape", "sha256")}}
    detail["candidate"] = {k: cand[k] for k in ("dtype", "shape", "sha256")}
    if ref["dtype"] != cand["dtype"] or ref["shape"] != cand["shape"]:
        detail["reason"] = "dtype/shape differ"
        return detail
    try:
        import numpy as np

        a = _read_tensor(reference, ref)
        b = _read_tensor(candidate, cand)
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        flat = int(np.argmax(diff))
        detail["reason"] = "values differ"
        detail["max_abs_diff"] = float(diff.max())
        detail["differing_elements"] = int((diff > 0).sum())
        detail["first_bad_index"] = [int(i) for i in np.unravel_index(flat, diff.shape)] if diff.ndim else [0]
        if diff.ndim >= 2 and ("expert" in key or key.endswith(("gate_up_proj", "down_proj"))):
            # Leading dim of a packed expert tensor is the expert index: name the ones that moved.
            per_expert = diff.reshape(diff.shape[0], -1).max(axis=1)
            bad = np.nonzero(per_expert > 0)[0]
            detail["differing_leading_slices"] = [int(i) for i in bad[:16]]
            detail["differing_leading_slice_count"] = int(bad.size)
    except Exception as exc:  # numpy missing or unreadable shard: the sha256 verdict already stands
        detail["reason"] = f"values differ (diagnosis unavailable: {type(exc).__name__}: {exc})"
    return detail


PROVENANCE = "export_provenance.json"


def read_provenance(root: Path) -> dict:
    """Read the stamp ov2_30b_export_ep8.py leaves in a completed export (EP, source ckpt, iteration)."""
    path = root / PROVENANCE
    if not path.exists():
        raise ParityError(
            f"{root}: no {PROVENANCE}. This export predates provenance stamping, so its expert-parallel "
            "size and source checkpoint cannot be checked -- re-export both sides with the current worker "
            "rather than trusting a hand-written label."
        )
    try:
        stamp = json.loads(path.read_text())
    except ValueError as exc:
        raise ParityError(f"{root}/{PROVENANCE} is not readable JSON: {exc}") from exc
    for field in ("expert_parallel_size", "source_checkpoint"):
        if stamp.get(field) in (None, ""):
            raise ParityError(f"{root}/{PROVENANCE} is missing {field}")
    return stamp


def check_pairing(reference: Path, candidate: Path) -> tuple[dict, dict]:
    """Refuse comparisons that cannot prove anything: same directory, or two unrelated exports."""
    if reference.resolve() == candidate.resolve() or (
        candidate.exists() and reference.exists() and reference.samefile(candidate)
    ):
        raise ParityError(
            "reference and candidate are the same directory (a self-comparison always passes and proves "
            "nothing); pass the second export produced at a different OV2_EP"
        )
    ref_stamp, cand_stamp = read_provenance(reference), read_provenance(candidate)
    if ref_stamp["expert_parallel_size"] == cand_stamp["expert_parallel_size"]:
        raise ParityError(
            f"both exports ran at OV2_EP={cand_stamp['expert_parallel_size']}: this compares a layout with "
            "itself and cannot detect a reshard bug"
        )
    if ref_stamp["source_checkpoint"] != cand_stamp["source_checkpoint"]:
        raise ParityError(
            f"different source checkpoints: {ref_stamp['source_checkpoint']} vs "
            f"{cand_stamp['source_checkpoint']} -- these exports are not comparable"
        )
    if ref_stamp.get("iteration") != cand_stamp.get("iteration"):
        raise ParityError(f"different iterations: {ref_stamp.get('iteration')} vs {cand_stamp.get('iteration')}")
    return ref_stamp, cand_stamp


def compare(reference: Path, candidate: Path, *, max_reported: int = 8) -> dict:
    """Compare two export directories tensor by tensor and return the parity report."""
    ref_stamp, cand_stamp = check_pairing(reference, candidate)
    ref_manifest = manifest(reference)
    cand_manifest = manifest(candidate)
    missing = sorted(ref_manifest.keys() - cand_manifest.keys())
    extra = sorted(cand_manifest.keys() - ref_manifest.keys())
    shared = sorted(ref_manifest.keys() & cand_manifest.keys())
    differing = [
        key
        for key in shared
        if (ref_manifest[key]["sha256"], ref_manifest[key]["dtype"], ref_manifest[key]["shape"])
        != (cand_manifest[key]["sha256"], cand_manifest[key]["dtype"], cand_manifest[key]["shape"])
    ]
    report = {
        "check": "ep_layout_invariance",
        "proves": "the EP reshard/expert remap; NOT agreement with the training SAVE (both sides share the "
        "export mapping, so an EP-independent mapping bug passes) and NOT HF-vs-mcore logits",
        "reference_path": str(reference),
        "candidate_path": str(candidate),
        "reference_provenance": ref_stamp,
        "candidate_provenance": cand_stamp,
        "candidate_manifest_digest": manifest_digest(cand_manifest),
        "tensors_compared": len(shared),
        "bytes_compared": sum(cand_manifest[key]["nbytes"] for key in shared),
        "mtp_tensors_compared": sum(1 for key in shared if key.startswith("mtp.")),
        "missing_in_candidate": missing,
        "unexpected_in_candidate": extra,
        "differing_tensor_count": len(differing),
        "differing_tensors": differing[:max_reported],
        "diagnosis": [
            describe_mismatch(reference, candidate, key, ref_manifest[key], cand_manifest[key])
            for key in differing[:max_reported]
        ],
    }
    report["verdict"] = "PASS" if not (missing or extra or differing) else "FAIL"
    return report


def main() -> None:
    """Compare two exports of one iteration and record the verdict next to the production export."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="export produced at the reference EP (e.g. EP2)")
    parser.add_argument("candidate", type=Path, help="production export to certify (the EP4 one)")
    parser.add_argument("--out", type=Path, default=None, help="report path (default: <candidate>/export_parity.json)")
    parser.add_argument("--max-reported", type=int, default=8, help="how many differing tensors to describe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # EP, source checkpoint and iteration come from each export's own provenance stamp, never from a flag.
    report = compare(args.reference, args.candidate, max_reported=args.max_reported)
    out = args.out or (args.candidate / "export_parity.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    if report["verdict"] != "PASS":
        logger.error(
            "[q35-export-parity] FAIL: %s differing, %s missing, %s unexpected -> %s",
            report["differing_tensor_count"],
            len(report["missing_in_candidate"]),
            len(report["unexpected_in_candidate"]),
            out,
        )
        for detail in report["diagnosis"]:
            logger.error("[q35-export-parity]   %s", json.dumps(detail))
        raise SystemExit(1)
    logger.info(
        "[q35-export-parity] PASS: %s tensors / %.1f GiB bit-identical between EP%s and EP%s (iter %s) -> %s",
        report["tensors_compared"],
        report["bytes_compared"] / (1 << 30),
        report["reference_provenance"]["expert_parallel_size"],
        report["candidate_provenance"]["expert_parallel_size"],
        report["candidate_provenance"].get("iteration"),
        out,
    )
    logger.info("[q35-export-parity] scope: EP reshard only. NOT compared against the SAVE, no logit check.")


if __name__ == "__main__":
    main()
