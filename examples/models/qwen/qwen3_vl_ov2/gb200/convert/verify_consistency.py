#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify weight CONSISTENCY between two OV2 torch_dist checkpoints (before/after conversion).

Single process, CPU only (no GPU, no torchrun, no model-parallel init). A torch_dist checkpoint
stores GLOBAL (unsharded) tensors, so this correctly compares checkpoints saved at DIFFERENT
parallelism (e.g. EP8 source vs EP4 reshard, or 16-GPU vs 8-GPU). That is exactly the
"转化前后一致性" guarantee: a lossless conversion / reshard leaves every MODEL tensor bit-identical.

What it compares (read from each checkpoint's DCP `.metadata`, which is parallelism-independent):
  1. key set      -- which tensors exist (per category)
  2. shape+dtype  -- per common tensor
  3. values       -- per tensor, loaded with mcore dist_checkpointing (reshards on load) in
                     bounded-memory chunks along dim 0, so the 38 GB stacked expert tensors are
                     compared without ever materializing the whole thing. Exact by default;
                     --atol/--rtol allow tolerance.

Key categories (reported separately):
  model       language_model.* / vision_model.* / adapter.*  -> MUST match (drives exit code)
  optimizer   optimizer.*                                     -> off by default (conversion drops optim)
  extra_state *._extra_state (TE byte blobs)                  -> presence only, never value-compared

Usage (CPU container -- see verify.sh):
  python verify_consistency.py --a <ckptA> --b <ckptB> [--values sample|full|none] [--atol 0]
Exit 0 = model weights consistent; 1 = mismatch; 2 = usage / IO error.
"""
import argparse
import os
import re
import sys


# Representative model tensors for --values sample (one per structural family). Deliberately
# INCLUDES experts + router (the risky bits when EP changes) -- chunked loading keeps them cheap.
_SAMPLE_PATTERNS = [
    r"embedding",
    r"output_layer",
    r"final_layernorm",
    r"experts.*linear_fc1\.weight",
    r"experts.*linear_fc2\.weight",
    r"mlp\.router\.weight",
    r"self_attention\.linear_qkv\.weight",
    r"self_attention\.linear_proj\.weight",
    # Qwen3 QK-LayerNorm + fused input/pre-MLP layernorms (trained, stacked per-layer) -- WITHOUT these
    # 'sample' silently skipped 4 LLM model-tensor families, so a layernorm-localized reshard bug could
    # slip past as CONSISTENT. (VALUES=full is still the exhaustive gate.)
    r"q_layernorm",
    r"k_layernorm",
    r"linear_qkv\.layer_norm_weight",
    r"pre_mlp_layernorm",
    r"\badapter\b",
    r"vision",
]


def _resolve_iter_dir(path):
    """Accept a Bridge ckpt root (latest_checkpointed_iteration.txt) or a direct dist-ckpt dir."""
    if os.path.isfile(os.path.join(path, ".metadata")):
        return path
    tag = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tag):
        with open(tag) as f:
            it = f.read().strip()
        cand = os.path.join(path, "iter_{:07d}".format(int(it)))
        if os.path.isfile(os.path.join(cand, ".metadata")):
            return cand
    if os.path.isdir(path):
        iters = sorted(d for d in os.listdir(path) if d.startswith("iter_"))
        for d in reversed(iters):
            if os.path.isfile(os.path.join(path, d, ".metadata")):
                return os.path.join(path, d)
    return path


def _category(key):
    if "_extra_state" in key:
        return "extra_state"
    if key.startswith("optimizer"):
        return "optimizer"
    # RNG / bookkeeping state -- NOT a model weight. save_megatron_model drops it (save_rng=False),
    # so it is legitimately absent from a converted ckpt; never let its absence fail the check.
    if key.split("/")[0] in ("rng_state", "rerun_state_machine", "rerun_state") or key in (
        "iteration", "checkpoint_version", "num_floating_point_operations_so_far"
    ):
        return "rng"
    return "model"


def _dtype_bytes(dtype):
    try:
        import torch
        return torch.empty(0, dtype=dtype).element_size()
    except Exception:
        return 2


def _pick_chunks(global0, per_row_bytes, budget_bytes):
    """Largest #rows along dim0 that fits budget AND evenly divides global0 -> (n_chunks, rows_per_chunk)."""
    if global0 <= 1 or per_row_bytes * global0 <= budget_bytes:
        return 1, global0
    max_rows = max(1, budget_bytes // max(1, per_row_bytes))
    # smallest n_chunks (=> largest chunk) that divides global0 and keeps chunk <= budget
    for n in range(1, global0 + 1):
        if global0 % n == 0 and (global0 // n) <= max_rows:
            return n, global0 // n
    return global0, 1  # fall back to one row per chunk


def _compare(ta, tb, atol, rtol, exact):
    """Return (match, max_abs_diff). NaN-aware: both-NaN positions count as EQUAL (so a
    bit-identical pair with NaNs is not a false mismatch); a lone NaN counts as an infinite diff."""
    import torch
    flt = ta.is_floating_point()
    if exact:
        eq = torch.eq(ta, tb)
        if flt:
            eq = eq | (torch.isnan(ta) & torch.isnan(tb))
        if bool(eq.all()):
            return True, 0.0
        match = False
    a, b = ta.float(), tb.float()
    if not exact:
        match = bool(torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True).all())
    diff = (a - b).abs()
    if flt:
        bn = torch.isnan(a) & torch.isnan(b)
        on = torch.isnan(a) ^ torch.isnan(b)
        diff = torch.where(bn, torch.zeros_like(diff), diff)
        diff = torch.where(on, torch.full_like(diff, float("inf")), diff)
    return match, (float(diff.max()) if diff.numel() else 0.0)


def main():
    ap = argparse.ArgumentParser(description="OV2 torch_dist before/after consistency verifier (CPU).")
    ap.add_argument("--a", required=True, help="checkpoint A (source / before)")
    ap.add_argument("--b", required=True, help="checkpoint B (converted / after)")
    ap.add_argument("--values", choices=["none", "sample", "full"], default="sample",
                    help="value comparison depth (default: sample = one tensor per structural family)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (default 0 = bit-exact)")
    ap.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (default 0 = bit-exact)")
    ap.add_argument("--include-optim", action="store_true",
                    help="also value-compare optimizer.* state (off by default; conversion drops optim)")
    ap.add_argument("--chunk-gb", type=float, default=4.0,
                    help="per-tensor-per-side load budget in GiB for dim0 chunking (default 4)")
    ap.add_argument("--max-report", type=int, default=25, help="max mismatching keys to list")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata, BytesStorageMetadata  # noqa: F401

    A, B = _resolve_iter_dir(args.a), _resolve_iter_dir(args.b)
    for tag, p in (("A", A), ("B", B)):
        if not os.path.isfile(os.path.join(p, ".metadata")):
            print("ERROR: {} has no .metadata at {} (not a torch_dist checkpoint dir)".format(tag, p),
                  file=sys.stderr)
            return 2
    print("[verify] A (before) = {}".format(A))
    print("[verify] B (after)  = {}".format(B))

    ma = FileSystemReader(A).read_metadata().state_dict_metadata
    mb = FileSystemReader(B).read_metadata().state_dict_metadata
    ka, kb = set(ma), set(mb)
    common = ka & kb

    def counts(ks):
        return {c: sum(1 for k in ks if _category(k) == c) for c in ("model", "optimizer", "rng", "extra_state")}

    print("[verify] keys: A={} B={} common={}".format(len(ka), len(kb), len(common)))
    print("[verify]   A {} | B {}".format(counts(ka), counts(kb)))

    fail = False

    # ---- 1. key-set: missing MODEL tensors are a real failure; optim/extra are informational ----
    model_only_a = sorted(k for k in (ka - kb) if _category(k) == "model")
    model_only_b = sorted(k for k in (kb - ka) if _category(k) == "model")
    if model_only_a or model_only_b:
        fail = True
        print("[verify] !! MODEL key-set mismatch: only-in-A={} only-in-B={}".format(
            len(model_only_a), len(model_only_b)))
        for k in model_only_a[:args.max_report]:
            print("            only A: {}".format(k))
        for k in model_only_b[:args.max_report]:
            print("            only B: {}".format(k))
    else:
        print("[verify] model key-set identical ({} model tensors present on both sides)".format(
            counts(common)["model"]))
    nonmodel_diff = sorted(k for k in (ka ^ kb) if _category(k) != "model")
    if nonmodel_diff:
        print("[verify]   (info) {} non-model keys differ in presence (optimizer/extra_state) -- "
              "expected when conversion drops optimizer state".format(len(nonmodel_diff)))

    # ---- 2. shape + dtype on common tensor keys ----
    # require BOTH sides be a real tensor; a storage-type flip (tensor<->bytes) is itself a mismatch.
    tensor_common = [k for k in common
                     if isinstance(ma[k], TensorStorageMetadata) and isinstance(mb[k], TensorStorageMetadata)]
    storage_flip = sorted(k for k in common
                          if isinstance(ma[k], TensorStorageMetadata) != isinstance(mb[k], TensorStorageMetadata))
    if storage_flip:
        if any(_category(k) == "model" for k in storage_flip):
            fail = True
        print("[verify] !! storage-type differs (tensor vs bytes) on {} common keys:".format(len(storage_flip)))
        for k in storage_flip[:args.max_report]:
            print("            {}: A={} B={}".format(k, type(ma[k]).__name__, type(mb[k]).__name__))
    # SHAPE (size) mismatch is a real structural problem. A dtype-only diff (same size) is NOT failed
    # here -- it is most often a lossless re-cast (e.g. bf16->fp32 when the rebuilt model holds a param
    # at higher precision); we value-compare it in fp32 below and only fail if the VALUES differ.
    shape_bad, dtype_diff = [], []
    for k in tensor_common:
        a, b = ma[k], mb[k]
        if tuple(a.size) != tuple(b.size):
            shape_bad.append((k, tuple(a.size), a.properties.dtype, tuple(b.size), b.properties.dtype))
        elif a.properties.dtype != b.properties.dtype:
            dtype_diff.append((k, a.properties.dtype, b.properties.dtype))
    if shape_bad:
        if any(_category(k) == "model" for k, *_ in shape_bad):
            fail = True
        print("[verify] !! SHAPE mismatch on {} common tensors:".format(len(shape_bad)))
        for k, sa, da, sb, db in shape_bad[:args.max_report]:
            print("            {}: A{} vs B{}".format(k, sa, sb))
    if dtype_diff:
        print("[verify] dtype differs on {} common tensors (value-checked in fp32 below; benign if equal):"
              .format(len(dtype_diff)))
        for k, da, db in dtype_diff[:args.max_report]:
            print("            {}: A={} B={}".format(k, da, db))
    if not shape_bad:
        print("[verify] shape identical on all {} common tensors".format(len(tensor_common)))

    if args.values == "none":
        return _finish(fail)

    # ---- 3. value comparison (chunked along dim0, reshard-aware via mcore) ----
    bad_keys = {k for k, *_ in shape_bad}
    cands = [k for k in tensor_common if k not in bad_keys]
    if not args.include_optim:
        cands = [k for k in cands if _category(k) == "model"]
    if args.values == "sample":
        pats = [re.compile(p) for p in _SAMPLE_PATTERNS]
        cands = [k for k in cands if any(p.search(k) for p in pats)]
    # ALWAYS value-check dtype-differing tensors (their dtype-only diff is benign ONLY if values match).
    dtype_diff_keys = {k for k, *_ in dtype_diff if k not in bad_keys
                       and (args.include_optim or _category(k) == "model")}
    cands = sorted(set(cands) | dtype_diff_keys)
    if not cands:
        print("[verify] (no tensors selected for value comparison)")
        return _finish(fail)
    print("[verify] value-comparing {} tensors (mode={}, atol={}, rtol={}, chunk<= {} GiB) ...".format(
        len(cands), args.values, args.atol, args.rtol, args.chunk_gb))
    if args.values == "sample":
        print("[verify] NOTE: 'sample' compares representative tensor families only -- a CONSISTENT here is "
              "not exhaustive; run VALUES=full for the real before/after gate.")

    from megatron.core import dist_checkpointing as mdc
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    if not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)

    budget = int(args.chunk_gb * (1024 ** 3))
    exact = (args.atol == 0.0 and args.rtol == 0.0)

    def _load_chunk(ckpt_dir, key, meta, n_chunks, chunk_idx):
        shape = list(meta.size)
        if not shape:  # 0-d scalar -> fully replicated, no axis fragmentation
            buf = torch.empty([], dtype=meta.properties.dtype)
            st = ShardedTensor.from_rank_offsets(key, buf, replica_id=0)
        else:
            local = list(shape)
            local[0] = shape[0] // n_chunks
            buf = torch.empty(tuple(local), dtype=meta.properties.dtype)
            st = ShardedTensor.from_rank_offsets(key, buf, (0, chunk_idx, n_chunks), replica_id=0)
        mdc.load({key: st}, ckpt_dir, validate_access_integrity=False)
        return st.data

    n_ok = n_bad = 0
    worst = []  # (key, max_abs_diff)
    for i, k in enumerate(cands):
        meta = ma[k]
        g0 = int(meta.size[0]) if len(meta.size) > 0 else 1
        per_row = _dtype_bytes(meta.properties.dtype)
        for d in meta.size[1:]:
            per_row *= int(d)
        n_chunks, _ = _pick_chunks(g0, per_row, budget)
        key_match = True
        key_maxdiff = 0.0
        for c in range(n_chunks):
            ta = _load_chunk(A, k, ma[k], n_chunks, c)
            tb = _load_chunk(B, k, mb[k], n_chunks, c)
            m, d = _compare(ta, tb, args.atol, args.rtol, exact)
            key_match = key_match and m
            key_maxdiff = max(key_maxdiff, d)
            del ta, tb
        if key_match:
            n_ok += 1
        else:
            n_bad += 1
            worst.append((k, key_maxdiff))
            if _category(k) == "model":
                fail = True
        if (i + 1) % 25 == 0 or (i + 1) == len(cands):
            print("[verify]   ... {}/{} compared ({} mismatched)".format(i + 1, len(cands), n_bad))

    worst.sort(key=lambda x: -x[1])
    print("[verify] values: {} identical, {} mismatched".format(n_ok, n_bad))
    for k, d in worst[:args.max_report]:
        print("            DIFF max|Δ|={:.3e}  {}".format(d, k))
    return _finish(fail)


def _finish(fail):
    print()
    if fail:
        print("[verify] RESULT: INCONSISTENT  (model weights differ -- see above)")
        return 1
    print("[verify] RESULT: CONSISTENT  (all compared model weights identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
