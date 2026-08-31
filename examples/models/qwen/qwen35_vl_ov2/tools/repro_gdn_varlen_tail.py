#!/usr/bin/env python3
"""Decisive single-GPU test for the GDN last-segment tail NaN (qwen3.5 s1.5).

Evidence this script tests (from two OV2_NAN_DUMP microbatches, fla 0.4.2 AND 0.5.0):
  * NaN count == (last PADDED segment length) % 64, exactly, twice
      rank8:  cu_padded [0, 5631, 9016] -> last 3385, 3385 % 64 = 57 == NaN count 57
      rank26: cu_padded [0, ?, 7476, 9040] -> last 1564, 1564 % 64 = 28 == NaN count 28
  * INTERIOR segments' partial tails are finite (seg0 tail of 63 positions is clean)
  * a single-segment call with the same lengths is CLEAN (repro_gdn_tail.py)
=> hypothesis: the final partial chunk of the LAST segment is never written by the
   kernel, so the output there is whatever the caching allocator handed over. Fresh
   allocations in a small test are zero-filled (looks clean); a training step reuses
   dirty blocks (NaN/inf garbage).

Two knobs make the hypothesis falsifiable:
  --poison        dirty the caching allocator with NaN before the kernel call, so an
                  UNWRITTEN output region shows up as NaN instead of luckily-zero.
  --pad-last-64   extend the buffer so the last padded segment is a multiple of 64
                  (the candidate fix, applied on OUR side in ov2_step) and re-test.

Usage (one GPU):
    python3 repro_gdn_varlen_tail.py                          # built-in dump-derived cu
    python3 repro_gdn_varlen_tail.py --poison
    python3 repro_gdn_varlen_tail.py --poison --pad-last-64
    python3 repro_gdn_varlen_tail.py --poison --dump ~/train_logs/nan_dumps_fla050/nan_bin_rank8.pt

Exit 1 if any NaN is found (so it can gate an A/B).
"""
import argparse
import sys

import torch


# cu_seqlens_PADDED reconstructed per ov2_step's partial-pack rule (cu_padded = [cu[:-1], seq_len]);
# the dumps store cu_seqlens_q (unpadded), so a dump is converted the same way below.
DEFAULT_CASES = [
    ("rank8-like", [0, 5631, 9016]),
    ("rank26-like", [0, 3200, 7476, 9040]),
    ("single-seg-control", [0, 9016]),
]


def poison_allocator(dev: str, mib: int = 512) -> None:
    """Fill and free a chunk of NaN so later torch.empty() blocks come back dirty."""
    blocks = [torch.full((mib * 1024 * 1024 // 8 // 4,), float("nan"), device=dev) for _ in range(4)]
    for b in blocks:
        b.mul_(1.0)
    del blocks
    # No empty_cache(): keeping the blocks in the caching allocator is the point.


def pad_last_to_64(cu: list[int]) -> list[int]:
    """Extend the final segment so its length is a multiple of 64 (candidate fix)."""
    cu = list(cu)
    last_len = cu[-1] - cu[-2]
    add = (-last_len) % 64
    cu[-1] += add
    return cu


def run_case(name: str, cu_list: list[int], args, dims) -> int:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    hk_l, hv_l, dk, dv = dims
    dev = "cuda"
    T = cu_list[-1]
    seglens = [b - a for a, b in zip(cu_list[:-1], cu_list[1:])]
    tails = [ln % 64 for ln in seglens]
    print(f"-- {name}: cu={cu_list} seglens={seglens} tails(%64)={tails}")

    nan_total = 0
    for t in range(args.trials):
        if args.poison:
            poison_allocator(dev)
        q = torch.randn(1, T, hk_l, dk, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, T, hk_l, dk, device=dev, dtype=torch.bfloat16)
        rep = hv_l // hk_l if hk_l and hv_l % hk_l == 0 else 1
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
        v = torch.randn(1, T, hv_l, dv, device=dev, dtype=torch.bfloat16)
        q, k = l2norm(q.contiguous()), l2norm(k.contiguous())
        a_log = torch.randn(hv_l, device=dev).abs() * 0.5
        alpha = torch.randn(1, T, hv_l, device=dev, dtype=torch.bfloat16)
        g = -a_log.exp() * torch.nn.functional.softplus(alpha.float() + 0.1)
        beta = torch.randn(1, T, hv_l, device=dev, dtype=torch.bfloat16).sigmoid().float()
        cu = torch.tensor(cu_list, device=dev, dtype=torch.int32)

        out, _ = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, cu_seqlens=cu,
        )
        bad = ~torch.isfinite(out.float())
        n = int(bad.sum())
        nan_total += n
        per_seg = []
        for si, (a, b) in enumerate(zip(cu_list[:-1], cu_list[1:])):
            nb = int(bad[0, a:b].sum())
            if nb:
                tail_start = b - (b - a) % 64
                in_tail = int(bad[0, tail_start:b].sum()) if (b - a) % 64 else 0
                per_seg.append(f"seg{si}(len={b - a}) nan={nb} in_tail={in_tail}")
        print(f"   trial={t} nan_elems={n}" + ("  | " + "; ".join(per_seg) if per_seg else ""))
    return nan_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="OV2_NAN_DUMP .pt to take cu_seqlens_q + seq length from")
    ap.add_argument("--poison", action="store_true", help="dirty the allocator with NaN first")
    ap.add_argument("--pad-last-64", action="store_true", help="test the candidate fix")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--heads", type=int, default=16, help="key heads (TP-local: halved for TP2)")
    ap.add_argument("--value-heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    import fla

    print(f"fla version={getattr(fla, '__version__', '?')} file={fla.__file__}")
    print(f"poison={args.poison} pad_last_64={args.pad_last_64} trials={args.trials}")

    cases = list(DEFAULT_CASES)
    if args.dump:
        d = torch.load(args.dump, map_location="cpu", weights_only=False)
        cu_unpadded = d["cu_seqlens_q"].tolist()
        seq = int(d["input_ids"].shape[1])
        cu_padded = cu_unpadded[:-1] + [seq]  # ov2_step's partial-pack fold
        cases = [(f"dump:{args.dump.split('/')[-1]}", cu_padded)]
        print(f"   dump cu_unpadded={cu_unpadded} seq={seq} -> cu_padded={cu_padded}")

    # TP2 shard of the Qwen3.5-35B-A3B linear-attention heads.
    dims = (max(1, args.heads // 2), max(1, args.value_heads // 2), args.dim, args.dim)
    total = 0
    for name, cu_list in cases:
        if args.pad_last_64:
            cu_list = pad_last_to_64(cu_list)
            name += "+pad64"
        total += run_case(name, cu_list, args, dims)

    print("RESULT:", "NAN-REPRODUCED" if total else "CLEAN")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
