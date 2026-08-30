#!/usr/bin/env python3
"""Minimal single-GPU repro for the GDN chunk-tail NaN (qwen3.5 s1.5 smoke case).

Observed: forward loss NaN in a CONTIGUOUS TAIL of exactly ``len % 64`` positions of a THD
segment (3384 -> 56, 1563 -> 27) — the final partial chunk of fla's chunk_gated_delta_rule
(chunk_size 64, varlen via cu_seqlens). This script calls the kernel exactly the way mcore's
GatedDeltaNet does (q/k l2norm outside, ``use_qk_l2norm_in_kernel=False``, g from
-exp(A_log)*softplus, beta from sigmoid, fp32 g/beta, bf16 qkv) on random inputs and checks
the tail for NaN — on the CURRENT image's fla and, after a pylibs swap, on a newer fla.

Run on ONE GPU (export workspace, feilong-nemo image):
    python3 repro_gdn_tail.py                       # default dims from Qwen3.5-35B-A3B text config
    python3 repro_gdn_tail.py --seqlens 1563 3384 1536 4096
    python3 repro_gdn_tail.py --config /path/to/Qwen3.5-35B-A3B-text/config.json

Exit code 1 if any NaN is found (so it can gate an A/B).
"""
import argparse
import json
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/ftan0055/Qwen3.5-35B-A3B-text/config.json")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[1563, 3384, 1536, 2048, 9008])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="input magnitude multiplier")
    args = ap.parse_args()

    import fla  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.l2norm import l2norm

    print(f"fla version={getattr(fla, '__version__', '?')} file={fla.__file__}")

    hk = hv = dk = dv = None
    try:
        cfg = json.load(open(args.config))
        hk = cfg.get("linear_num_key_heads")
        hv = cfg.get("linear_num_value_heads")
        dk = cfg.get("linear_key_head_dim")
        dv = cfg.get("linear_value_head_dim")
        print(f"config: key_heads={hk} value_heads={hv} key_dim={dk} value_dim={dv}")
    except Exception as e:  # noqa: BLE001
        print(f"config unreadable ({e}); using fallback dims")
    hk, hv, dk, dv = hk or 16, hv or 32, dk or 128, dv or 128
    # mcore runs TP-split heads; emulate the TP2 shard.
    hk_l, hv_l = max(1, hk // 2), max(1, hv // 2)

    dev = "cuda"
    torch.manual_seed(0)
    any_nan = False
    for s in args.seqlens:
        for t in range(args.trials):
            # GQA-style: value heads > key heads -> repeat q/k like mcore does before the kernel.
            q = torch.randn(1, s, hk_l, dk, device=dev, dtype=torch.bfloat16) * args.scale
            k = torch.randn(1, s, hk_l, dk, device=dev, dtype=torch.bfloat16) * args.scale
            rep = hv_l // hk_l if hv_l % hk_l == 0 else 1
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            v = torch.randn(1, s, hv_l, dv, device=dev, dtype=torch.bfloat16) * args.scale
            q, k = l2norm(q.contiguous()), l2norm(k.contiguous())
            # g = -exp(A_log) * softplus(alpha + dt_bias), fp32; beta = sigmoid, fp32.
            a_log = torch.randn(hv_l, device=dev).abs() * 0.5
            alpha = torch.randn(1, s, hv_l, device=dev, dtype=torch.bfloat16)
            g = (-a_log.exp() * torch.nn.functional.softplus(alpha.float() + 0.1))
            beta = torch.randn(1, s, hv_l, device=dev, dtype=torch.bfloat16).sigmoid().float()
            cu = torch.tensor([0, s], device=dev, dtype=torch.int32)
            out, _ = chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, cu_seqlens=cu,
            )
            bad = ~torch.isfinite(out.float())
            n = int(bad.sum())
            tail = s - (s % 64)
            tail_bad = int(bad[0, tail:].sum()) if s % 64 else 0
            flag = " <== NaN" if n else ""
            print(f"seq={s} trial={t} nan_elems={n} tail_region=({tail},{s}) tail_nan={tail_bad}{flag}")
            any_nan = any_nan or n > 0
    print("RESULT:", "NAN-REPRODUCED" if any_nan else "CLEAN")
    sys.exit(1 if any_nan else 0)


if __name__ == "__main__":
    main()
