#!/usr/bin/env python3
"""Replay an OV2_GDN_KERNEL_DUMP capture on one GPU and bisect the NaN.

The dump holds the EXACT kernel arguments (q, k, v, g, beta, cu_seqlens) from the training
step that produced a non-finite GatedDeltaNet output, so the failure becomes reproducible off
the 32-GPU job. Variants run in one pass to localize the cause:

  as-is            the captured call, verbatim            -> must reproduce, else state/env matters
  fp32-qkv         q/k/v upcast to fp32                   -> bf16 range issue?
  g-clamped        g clamped to >= -30                     -> decay underflow / exp(g) regime?
  single-seg       one segment covering all tokens        -> varlen segmentation issue?
  pad-last-64      last segment extended to a %64 multiple -> the candidate ov2_step fix
  torch-reference  mcore's torch_chunk_gated_delta_rule    -> kernel bug vs math (needs cu=None)

Usage:
    python3 replay_gdn_kernel_dump.py ~/train_logs/nan_dumps_layer/gdn_kernel_rank26_layer0.pt
"""
import argparse
import sys

import torch


def _stats(name: str, t) -> str:
    if not torch.is_tensor(t):
        return f"{name}=n/a"
    f = t.detach().float()
    return (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"range=[{f.min().item():.3e},{f.max().item():.3e}] finite={bool(torch.isfinite(f).all())}")


def _report(tag: str, out, cu_list) -> int:
    core = out[0] if isinstance(out, tuple) else out
    bad = ~torch.isfinite(core.float())
    n = int(bad.sum())
    detail = ""
    if n and cu_list:
        pos = bad.reshape(bad.shape[0], bad.shape[1], -1).any(dim=-1)[0]
        idx = pos.nonzero().flatten()
        segs = []
        for si, (a, b) in enumerate(zip(cu_list[:-1], cu_list[1:])):
            nb = int(pos[a:b].sum())
            if nb:
                segs.append(f"seg{si}(len={b - a},tail%64={(b - a) % 64}) bad_pos={nb}")
        detail = (f" first={int(idx[0])} last={int(idx[-1])} | " + "; ".join(segs)) if idx.numel() else ""
    print(f"  {tag:<16} nan_elems={n}{detail}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--skip", nargs="*", default=[], help="variant names to skip")
    args = ap.parse_args()

    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    print(f"fla version={getattr(fla, '__version__', '?')} file={fla.__file__}")
    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    dev = "cuda"
    q, k, v = (d["q"].to(dev), d["k"].to(dev), d["v"].to(dev))
    g, beta = d["g"].to(dev), d["beta"].to(dev)
    cu = d["cu_seqlens"].to(dev) if torch.is_tensor(d["cu_seqlens"]) else None
    cu_list = cu.tolist() if cu is not None else None
    static_kw = {kk: vv for kk, vv in (d.get("kwargs") or {}).items()
                 if kk in ("use_qk_l2norm_in_kernel", "output_final_state", "scale")}
    print(f"dump: layer={d.get('layer')} rank={d.get('rank')} cu={cu_list} static_kwargs={static_kw}")
    for nm, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        print("  " + _stats(nm, t))
    if torch.is_tensor(d.get("out")):
        print("  " + _stats("captured_out", d["out"]))

    def call(**over):
        kw = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False)
        kw.update(static_kw)
        kw.update(g=g, beta=beta, cu_seqlens=cu)
        kw.update(over)
        qq = kw.pop("_q", q)
        kk_ = kw.pop("_k", k)
        vv = kw.pop("_v", v)
        return chunk_gated_delta_rule(qq, kk_, vv, **kw)

    total_bad = 0
    print("variants:")
    if "as-is" not in args.skip:
        total_bad += _report("as-is", call(), cu_list)
    if "fp32-qkv" not in args.skip:
        _report("fp32-qkv", call(_q=q.float(), _k=k.float(), _v=v.float()), cu_list)
    if "g-clamped" not in args.skip:
        _report("g-clamped", call(g=g.clamp_min(-30.0)), cu_list)
    if "single-seg" not in args.skip and cu_list:
        one = torch.tensor([0, cu_list[-1]], device=dev, dtype=cu.dtype)
        _report("single-seg", call(cu_seqlens=one), [0, cu_list[-1]])
    if "pad-last-64" not in args.skip and cu_list and (cu_list[-1] - cu_list[-2]) % 64:
        # Extend the buffer so the last segment is 64-aligned (zero-pad q/k/v/g/beta).
        add = (-(cu_list[-1] - cu_list[-2])) % 64
        pad = lambda t: torch.cat([t, t.new_zeros((t.shape[0], add) + tuple(t.shape[2:]))], dim=1)
        cu2 = torch.tensor(cu_list[:-1] + [cu_list[-1] + add], device=dev, dtype=cu.dtype)
        _report("pad-last-64", call(_q=pad(q), _k=pad(k), _v=pad(v), g=pad(g), beta=pad(beta), cu_seqlens=cu2),
                cu2.tolist())

    print("RESULT:", "NAN-REPRODUCED (as-is)" if total_bad else "as-is CLEAN — state/env dependent")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
