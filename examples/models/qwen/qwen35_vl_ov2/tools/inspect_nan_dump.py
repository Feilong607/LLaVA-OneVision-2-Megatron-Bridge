#!/usr/bin/env python3
"""Post-mortem for OV2_NAN_DUMP microbatch dumps (llava_ov2's segment-level NaN probe).

Answers the questions that decide the fix direction:
  * WHERE are the NaN loss positions — masked (loss_mask 0) or supervised?
  * WHAT sits there — image pads, text, labels in vocab range?
  * Is the finite loss around them already exploding (bf16 overflow chain) or calm?
  * Which THD segment / vision run do they fall in?

Usage (CPU, any box with torch):
    python3 inspect_nan_dump.py ~/train_logs/nan_dumps/nan_bin_rank8.pt [more.pt ...]
"""
import sys

import torch


IMG_PAD = 248056
VIS_START = 248053
VOCAB = 248320


def inspect(path: str) -> None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids = d["input_ids"][0]
    loss = d["loss"][0].float()
    lm = d["loss_mask"][0].float() if d.get("loss_mask") is not None else torch.ones_like(loss)
    lb = d["labels"][0] if d.get("labels") is not None else None
    cu = d.get("cu_seqlens_q")
    cu = cu.tolist() if torch.is_tensor(cu) else (cu or [0, len(ids)])

    bad = ~torch.isfinite(loss)
    idx = bad.nonzero().flatten()
    fin = loss[~bad]
    print(f"==== {path}")
    print(f"seq={len(ids)} segments={len(cu) - 1} nan={int(bad.sum())} "
          f"finite_loss max={float(fin.max()):.3f} mean={float(fin.mean()):.4f}")
    if lb is not None:
        print(f"labels: min={int(lb.min())} max={int(lb.max())} out_of_vocab={int((lb >= VOCAB).sum())}")
    print(f"ids: max={int(ids.max())} img_pad_total={int((ids == IMG_PAD).sum())}")

    if len(idx) == 0:
        print("no NaN in this dump")
        return

    # Classify every NaN position.
    at_pad = int((ids[idx] == IMG_PAD).sum())
    at_masked = int((lm[idx] == 0).sum())
    at_supervised = int((lm[idx] > 0).sum())
    print(f"NaN positions: {len(idx)} | on_img_pad={at_pad} masked={at_masked} supervised={at_supervised}")

    # Segment + vision-run attribution for each NaN position (summarized).
    seg_of = {}
    for si, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        n = int(((idx >= a) & (idx < b)).sum())
        if n:
            seg_ids = ids[a:b]
            seg_of[si] = (n, int((seg_ids == IMG_PAD).sum()), int((seg_ids == VIS_START).sum()), b - a)
    for si, (n, npad, nrun, ln) in seg_of.items():
        print(f"  seg#{si}: nan={n} len={ln} img_pad={npad} vis_runs={nrun}")

    # First NaN positions with a +-4 window of loss values and token context.
    print("first NaN positions (pos | tok | label | mask | loss window +-4):")
    for p in idx[:8].tolist():
        lo, hi = max(0, p - 4), min(len(ids), p + 5)
        win = [f"{v:.1f}" if torch.isfinite(torch.tensor(v)) else "NaN" for v in loss[lo:hi].tolist()]
        print(f"  {p} | tok={int(ids[p])} | label={int(lb[p]) if lb is not None else '?'} | "
              f"mask={float(lm[p]):.0f} | {win}")

    # Are the exploding-but-finite losses clustered near vision runs?
    hot = (loss > 30) & ~bad
    print(f"finite loss>30 count={int(hot.sum())} (on_img_pad={int((ids[hot.nonzero().flatten()] == IMG_PAD).sum())})")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: inspect_nan_dump.py <dump.pt> [more.pt ...]")
    for p in sys.argv[1:]:
        inspect(p)


if __name__ == "__main__":
    main()
