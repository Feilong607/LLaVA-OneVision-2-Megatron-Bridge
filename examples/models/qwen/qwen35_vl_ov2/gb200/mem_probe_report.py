#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize ``[MEMPROBE]`` lines produced with ``OV2_MEM_PROBE_DEVICE=1`` (plus NCCL ALLOC log lines).

    python mem_probe_report.py ~/train_logs/smoke_qwen35_merged64k_<job>*.log [--min-click 0.5] [--all]

For every rank: first/last sample, totals, and every "click" (``d_dev`` >= --min-click GiB) with the owner
class that moved by the same amount (torch pool / in-process non-torch / unattributed), whether the click
coincided with a new batch-level max shape, a HybridEP/DeepEP buffer re-creation, a new process group, or
NCCL allocations logged (``NCCL_DEBUG_SUBSYS=INIT,ALLOC,NVLS``) by the same pid since the previous sample.
Ends with a verdict per owner class over all ranks. Reads any number of pod logs; stdin with ``-``.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

_GIB = 1024**3
_MP = re.compile(r"\[MEMPROBE r(\d+)\] fwd#(\d+) (.*)")
_KV = re.compile(r"(\S+?)=(\S+)")
# NCCL: "<host>:<pid>:<tid> [<dev>] NCCL INFO ... Alloc Size <bytes> ..." (ncclCudaMalloc / CUMEM paths)
_NCCL = re.compile(r"^\S+?:(\d+):\d+ \[\d+\] NCCL INFO (.*)$")
_ALLOC_SZ = re.compile(r"Alloc Size (\d+)")
_INIT_DONE = re.compile(r"Init COMPLETE|comm 0x[0-9a-f]+ rank \d+ nranks \d+.*Init COMPLETE")


def _g(v: str) -> float | None:
    """'+1.50G' / '12.00G' -> GiB float; '?' -> None."""
    if v is None or v == "?":
        return None
    v = v.rstrip("G")
    try:
        return float(v)
    except ValueError:
        return None


def _mib(v: str) -> float | None:
    if v is None or v == "?" or not v.endswith("MiB"):
        return None
    try:
        return int(v[:-3]) / 1024.0
    except ValueError:
        return None


def parse(lines):
    """Yield per-rank sample dicts in log order; attach NCCL alloc bytes/comm inits since the previous sample."""
    pending_alloc: dict[str, int] = defaultdict(int)
    pending_alloc_n: dict[str, int] = defaultdict(int)
    pending_init: dict[str, int] = defaultdict(int)
    for raw in lines:
        m = _NCCL.search(raw)
        if m:
            pid, rest = m.group(1), m.group(2)
            a = _ALLOC_SZ.search(rest)
            if a:
                pending_alloc[pid] += int(a.group(1))
                pending_alloc_n[pid] += 1
            if _INIT_DONE.search(rest):
                pending_init[pid] += 1
            continue
        m = _MP.search(raw)
        if not m:
            continue
        rank, fwd, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        kv = dict(_KV.findall(rest))
        if "dev_used" not in kv:
            continue  # plain MEMPROBE line without the device account
        pid = kv.get("pid", "?")
        s = {
            "rank": rank, "fwd": fwd, "pid": pid, "t": kv.get("t", "?"),
            "tp": kv.get("tp", "?"), "etp": kv.get("etp", "?"), "ep": kv.get("ep", "?"), "dp": kv.get("dp", "?"),
            "dev": _mib(kv.get("dev_used")), "proc": _mib(kv.get("proc_used")), "pidmatch": kv.get("pidmatch", "?"),
            "res": _g(kv.get("res")), "alloc": _g(kv.get("alloc")),
            "nontorch": _g(kv.get("nontorch_proc")), "unattr": _g(kv.get("unattr")),
            "d_dev": _g(kv.get("d_dev")), "d_proc": _g(kv.get("d_proc")), "d_res": _g(kv.get("d_res")),
            "d_nontorch": _g(kv.get("d_nontorch")), "d_unattr": _g(kv.get("d_unattr")),
            "d_inactive": _g(kv.get("d_inactive")), "inactive": _g(kv.get("inactive_split")),
            "segs": kv.get("segs", "?"), "retries": kv.get("retries", "?"), "ooms": kv.get("ooms", "?"),
            "pgs": kv.get("pgs", "?"), "hep": kv.get("hep", "?"), "dep": kv.get("dep", "?"),
            "dyn": kv.get("dyn", "?"), "triton": kv.get("triton", "?"),
            "tokens": kv.get("tokens", "?"), "patches": kv.get("patches", "?"), "nimg": kv.get("nimg", "?"),
            "smax": kv.get("smax", "?"), "pmax": kv.get("pmax", "?"),
            "nccl_alloc": pending_alloc.pop(pid, 0), "nccl_alloc_n": pending_alloc_n.pop(pid, 0),
            "nccl_init": pending_init.pop(pid, 0),
        }
        yield s


def _owner(s: dict) -> tuple[str, float]:
    """Owner class whose delta best matches d_dev, and its share of d_dev."""
    d = s["d_dev"] or 0.0
    cands = {"torch_pool": s["d_res"], "nontorch_proc": s["d_nontorch"], "unattr": s["d_unattr"]}
    cands = {k: v for k, v in cands.items() if v is not None}
    if not cands or d <= 0:
        return "?", 0.0
    k = max(cands, key=lambda n: cands[n])
    return k, (cands[k] / d if d else 0.0)


def _new_max(s: dict) -> str:
    flags = []
    try:
        if s["tokens"] != "?" and int(s["tokens"]) >= int(s["smax"]):
            flags.append("tokens=smax")
        if s["patches"] != "?" and int(s["patches"]) >= int(s["pmax"]):
            flags.append("patches=pmax")
    except ValueError:
        pass
    return ",".join(flags) if flags else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="pod log files ('-' = stdin)")
    ap.add_argument("--min-click", type=float, default=0.5, help="GiB of d_dev that counts as a click (default 0.5)")
    ap.add_argument("--all", action="store_true", help="print every sample, not only clicks")
    args = ap.parse_args(argv)

    samples = []
    for path in args.logs:
        fh = sys.stdin if path == "-" else open(path, "r", errors="replace", encoding="utf-8")
        with fh:
            samples.extend(parse(fh))
    if not samples:
        print("no [MEMPROBE] lines with dev_used= found (is OV2_MEM_PROBE_DEVICE=1 and OV2_MEM_PROBE>0 set?)")
        return 2

    by_rank: dict[int, list] = defaultdict(list)
    for s in samples:
        by_rank[s["rank"]].append(s)

    tot_owner: dict[str, float] = defaultdict(float)
    tot_clicks = 0
    prev_state: dict[int, dict] = {}
    for rank in sorted(by_rank):
        ss = by_rank[rank]
        f, l = ss[0], ss[-1]
        ddev = (l["dev"] - f["dev"]) if (l["dev"] is not None and f["dev"] is not None) else None
        dproc = (l["proc"] - f["proc"]) if (l["proc"] is not None and f["proc"] is not None) else None
        dres = (l["res"] - f["res"]) if (l["res"] is not None and f["res"] is not None) else None
        dun = (l["unattr"] - f["unattr"]) if (l["unattr"] is not None and f["unattr"] is not None) else None
        dnt = (l["nontorch"] - f["nontorch"]) if (l["nontorch"] is not None and f["nontorch"] is not None) else None

        def fmt(x):
            return "?" if x is None else f"{x:+.2f}G"

        print(f"\n== rank {rank} pid {f['pid']} tp={f['tp']} etp={f['etp']} ep={f['ep']} dp={f['dp']} "
              f"pidmatch={l['pidmatch']} samples={len(ss)} fwd#{f['fwd']}..{l['fwd']} {f['t']}..{l['t']}")
        print(f"   dev_used {fmt(f['dev'])[1:] if f['dev'] is not None else '?'} -> "
              f"{fmt(l['dev'])[1:] if l['dev'] is not None else '?'}  total d_dev={fmt(ddev)}  "
              f"d_proc={fmt(dproc)} d_res={fmt(dres)} d_nontorch={fmt(dnt)} d_unattr={fmt(dun)}")
        print(f"   torch: alloc {f['alloc']}->{l['alloc']}G res {f['res']}->{l['res']}G "
              f"inactive_split {f['inactive']}->{l['inactive']}G segs {f['segs']}->{l['segs']} "
              f"retries {f['retries']}->{l['retries']} ooms {f['ooms']}->{l['ooms']}")
        print(f"   libs: pgs {f['pgs']}->{l['pgs']} hep {f['hep']}->{l['hep']} dep {f['dep']}->{l['dep']} "
              f"dyn {f['dyn']}->{l['dyn']} triton {f['triton']}->{l['triton']} "
              f"nccl_alloc_total={sum(s['nccl_alloc'] for s in ss) / _GIB:.2f}G "
              f"nccl_inits={sum(s['nccl_init'] for s in ss)} smax={l['smax']} pmax={l['pmax']}")
        for s in ss:
            lib = []
            ps = prev_state.get(rank, {})
            if s["hep"] != "?" and ps.get("hep") not in (None, s["hep"]):
                lib.append("HEP-REALLOC")
            if s["dep"] != "?" and ps.get("dep") not in (None, s["dep"]):
                lib.append("DEP-REALLOC")
            if s["pgs"] != "?" and ps.get("pgs") not in (None, s["pgs"]):
                lib.append("NEW-PG")
            prev_state[rank] = {"hep": s["hep"], "dep": s["dep"], "pgs": s["pgs"]}
            if s["nccl_alloc"]:
                lib.append(f"nccl_alloc={s['nccl_alloc'] / _GIB:.2f}G/{s['nccl_alloc_n']}")
            if s["nccl_init"]:
                lib.append(f"nccl_init={s['nccl_init']}")
            click = s["d_dev"] is not None and s["d_dev"] >= args.min_click
            if not (click or args.all):
                continue
            owner, share = _owner(s)
            if click:
                tot_clicks += 1
                tot_owner[owner] += s["d_dev"]
            tag = "CLICK" if click else "     "
            print(f"   {tag} fwd#{s['fwd']:<5} {s['t']} d_dev={fmt(s['d_dev'])} -> {owner} ({share:.0%}) "
                  f"[d_res={fmt(s['d_res'])} d_nontorch={fmt(s['d_nontorch'])} d_unattr={fmt(s['d_unattr'])} "
                  f"d_inactive={fmt(s['d_inactive'])}] shape tokens={s['tokens']} patches={s['patches']} "
                  f"nimg={s['nimg']} newmax={_new_max(s)} libs={','.join(lib) if lib else '-'}")

    print("\n== verdict over all ranks")
    if not tot_clicks:
        print(f"   no click >= {args.min_click} GiB in {len(samples)} samples: nothing ratchets at this resolution")
        return 0
    tot = sum(tot_owner.values()) or 1.0
    for k in sorted(tot_owner, key=lambda n: -tot_owner[n]):
        print(f"   {k:<14} {tot_owner[k]:8.2f} GiB  {tot_owner[k] / tot:6.1%} of clicked growth")
    print("   torch_pool -> caching-allocator high-water (expandable_segments); nontorch_proc -> a library's"
          " cudaMalloc in-process; unattr -> cuMem/NVLS/HybridEP (see nccl_alloc/HEP-REALLOC on the same line)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
