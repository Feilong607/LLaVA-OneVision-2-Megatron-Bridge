#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Account the cumem_hook ledgers: who (which library) made which cuMem*/cuMulticast* calls, how many bytes
stay outstanding, and which calls fall inside each MEMPROBE unattributed "click" window.

    python cumem_ledger_report.py ~/train_logs/cumem/cumem_<host>_*.log [--memprobe train_node0.log] [--events]

Per ledger (= per rank process): counts and bytes per API, net outstanding bytes (cuMemCreate - cuMemRelease of
known handles), multicast bind/unbind bytes, imports, and the caller library of each call (first frame-pointer-walk frame
outside libcumemhook / libcupti / libcuda, resolved through the cumem_<host>_<pid>.maps dump written next to
the ledger; falls back to the calling thread's name ``thr:<comm>`` when the walk yields nothing usable). With --memprobe, MEMPROBE lines of the same pid are read and every click (d_unattr >= --min-click
GiB) is printed with the ledger events that happened between the previous sample and this one -- a bind or
create of matching size in that window, plus its caller library, closes the case.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import os
import re
import sys
from collections import defaultdict

_GIB = 1024**3
_KV = re.compile(r"(\S+?)=(\S+)")
_MP = re.compile(r"\[MEMPROBE r(\d+)\] fwd#(\d+) (.*)")
_SKIP_LIBS = ("libcumemhook", "libcupti", "libcuda.so", "[vdso]", "ld-linux", "libc.so", "libpthread")


def load_maps(path: str):
    """[(start, end, offset, name)] sorted by start."""
    out = []
    try:
        with open(path, errors="replace") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) < 6:
                    continue
                a, b = parts[0].split("-")
                out.append((int(a, 16), int(b, 16), int(parts[2], 16), parts[5]))
    except OSError:
        return []
    out.sort()
    return out


def symbolize(addr: int, maps) -> str:
    if not maps:
        return f"?+0x{addr:x}"
    starts = [m[0] for m in maps]
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0 and maps[i][0] <= addr < maps[i][1]:
        s, _e, off, name = maps[i]
        return f"{os.path.basename(name)}+0x{addr - s + off:x}"
    return f"?+0x{addr:x}"


def caller_lib(frames: list[str]) -> str:
    for f in frames:
        lib = f.split("+", 1)[0]
        if lib != "?" and not any(lib.startswith(s) for s in _SKIP_LIBS):
            return lib
    return frames[-1].split("+", 1)[0] if frames else "?"


def parse_ledger(path: str):
    maps = load_maps(re.sub(r"\.log$", ".maps", path))
    events = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            if ln.startswith("#") or "api=" not in ln:
                continue
            kv = dict(_KV.findall(ln))
            bt = [int(x, 16) for x in kv.get("bt", "").split(",") if x.startswith("0x")]
            frames = [symbolize(a, maps) for a in bt]
            events.append({
                "t": float(kv.get("t", "0")), "lt": kv.get("lt", "?"), "pid": kv.get("pid", "?"),
                "api": kv.get("api", "?"), "ret": int(kv.get("ret", "-1")),
                "size": int(kv.get("size", "0")), "handle": kv.get("handle"), "mc": kv.get("mc"),
                "dev": kv.get("dev"), "thr": kv.get("thr", "?"), "frames": frames,
                "caller": caller_lib(frames) if frames else "?",
            })
            if events[-1]["caller"] == "?":
                events[-1]["caller"] = "thr:" + events[-1]["thr"]
    return events


def account(events):
    by_api = defaultdict(lambda: [0, 0])   # api -> [count, bytes]
    by_caller = defaultdict(lambda: [0, 0])  # (api, caller) -> [count, bytes]
    created: dict[str, int] = {}
    net_create = 0
    released_unknown = 0
    mc_bound = 0
    mc_unbound = 0
    for e in events:
        if e["ret"] != 0:
            continue
        by_api[e["api"]][0] += 1
        by_api[e["api"]][1] += e["size"]
        by_caller[(e["api"], e["caller"])][0] += 1
        by_caller[(e["api"], e["caller"])][1] += e["size"]
        if e["api"] == "cuMemCreate" and e["handle"]:
            created[e["handle"]] = e["size"]
            net_create += e["size"]
        elif e["api"] == "cuMemRelease" and e["handle"]:
            sz = created.pop(e["handle"], None)
            if sz is None:
                released_unknown += 1
            else:
                net_create -= sz
        elif e["api"] == "cuMulticastBindMem" or e["api"] == "cuMulticastBindAddr":
            mc_bound += e["size"]
        elif e["api"] == "cuMulticastUnbind":
            mc_unbound += e["size"]
    return by_api, by_caller, net_create, released_unknown, mc_bound, mc_unbound


def parse_memprobe(paths, pid: str):
    """[(HH:MM:SS, fwd, d_unattr GiB)] for the given pid, in log order."""
    out = []
    for p in paths:
        with open(p, errors="replace") as fh:
            for ln in fh:
                m = _MP.search(ln)
                if not m:
                    continue
                kv = dict(_KV.findall(m.group(3)))
                if kv.get("pid") != pid or "d_unattr" not in kv:
                    continue
                du = kv["d_unattr"]
                out.append((kv.get("t", "?"), int(m.group(2)), None if du == "?" else float(du.rstrip("G"))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledgers", nargs="+", help="cumem_<host>_<pid>.log files (globs ok)")
    ap.add_argument("--memprobe", nargs="*", default=[], help="train_node*.log with OV2_MEM_PROBE_DEVICE=1 lines")
    ap.add_argument("--min-click", type=float, default=0.5)
    ap.add_argument("--events", action="store_true", help="print every ledger event")
    args = ap.parse_args(argv)

    paths = []
    for g in args.ledgers:
        paths.extend(sorted(glob.glob(g)) or [g])
    grand = defaultdict(lambda: [0, 0])
    for path in paths:
        try:
            events = parse_ledger(path)
        except OSError as e:
            print(f"!! {path}: {e}")
            continue
        pid = events[0]["pid"] if events else re.search(r"_(\d+)\.log$", path).group(1) if re.search(r"_(\d+)\.log$", path) else "?"
        by_api, by_caller, net, rel_unknown, mcb, mcu = account(events)
        print(f"\n== {os.path.basename(path)} pid {pid}: {len(events)} events, "
              f"net cuMemCreate outstanding {net / _GIB:.2f} GiB, multicast bound {mcb / _GIB:.2f} GiB "
              f"(unbound {mcu / _GIB:.2f}), releases of foreign handles {rel_unknown}")
        for api in sorted(by_api, key=lambda a: -by_api[a][1]):
            c, b = by_api[api]
            print(f"   {api:<32} n={c:<6} bytes={b / _GIB:8.2f} GiB")
        print("   by caller library:")
        for (api, lib) in sorted(by_caller, key=lambda k: -by_caller[k][1]):
            c, b = by_caller[(api, lib)]
            print(f"   {api:<32} {lib:<28} n={c:<6} {b / _GIB:8.2f} GiB")
            grand[(api, lib)][0] += c
            grand[(api, lib)][1] += b
        if args.events:
            for e in events:
                print(f"   {e['lt']} {e['api']:<28} ret={e['ret']} size={e['size'] / _GIB:.3f}G dev={e['dev']} "
                      f"thr={e['thr']} handle={e['handle']} mc={e['mc']} <- {' <- '.join(e['frames'][:6])}")
        if args.memprobe:
            samples = parse_memprobe(args.memprobe, pid)
            if not samples:
                print(f"   (no MEMPROBE lines for pid {pid} in --memprobe files)")
            prev_t = None
            for t, fwd, du in samples:
                if du is not None and du >= args.min_click and prev_t is not None:
                    win = [e for e in events if prev_t < e["lt"][:8] <= t]
                    print(f"   CLICK fwd#{fwd} {prev_t}..{t} d_unattr=+{du:.2f}G: {len(win)} ledger events in window")
                    for e in win:
                        print(f"      {e['lt']} {e['api']:<26} ret={e['ret']} size={e['size'] / _GIB:.3f}G dev={e['dev']} "
                              f"thr={e['thr']} caller={e['caller']} <- {' <- '.join(e['frames'][:5])}")
                    if not win:
                        print("      -> nothing in this process: the bytes were bound/imported by ANOTHER process "
                              "(multicast pages from a peer, fabric import) or the driver -- check the peers' ledgers "
                              "for cuMulticastBindMem/cuMemExportToShareableHandle at this time")
                prev_t = t
    if len(paths) > 1:
        print("\n== all ledgers, by (api, caller library)")
        for (api, lib) in sorted(grand, key=lambda k: -grand[k][1]):
            c, b = grand[(api, lib)]
            print(f"   {api:<32} {lib:<28} n={c:<7} {b / _GIB:9.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
