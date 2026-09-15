# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline repro for the dataloader-worker host-memory growth seen on the merged48 production runs.

Runs the SAME energon pipeline a `pt_data_worker` runs (task encoder + energon dataset from the blend
yaml), in ONE process with num_workers=0, and every N samples prints three memory views side by side:

  rss      RssAnon of this process (/proc/self/status)       -- what the cgroup sees
  py       tracemalloc live bytes (Python-owned allocations)  -- Python-level retention
  malloc   glibc mallinfo2: in-use (uordblks+hblkhd) / free-but-held (fordblks)

Verdict logic (printed at the end, plus a malloc_trim(0) test):
  rss grows, py flat, malloc in-use flat, malloc free grows  => glibc heap fragmentation (allocator)
  rss grows, py grows                                        => Python-level per-sample retention;
                                                                 the tracemalloc top list names the line
  rss grows, py flat, malloc in-use grows                    => C/C++-level leak (PIL / PyAV / numpy /
                                                                 tokenizers), not visible to tracemalloc

CPU-only, no GPU, no torchrun. Needs the datasets mount and the HF processor dir. Typical run
(from ~/bridge-export, in the training image; no PYTHONPATH needed -- the script folds in src/,
Megatron-LM, aiak_shim, pylibs and the _verify_stubs sitecustomize itself):

  python examples/models/qwen/qwen35_vl_ov2/gb200/probe_worker_leak.py --n 300 --every 25

Env knobs: OV2_HF_PROC_QWEN35_P16M33 (processor dir; default $HOME/qwen35_p16m33_auto_model like the
production wrapper), OV2_PRETRAIN_ROOT, OV2_EXTRA_PYLIBS. To A/B an allocator fix, prefix the same command
with MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072, or (GB200 qwen35-fla image, aarch64)
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4, and compare the rss column only -- under
a non-glibc allocator the mallinfo2 columns describe glibc's (idle) arena and the verdict is meaningless.
"""
import argparse
import ctypes
import gc
import logging
import os
import sys
import time
import tracemalloc
import types
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[4]
for _p in (_REPO / "src", _REPO / "3rdparty" / "Megatron-LM", _REPO / "aiak_shim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# Offline packages the launchers fold in ($HOME/pylibs, $REPO/pylibs, OV2_EXTRA_PYLIBS), same precedence.
for _extra in [*filter(None, os.environ.get("OV2_EXTRA_PYLIBS", "").split(":")),
               str(_REPO / "pylibs"), os.path.join(os.path.expanduser("~"), "pylibs")]:
    if os.path.isdir(_extra) and _extra not in sys.path:
        sys.path.insert(0, _extra)
# The launchers put $REPO/_verify_stubs FIRST on PYTHONPATH so its sitecustomize runs at interpreter start
# (boto3<->botocore rename compat for this image's mixed dist-packages/venv, diffusers/modelopt stubs).
# A bare `python probe.py` skips that, and `from transformers import AutoProcessor` then dies in
# accelerate -> boto3. Load the same module by path (not by name: the system sitecustomize owns that name).
_STUBS = _REPO / "_verify_stubs" / "sitecustomize.py"
if _STUBS.is_file():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("_ov2_verify_stubs", _STUBS)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _libc():
    try:
        lib = ctypes.CDLL("libc.so.6")
        lib.mallinfo2.restype = _MallInfo2
        lib.malloc_trim.argtypes = [ctypes.c_size_t]
        lib.malloc_trim.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


def _rss_anon_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _malloc_mb(lib):
    if lib is None:
        return float("nan"), float("nan")
    mi = lib.mallinfo2()
    return (mi.uordblks + mi.hblkhd) / 2**20, mi.fordblks / 2**20


def _sample_bytes(batch):
    total = 0
    vals = batch.values() if isinstance(batch, dict) else (vars(batch).values() if hasattr(batch, "__dict__") else [])
    for v in vals:
        if hasattr(v, "numel") and hasattr(v, "element_size"):
            total += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            for t in v:
                if hasattr(t, "numel") and hasattr(t, "element_size"):
                    total += t.numel() * t.element_size()
    return total / 2**20


def _live_big_objects():
    """Count live large bytes / PIL images / torch tensors (the things a retained decoded sample is made of)."""
    import torch

    gc.collect()
    objs = gc.get_objects()
    big = [o for o in objs if type(o) is bytes and len(o) > 100_000]
    pil = [o for o in objs if type(o).__module__.startswith("PIL.") and hasattr(o, "size") and hasattr(o, "mode")]
    tens = [o for o in objs if torch.is_tensor(o) and o.device.type == "cpu"]
    arrs = [o for o in objs if type(o).__module__ == "numpy" and hasattr(o, "nbytes") and o.nbytes > 100_000]
    from collections import Counter

    hist = Counter((o.mode, o.size) for o in pil).most_common(3)
    # Whole-sample containers still alive (energon sample dataclasses, the encoder's SimpleNamespace subs,
    # OV2TaskSample) -- if these grow with n, entire samples are retained, not just their images.
    _names = ("PackedCaptioningSample", "MultiMixQASample", "OV2TaskSample", "SimpleNamespace", "OV2TaskBatch")
    samples = Counter(type(o).__name__ for o in objs if type(o).__name__ in _names)
    sample_objs = [o for o in objs if type(o).__name__ == "PackedCaptioningSample"]
    # Census of the things that can keep a sample alive: suspended generators / live frames (by function),
    # tracebacks and exceptions (an exception kept alive pins every local of every frame in its traceback),
    # and list_iterators (run 4: the retained images hang off a suspended iteration over an image list).
    import inspect

    gens = Counter(f"{g.gi_code.co_name}@{os.path.basename(g.gi_code.co_filename)}" for g in objs if inspect.isgenerator(g))
    frames = Counter(f"{f.f_code.co_name}@{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" for f in objs if isinstance(f, types.FrameType))
    tbs = [o for o in objs if isinstance(o, types.TracebackType)]
    tb_where = Counter(f"{t.tb_frame.f_code.co_name}@{os.path.basename(t.tb_frame.f_code.co_filename)}:{t.tb_lineno}" for t in tbs)
    excs = Counter(type(o).__name__ for o in objs if isinstance(o, BaseException))
    list_iters = [o for o in objs if type(o).__name__ == "list_iterator"]
    return {
        "pil_hist": hist, "_pil": pil, "samples": dict(samples), "_samples": sample_objs,
        "census": {"generators": sum(gens.values()), "gen_top": gens.most_common(6),
                   "frames_top": frames.most_common(6), "tracebacks": len(tbs), "tb_top": tb_where.most_common(4),
                   "exceptions": dict(excs.most_common(4)), "list_iterators": len(list_iters), "gc_garbage": len(gc.garbage)},
        "_list_iters": list_iters,
        "big_bytes": (len(big), sum(len(o) for o in big) / 2**20),
        "pil_images": (len(pil), sum((o.size[0] * o.size[1] * len(o.getbands())) for o in pil) / 2**20),
        "cpu_tensors": (len(tens), sum(t.numel() * t.element_size() for t in tens) / 2**20),
        "np_arrays": (len(arrs), sum(o.nbytes for o in arrs) / 2**20),
        "_big": big, "_frame_t": types.FrameType,
    }


def _report_live(live):
    """Log the census dict from _live_big_objects (tuple entries = (count, MB); dict entries verbatim)."""
    counts = "  ".join(f"{k}={v[0]} ({v[1]:.0f}M)" for k, v in live.items() if not k.startswith("_") and isinstance(v, tuple))
    logger.info("[probe]      live: " + counts)
    logger.info(f"[probe]      live sample objects: {live['samples']}")
    logger.info(f"[probe]      live PIL (mode,size) top3: {live['pil_hist']}")
    logger.info(f"[probe]      census: {live['census']}")


def _who_holds(sample_objs, frame_t, depth=16, chains=3, skip=()):
    """Walk gc.get_referrers upward from a few retained objects and name the containers (type, dict key, len)."""
    import inspect

    skip_ids = {id(sample_objs), *(id(x) for x in skip)}
    _self_file = os.path.abspath(__file__)

    def _frame_desc(fr, target):
        names = [k for k, v in fr.f_locals.items() if v is target]
        return f"frame[{fr.f_code.co_name} @ {os.path.basename(fr.f_code.co_filename)}:{fr.f_lineno}, var={names[:2]}]"

    for o in sample_objs[:chains]:
        _sz = getattr(o, "size", None) if not isinstance(o, (bytes, bytearray)) else len(o)
        cur, seen, chain = o, {id(o)}, [f"{type(o).__name__}({_sz})"]
        # Where was this retained object allocated? (tracemalloc is on) -> original decode vs. a copy.
        tb = tracemalloc.get_object_traceback(o)
        if tb is not None:
            frames = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in list(tb)[-6:]]
            logger.info(f"[probe] allocated at (innermost last): {' > '.join(frames)}")
        for _ in range(depth):
            refs = [r for r in gc.get_referrers(cur)
                    if id(r) not in seen and id(r) not in skip_ids and not inspect.isroutine(r) and r is not chain
                    and not (isinstance(r, frame_t) and os.path.abspath(r.f_code.co_filename) == _self_file)]
            seen.add(id(refs))   # this very list references cur -- never climb into it next level
            if not refs:
                chain.append("<no referrers>")
                break
            # Prefer a non-container object (it names the owner); else the first container.
            r = next((x for x in refs if not isinstance(x, (dict, list, tuple, set, frame_t))), refs[0])
            desc = type(r).__name__
            if isinstance(r, frame_t):
                desc = _frame_desc(r, cur)
            elif inspect.isgenerator(r):
                _gl = r.gi_frame.f_lineno if r.gi_frame is not None else "finished"
                desc = f"generator[{r.gi_code.co_name} @ {os.path.basename(r.gi_code.co_filename)}:{_gl}]"
            elif isinstance(r, types.ModuleType):
                chain.append(f"module[{r.__name__}]")
                break
            elif isinstance(r, types.TracebackType):
                desc = f"traceback[{r.tb_frame.f_code.co_name} @ {os.path.basename(r.tb_frame.f_code.co_filename)}:{r.tb_lineno}]"
            elif isinstance(r, BaseException):
                desc = f"exception[{type(r).__name__}: {str(r)[:60]!r}]"
            elif type(r).__name__ == "list_iterator":
                desc = "list_iterator"
            elif isinstance(r, dict):
                keys = [k for k, v in r.items() if v is cur]
                desc += f"[key={keys[:2]!r}, len={len(r)}]"
            elif isinstance(r, (list, tuple, set)):
                desc += f"[len={len(r)}]"
            else:
                attrs = [a for a, v in getattr(r, "__dict__", {}).items() if v is cur]
                if attrs:
                    desc += f".{attrs[0]}"
            if len(refs) > 1:
                _others = []
                for x in refs[1:4]:
                    _d = type(x).__name__
                    if isinstance(x, frame_t):
                        _d = _frame_desc(x, cur)
                    elif isinstance(x, dict):
                        _d += f"[keys={[k for k, v in x.items() if v is cur][:2]!r}]"
                    elif isinstance(x, (list, tuple)):
                        _d += f"[len={len(x)}]"
                    _others.append(_d)
                desc += f" (+{len(refs) - 1} other referrers: {_others})"
            chain.append(desc)
            seen.add(id(r))
            cur = r
        logger.info("[probe] holder chain: " + "  <-  ".join(chain))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(_HERE / "stage3_img38_video62_maveric.yaml"), help="blend yaml (energon Metadataset)")
    ap.add_argument("--proc", default=None, help="HF processor dir (default: the qwen3.5 backbone's hf_proc)")
    ap.add_argument("--seq", type=int, default=73728)
    ap.add_argument("--n", type=int, default=300, help="samples to pull")
    ap.add_argument("--every", type=int, default=25, help="report interval (samples)")
    ap.add_argument("--buffer", type=int, default=8, help="shuffle_buffer_size (production: 8)")
    ap.add_argument("--top", type=int, default=8, help="tracemalloc top-N lines per report")
    ap.add_argument("--merge", type=int, default=3, help="spatial_merge_size (qwen3.5 p16m33: 3)")
    args = ap.parse_args()

    import megatron.bridge.recipes.ov2.ov2_qwen35 as q35  # registers the backbone
    from megatron.bridge.recipes.ov2.ov2 import _OV2_BACKBONES
    from megatron.bridge.recipes.ov2.data.energon.task_encoder import OV2TaskEncoder
    from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset

    # Processor dir resolution mirrors the production wrapper (ax_ov2_qwen35_merged48.sh): explicit flag,
    # then $OV2_HF_PROC_QWEN35_P16M33, then $HOME/qwen35_p16m33_auto_model, then the backbone default.
    _cands = [args.proc, os.environ.get("OV2_HF_PROC_QWEN35_P16M33"),
              os.path.join(os.path.expanduser("~"), "qwen35_p16m33_auto_model"),
              _OV2_BACKBONES[q35._QWEN35_BACKBONE]["hf_proc"]]
    proc = next((c for c in _cands if c and os.path.isfile(os.path.join(c, "preprocessor_config.json"))), None)
    if proc is None:
        sys.exit(f"[probe] FATAL: no HF processor dir with preprocessor_config.json among {[c for c in _cands if c]}; pass --proc")
    logger.info(f"[probe] repo={_REPO} data={args.data} proc={proc} seq={args.seq} buffer={args.buffer} n={args.n}")
    logger.info(f"[probe] pid={os.getpid()} MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX')} "
          f"MALLOC_MMAP_THRESHOLD_={os.environ.get('MALLOC_MMAP_THRESHOLD_')} "
          f"MALLOC_TRIM_THRESHOLD_={os.environ.get('MALLOC_TRIM_THRESHOLD_')} LD_PRELOAD={os.environ.get('LD_PRELOAD')}")

    lib = _libc()
    tracemalloc.start(12)
    te = OV2TaskEncoder(hf_processor_path=proc, seq_length=args.seq, spatial_merge_size=args.merge)
    # The Qwen3 s2/s3 runs (feilong-nemo image) never showed this growth on the same code, data and
    # settings, so the dependency stack is a prime suspect -- print what THIS image resolves to.
    import numpy, PIL, torch, transformers
    import megatron.energon as energon
    _ip = getattr(te.proc, "image_processor", None)
    logger.info(f"[probe] stack: torch={torch.__version__} transformers={transformers.__version__} PIL={PIL.__version__} "
                f"numpy={numpy.__version__} energon={getattr(energon, '__version__', '?')} "
                f"processor={type(te.proc).__name__} image_processor={type(_ip).__name__} "
                f"tokenizer={type(getattr(te.proc, 'tokenizer', te.proc)).__name__} "
                f"torch_threads={torch.get_num_threads()}")
    wc = WorkerConfig.default_worker_config(0)  # in-process: this process IS the worker
    ds = get_train_dataset(
        args.data, batch_size=1, task_encoder=te, worker_config=wc, split_part="train",
        shuffle_buffer_size=args.buffer, max_samples_per_sequence=None, packing_buffer_size=None,
        image_decode="pil", parallel_shard_iters=int(os.environ.get("OV2_PARALLEL_SHARD_ITERS", "16")),
    )
    loader = get_savable_loader(ds, worker_config=wc)
    gc.collect()
    base_snap = tracemalloc.take_snapshot()
    rss0, (mu0, mf0) = _rss_anon_mb(), _malloc_mb(lib)
    py0 = tracemalloc.get_traced_memory()[0] / 2**20
    logger.info(f"[probe] baseline: rss={rss0:.0f}M py={py0:.0f}M malloc_inuse={mu0:.0f}M malloc_free={mf0:.0f}M")
    logger.info("[probe]   n   rss_M  d_rss   py_M  d_py  minuse_M d_minuse  mfree_M d_mfree  batch_MB  s/sample")

    t0 = time.time()
    n = 0
    hist = []
    batch_mb = 0.0
    for batch in loader:
        n += 1
        batch_mb += _sample_bytes(batch)
        del batch
        if n % args.every == 0 or n == args.n:
            gc.collect()
            rss, (mu, mf) = _rss_anon_mb(), _malloc_mb(lib)
            py = tracemalloc.get_traced_memory()[0] / 2**20
            hist.append((n, rss, py, mu, mf))
            logger.info(f"[probe] {n:4d} {rss:7.0f} {rss - rss0:+6.0f} {py:6.0f} {py - py0:+5.0f} {mu:9.0f} {mu - mu0:+8.0f} {mf:8.0f} {mf - mf0:+7.0f} "
                  f"{batch_mb / args.every:9.1f} {(time.time() - t0) / n:9.2f}")
            batch_mb = 0.0
            snap = tracemalloc.take_snapshot()
            stats = [s for s in snap.compare_to(base_snap, "lineno") if "tracemalloc" not in str(s.traceback)]
            for s in stats[: args.top]:
                fr = s.traceback[0]
                logger.info(f"[probe]      py top: {s.size_diff / 2**20:+8.1f}M ({s.count_diff:+d} blocks) {fr.filename}:{fr.lineno}")
            live = _live_big_objects()
            _report_live(live)
            pil = big = None
            if n == args.n or n == args.every * 2:
                pil, big, smp, lis = live["_pil"], live["_big"], live["_samples"], live["_list_iters"]
                _mine = (live, pil, big, smp, lis)   # the probe's own containers must not show up as "holders"
                # (a) oldest retained PIL images (gc.get_objects is roughly allocation-ordered)
                _who_holds(pil[: 2] if pil else big, live["_frame_t"], skip=_mine)
                # (b) oldest retained whole samples -- far fewer levels to the holder
                if smp:
                    logger.info(f"[probe] oldest PackedCaptioningSample refcount={sys.getrefcount(smp[0]) - 2}")
                    _who_holds(smp[: 2], live["_frame_t"], chains=2, skip=_mine)
                # (c) list_iterators whose underlying list holds PIL images: which list, where in it, who holds it
                _pil_iters = []
                for li in lis:
                    try:
                        _, (lst,), idx = li.__reduce__()
                    except Exception:
                        continue
                    if lst and type(lst[0]).__module__.startswith("PIL."):
                        _pil_iters.append((li, len(lst), idx))
                logger.info(f"[probe] list_iterators over PIL lists: {len(_pil_iters)} (of {len(lis)} list_iterators)")
                for li, ln, idx in _pil_iters[:2]:
                    logger.info(f"[probe]   iterator at index {idx}/{ln}:")
                    _who_holds([li], live["_frame_t"], chains=1, skip=_mine + (_pil_iters,))
                del smp, lis, _pil_iters
            del live, pil, big
        if n >= args.n:
            break

    # ---- release experiment: does tearing down the pipeline free the retained samples? ----
    # Freed => the holder lives inside the energon dataset/loader object state. Still alive => a module-level
    # cache, an exception/traceback, or another global keeps them.
    before = _live_big_objects()
    b_pil, b_smp = before["pil_images"][0], before["samples"].get("PackedCaptioningSample", 0)
    del before
    del loader, ds, te
    gc.collect()
    after = _live_big_objects()
    logger.info(f"[probe] release test: del loader/dataset/encoder + gc -> PIL images {b_pil} -> {after['pil_images'][0]}, "
                f"PackedCaptioningSample {b_smp} -> {after['samples'].get('PackedCaptioningSample', 0)}, rss={_rss_anon_mb():.0f}M")
    if after["samples"].get("PackedCaptioningSample", 0) > 2 and after["_samples"]:
        logger.info("[probe] still held after teardown -- the holder is outside the pipeline objects:")
        _who_holds(after["_samples"][:2], after["_frame_t"], chains=2, skip=(after, after["_samples"], after["_pil"], after["_list_iters"]))
    del after

    # ---- verdict ----
    rss1, py1, mu1, mf1 = hist[-1][1], hist[-1][2], hist[-1][3], hist[-1][4]
    d_rss, d_py, d_mu, d_mf = rss1 - rss0, py1 - py0, mu1 - mu0, mf1 - mf0
    trimmed = lib.malloc_trim(0) if lib is not None else -1
    rss_after_trim = _rss_anon_mb()
    logger.info(f"[probe] after malloc_trim(0): rss={rss_after_trim:.0f}M (released {rss1 - rss_after_trim:+.0f}M; trim rc={trimmed})")
    # second-half slope: is growth still linear or flattening?
    if len(hist) >= 4:
        h = len(hist) // 2
        s1 = (hist[h][1] - hist[0][1]) / max(1, hist[h][0] - hist[0][0])
        s2 = (hist[-1][1] - hist[h][1]) / max(1, hist[-1][0] - hist[h][0])
        logger.info(f"[probe] rss slope MB/sample: first half {s1:.2f}, second half {s2:.2f}")
    grew = d_rss > 200
    if not grew:
        verdict = "NO significant RSS growth in-process (<200 MB); the production growth needs the worker/IPC path (torch shm sends) -- rerun with --workers via the real loader"
    elif d_py > 0.5 * d_rss:
        verdict = "PYTHON-LEVEL retention: tracemalloc grows with RSS; see the 'py top' lines for the allocating line"
    elif d_mu > 0.5 * d_rss:
        verdict = "C-LEVEL leak: malloc in-use grows with RSS but Python does not see it (PIL/PyAV/numpy/tokenizers)"
    elif d_mf > 0.3 * d_rss or (rss1 - rss_after_trim) > 0.3 * d_rss:
        verdict = "GLIBC FRAGMENTATION: RSS grows while malloc in-use is flat and free-but-held grows / trim releases it -> MALLOC_MMAP_THRESHOLD_/TRIM_THRESHOLD_ or jemalloc"
    else:
        verdict = "UNCLASSIFIED: RSS grows but neither py, malloc in-use nor malloc free explains it (mmap outside malloc? torch shm?)"
    logger.info(f"[probe] VERDICT over {n} samples: d_rss={d_rss:+.0f}M d_py={d_py:+.0f}M d_malloc_inuse={d_mu:+.0f}M d_malloc_free={d_mf:+.0f}M -> {verdict}")


if __name__ == "__main__":
    main()
