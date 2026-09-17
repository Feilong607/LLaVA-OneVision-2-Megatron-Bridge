# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU memory owner account for the ``[MEMPROBE]`` line (``OV2_MEM_PROBE_DEVICE=1``).

Why
---
The merged48 production run ``q35-img38-a7-48gpu-psi16-3`` (2026-09-17) showed per-GPU ``nvidia-smi`` *used*
climbing in steps of identical byte counts inside each expert-TP pair while the sampled rank's torch
``reserved`` stayed flat, and the run before it died of CUDA OOM with ~38 GiB on the device that NVML
attributed to no process. The plain MEMPROBE line sees only torch's caching allocator; nvidia-smi sees only
totals. This helper puts every owner class on ONE line per sampled forward so a single run names the owner
of each memory step ("ratchet click").

Fields (all sizes GiB unless suffixed MiB; ``d_*`` = delta vs the previous sample of this process, ``?`` =
unavailable)
------------------------------------------------------------------------------------------------------
* ``t`` wall clock, ``pid`` -- to line MEMPROBE up with NCCL's ``host:pid:tid`` log prefix when
  ``NCCL_DEBUG_SUBSYS=INIT,ALLOC,NVLS`` is on (those lines carry no timestamp of their own).
* ``tp/etp/ep/dp`` -- parallel ranks, so lines can be grouped by expert-TP pair / EP rank.
* ``dev_used``  -- device total - free (``cudaMemGetInfo``): everything on the GPU, any owner.
* ``proc_used`` -- this PID per NVML (``nvmlDeviceGetComputeRunningProcesses``). ``pidmatch=0`` = the PID
  namespace hid us and the single compute process on the GPU was taken instead.
* ``res``       -- torch reserved; ``alloc`` -- torch allocated.
* ``nontorch_proc = proc_used - res`` -- raw cudaMalloc inside the process (NCCL buffers when not cuMem,
  nvshmem symmetric heap with ``NVSHMEM_DISABLE_CUDA_VMM=1``, loaded kernel modules, CUDA context).
* ``unattr = dev_used - proc_used`` -- memory NVML attributes to nobody (cuMem/VMM mappings of NCCL with
  ``NCCL_CUMEM_ENABLE=1``, NVLS multicast objects, HybridEP's ExtendedMemoryAllocator).
* ``segs/inactive_split/retries/ooms`` -- torch allocator fragmentation: number of cudaMalloc'd segments,
  bytes reserved-but-free inside split segments, ``num_alloc_retries`` (allocator had to free cached blocks
  and retry), ``num_ooms``. The -5 death had 37 GiB ``inactive_split``.
* ``pgs`` -- number of torch process groups (a new NCCL communicator = new NCCL buffers).
* ``hep`` -- identity counter of the HybridEP buffer (``fused_a2a._hybrid_ep_buffer``); increments when the
  buffer object was re-created (= reallocation). ``dep`` -- same for the DeepEP ``_buffer``.
* ``dyn`` -- torch.compile frame compilations so far; ``triton`` -- entries in ``TRITON_CACHE_DIR`` (new
  kernels = new cuModules in device memory).
* ``tokens/patches/nimg`` -- LLM sequence length, vision patch rows, image/video items of THIS forward;
  ``smax/pmax`` -- running maxima. A click while ``tokens<=smax and patches<=pmax`` is not a batch-level
  new-max-shape event.

Reading a click (``d_dev`` > 0 on some line)
--------------------------------------------
``d_res ≈ d_dev`` -> torch pool high-water (expandable_segments addresses it); ``d_nontorch ≈ d_dev`` -> a
library's cudaMalloc inside the process; ``d_unattr ≈ d_dev`` -> cuMem/NVLS/HybridEP -- then read the NCCL
ALLOC lines of that ``pid`` between this sample's ``t`` and the previous one. ``hep``/``dep``/``pgs`` moving
on the same line names the library directly.

One extra line ``[MEMPROBE-ENV r<rank>]`` is printed at the first sample: allocator conf, NCCL/nvshmem knobs,
HybridEP token cap and the RUNTIME MoE dispatcher type/backend of the model (settles "alltoall or hybridep"
from inside the process).

Cost: host-side queries only (one ``cudaMemGetInfo``, one NVML process list, dict reads, one ``os.listdir``);
no device sync. Any failing source degrades to ``?`` and is reported once. Default (env unset) = the plain
MEMPROBE line, unchanged.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MIB = 1024**2
_GIB = 1024**3

_STATE_ATTR = "_ov2_mem_probe_dev_state"

_ENV_KEYS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "OV2_CUDA_MEM_FRACTION",
    "OV2_FLEX_BACKEND",
    "HYBRID_EP_MAX_TOKENS_PER_RANK",
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NVSHMEM_DISABLE_CUDA_VMM",
    "NVSHMEM_SYMMETRIC_SIZE",
    "NCCL_CUMEM_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_MNNVL_ENABLE",
    "NCCL_ALGO",
    "NCCL_BUFFSIZE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "CUDA_VISIBLE_DEVICES",
)


def _shape_of(t: Any, dim: int) -> Optional[int]:
    try:
        return int(t.shape[dim])
    except Exception:  # None / not a tensor / scalar
        return None


def _nvml_handle_for_current_device(torch_mod):
    """NVML handle of torch's current device, matched by UUID (CUDA_VISIBLE_DEVICES-safe)."""
    import pynvml  # ships with torch (deprecated alias of nvidia-ml-py; both expose this API)

    pynvml.nvmlInit()
    dev = torch_mod.cuda.current_device()
    props = torch_mod.cuda.get_device_properties(dev)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        s = str(uuid)
        if not s.startswith("GPU-"):
            s = "GPU-" + s
        return pynvml.nvmlDeviceGetHandleByUUID(s.encode() if isinstance(s, str) else s)
    # Fallback: map the torch ordinal through CUDA_VISIBLE_DEVICES (only correct when it lists indices).
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    idx = int(vis.split(",")[dev]) if vis else dev
    return pynvml.nvmlDeviceGetHandleByIndex(idx)


def _proc_used_bytes(handle, pid: int) -> tuple[Optional[int], bool, int]:
    """(bytes NVML attributes to us, pid_matched, number of compute processes on the device).

    Inside a container without hostPID the PIDs NVML reports are host-namespace PIDs and never equal
    ``os.getpid()``; on an exclusively-assigned GPU there is exactly one compute process, so fall back to
    that single entry and flag ``pid_matched=False``. Two or more foreign processes -> (0, False, n)."""
    import pynvml

    getters = [
        getattr(pynvml, n, None)
        for n in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetComputeRunningProcesses_v2",
        )
    ]
    last_err: Optional[Exception] = None
    for g in getters:
        if g is None:
            continue
        try:
            procs = list(g(handle))
        except Exception as e:  # noqa: BLE001 -- try the next API flavour
            last_err = e
            continue
        for p in procs:
            if int(p.pid) == pid:
                used = p.usedGpuMemory
                return (None if used is None else int(used)), True, len(procs)
        if len(procs) == 1:
            used = procs[0].usedGpuMemory
            return (None if used is None else int(used)), False, 1
        return 0, False, len(procs)  # our pid not listed and the GPU is not exclusively ours
    if last_err is not None:
        raise last_err
    raise RuntimeError("pynvml has no nvmlDeviceGetComputeRunningProcesses")


def _dynamo_frames() -> Optional[int]:
    try:
        from torch._dynamo.utils import counters

        return int(sum(counters["frames"].values()))
    except Exception:  # noqa: BLE001
        return None


def _triton_cache_entries() -> Optional[int]:
    d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    try:
        return len(os.listdir(d))
    except OSError:
        return None


def _num_process_groups() -> Optional[int]:
    try:
        from torch.distributed import distributed_c10d as c10d

        return len(c10d._world.pg_map)
    except Exception:  # noqa: BLE001
        return None


def _parallel_ranks() -> str:
    try:
        from megatron.core import parallel_state as ps

        vals = []
        for name, fn in (
            ("tp", "get_tensor_model_parallel_rank"),
            ("etp", "get_expert_tensor_parallel_rank"),
            ("ep", "get_expert_model_parallel_rank"),
            ("dp", "get_data_parallel_rank"),
        ):
            try:
                vals.append(f"{name}={getattr(ps, fn)()}")
            except Exception:  # noqa: BLE001 -- group not initialized / API absent
                vals.append(f"{name}=?")
        return " ".join(vals)
    except Exception:  # noqa: BLE001
        return "tp=? etp=? ep=? dp=?"


def _comm_buffer_ids() -> tuple[Optional[int], Optional[int]]:
    """id() of the live HybridEP / DeepEP buffer objects (None = module absent or no buffer yet)."""
    try:
        import sys

        m = sys.modules.get("megatron.core.transformer.moe.fused_a2a")
        if m is None:
            return None, None
        hep = getattr(m, "_hybrid_ep_buffer", None)
        dep = getattr(m, "_buffer", None)
        return (id(hep) if hep is not None else 0), (id(dep) if dep is not None else 0)
    except Exception:  # noqa: BLE001
        return None, None


def _torch_alloc_stats(torch_mod) -> dict:
    try:
        st = torch_mod.cuda.memory_stats()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for key, name in (
        ("segment.all.current", "segs"),
        ("inactive_split_bytes.all.current", "inactive_split"),
        ("num_alloc_retries", "retries"),
        ("num_ooms", "ooms"),
    ):
        v = st.get(key)
        out[name] = None if v is None else int(v)
    return out


def _fmt_g(b: Optional[int]) -> str:
    return "?" if b is None else f"{b / _GIB:.2f}G"


def _fmt_d(b: Optional[int]) -> str:
    return "?" if b is None else f"{b / _GIB:+.2f}G"


def _q(v: Any) -> str:
    return "?" if v is None else str(v)


def _dispatcher_of(owner: Any) -> str:
    """Runtime MoE dispatcher type/backend of the model that owns the probe (what actually runs)."""
    try:
        cfg = owner.language_model.config
        return (
            f"dispatcher={getattr(cfg, 'moe_token_dispatcher_type', None)!s}"
            f" backend={getattr(cfg, 'moe_flex_dispatcher_backend', None)!s}"
            f" experts={getattr(cfg, 'num_moe_experts', None)!s} topk={getattr(cfg, 'moe_router_topk', None)!s}"
            f" etp={getattr(cfg, 'expert_tensor_parallel_size', None)!s}"
            f" ep={getattr(cfg, 'expert_model_parallel_size', None)!s}"
        )
    except Exception:  # noqa: BLE001
        return "dispatcher=? backend=?"


def env_line(owner: Any, rank: Any) -> str:
    """The one-time ``[MEMPROBE-ENV]`` line: knobs that decide who may own GPU memory, plus device totals."""
    parts = [f"[MEMPROBE-ENV r{rank}] pid={os.getpid()}", _parallel_ranks(), _dispatcher_of(owner)]
    parts += [f"{k}={os.environ.get(k, '')!r}" for k in _ENV_KEYS]
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        parts.append(f"dev_total={int(total) // _MIB}MiB dev_free_now={int(free) // _MIB}MiB")
        parts.append(f"torch={torch.__version__} nccl={'.'.join(map(str, torch.cuda.nccl.version()))}")
    except Exception:  # noqa: BLE001
        parts.append("dev_total=? torch=? nccl=?")
    return " ".join(parts)


def device_mem_suffix(owner: Any, input_ids: Any = None, images: Any = None, image_grid_thw: Any = None) -> str:
    """Return the extra fields for one MEMPROBE sample; state (previous sample, maxima, NVML handle) lives on
    ``owner`` so the same module instance produces deltas across calls. Never raises."""
    import torch  # lazy: this module must import without CUDA/torch for the offline selftest

    st = getattr(owner, _STATE_ATTR, None)
    if st is None:
        st = {
            "handle": None, "handle_err": False, "prev": {}, "smax": 0, "pmax": 0, "pid": os.getpid(),
            "hep_id": None, "hep_n": 0, "dep_id": None, "dep_n": 0, "ranks": None,
        }
        setattr(owner, _STATE_ATTR, st)
    if st["ranks"] is None:
        st["ranks"] = _parallel_ranks()

    dev_used: Optional[int] = None
    try:
        free, total = torch.cuda.mem_get_info()
        dev_used = int(total) - int(free)
    except Exception as e:  # noqa: BLE001
        if not st.get("meminfo_err"):
            st["meminfo_err"] = True
            logger.warning("[ov2 mem probe] cudaMemGetInfo failed: %s", e)

    proc_used: Optional[int] = None
    pid_ok: Optional[bool] = None
    nproc: Optional[int] = None
    if not st["handle_err"]:
        try:
            if st["handle"] is None:
                st["handle"] = _nvml_handle_for_current_device(torch)
            proc_used, pid_ok, nproc = _proc_used_bytes(st["handle"], st["pid"])
        except Exception as e:  # noqa: BLE001
            st["handle_err"] = True
            logger.warning("[ov2 mem probe] NVML per-process query unavailable (%s: %s); proc_used='?'",
                           type(e).__name__, e)

    try:
        reserved: Optional[int] = int(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001
        reserved = None
    try:
        allocated: Optional[int] = int(torch.cuda.memory_allocated())
    except Exception:  # noqa: BLE001
        allocated = None
    astats = _torch_alloc_stats(torch)

    nontorch = proc_used - reserved if (proc_used is not None and reserved is not None) else None
    unattr = dev_used - proc_used if (dev_used is not None and proc_used is not None) else None

    prev = st["prev"]

    def _d(key: str, cur: Optional[int]) -> Optional[int]:
        old = prev.get(key)
        prev[key] = cur
        return None if (cur is None or old is None) else cur - old

    d_dev, d_proc, d_res = _d("dev", dev_used), _d("proc", proc_used), _d("res", reserved)
    d_nt, d_un = _d("nt", nontorch), _d("un", unattr)
    d_alloc = _d("alloc", allocated)
    d_inact = _d("inact", astats.get("inactive_split"))

    # hep/dep = how many times the buffer object changed identity since the first sample (creation after the
    # first sample counts once; every re-creation = reallocation counts once more).
    hep_id, dep_id = _comm_buffer_ids()
    for key, cur in (("hep", hep_id), ("dep", dep_id)):
        if cur is None:
            continue
        if st[f"{key}_id"] is not None and cur != st[f"{key}_id"]:
            st[f"{key}_n"] += 1
        st[f"{key}_id"] = cur

    tokens = _shape_of(input_ids, 1) if _shape_of(input_ids, 1) is not None else _shape_of(input_ids, 0)
    patches = _shape_of(images, 0)
    nimg = _shape_of(image_grid_thw, 0)
    if tokens is not None:
        st["smax"] = max(st["smax"], tokens)
    if patches is not None:
        st["pmax"] = max(st["pmax"], patches)

    dev_mib = "?" if dev_used is None else f"{dev_used // _MIB}MiB"
    proc_mib = "?" if proc_used is None else f"{proc_used // _MIB}MiB"
    dyn = _dynamo_frames()
    return (
        f" t={time.strftime('%H:%M:%S')} pid={st['pid']} {st['ranks']}"
        f" dev_used={dev_mib} proc_used={proc_mib} pidmatch={'?' if pid_ok is None else int(pid_ok)}"
        f" nproc={_q(nproc)} alloc={_fmt_g(allocated)} res={_fmt_g(reserved)}"
        f" nontorch_proc={_fmt_g(nontorch)} unattr={_fmt_g(unattr)}"
        f" d_dev={_fmt_d(d_dev)} d_proc={_fmt_d(d_proc)} d_res={_fmt_d(d_res)} d_alloc={_fmt_d(d_alloc)}"
        f" d_nontorch={_fmt_d(d_nt)} d_unattr={_fmt_d(d_un)}"
        f" segs={_q(astats.get('segs'))} inactive_split={_fmt_g(astats.get('inactive_split'))}"
        f" d_inactive={_fmt_d(d_inact)} retries={_q(astats.get('retries'))} ooms={_q(astats.get('ooms'))}"
        f" pgs={_q(_num_process_groups())} hep={_q(None if hep_id is None else st['hep_n'])}"
        f" dep={_q(None if dep_id is None else st['dep_n'])}"
        f" dyn={_q(dyn)} triton={_q(_triton_cache_entries())}"
        f" tokens={_q(tokens)} patches={_q(patches)} nimg={_q(nimg)} smax={st['smax']} pmax={st['pmax']}"
    )
