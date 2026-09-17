# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline test of the MEMPROBE three-level account (no GPU, no real torch: fake ``torch``/``pynvml``).

Run directly (``python tests/unit_tests/models/ov2/test_ov2_mem_probe.py``) or via pytest. Unlike the other
OV2 tests this one must not import ``ov2_bridge``; ``ov2_mem_probe`` imports torch lazily so it works here.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types


class _T:
    def __init__(self, *shape):
        self.shape = tuple(shape)


class _Fake:
    """Mutable fake device: total, free, this-pid used, torch reserved."""

    total = 190 * 1024**3
    free = 30 * 1024**3
    proc = 150 * 1024**3
    reserved = 140 * 1024**3
    pid_listed = True
    handle_by_uuid_ok = True
    frames = {"ok": 3}
    segs = 40
    inactive = 3 * 1024**3
    pg_map = {"a": 1, "b": 2, "c": 3}
    hep_buf = None  # HybridEP buffer not created yet at the first sample


def _install_fakes(monkeypatch_modules: dict):
    f = _Fake

    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.mem_get_info = lambda: (f.free, f.total)
    cuda.memory_reserved = lambda: f.reserved
    cuda.memory_allocated = lambda: f.reserved - 10 * 1024**3
    cuda.memory_stats = lambda: {"segment.all.current": f.segs, "inactive_split_bytes.all.current": f.inactive,
                                 "num_alloc_retries": 2, "num_ooms": 0}
    cuda.current_device = lambda: 0
    nccl = types.ModuleType("torch.cuda.nccl")
    nccl.version = lambda: (2, 29, 3)
    cuda.nccl = nccl
    torch.__version__ = "2.9.0-fake"

    class _Props:
        uuid = "0123abcd-0000-0000-0000-000000000001"

    cuda.get_device_properties = lambda dev: _Props()
    torch.cuda = cuda
    dyn = types.ModuleType("torch._dynamo")
    utils = types.ModuleType("torch._dynamo.utils")
    utils.counters = {"frames": f.frames}
    dyn.utils = utils
    torch._dynamo = dyn

    dist = types.ModuleType("torch.distributed")
    c10d = types.ModuleType("torch.distributed.distributed_c10d")
    c10d._world = types.SimpleNamespace(pg_map=f.pg_map)
    dist.distributed_c10d = c10d
    torch.distributed = dist

    fa2a = types.ModuleType("megatron.core.transformer.moe.fused_a2a")
    fa2a._hybrid_ep_buffer = f.hep_buf
    fa2a._buffer = None

    pynvml = types.ModuleType("pynvml")
    pynvml.nvmlInit = lambda: None

    def by_uuid(u):
        assert u == b"GPU-0123abcd-0000-0000-0000-000000000001", u
        if not f.handle_by_uuid_ok:
            raise RuntimeError("no uuid lookup")
        return "H"

    pynvml.nvmlDeviceGetHandleByUUID = by_uuid
    pynvml.nvmlDeviceGetHandleByIndex = lambda i: "H"

    class _P:
        def __init__(self, pid, used):
            self.pid, self.usedGpuMemory = pid, used

    def procs(h):
        assert h == "H"
        out = [_P(1, 5 * 1024**3)]
        if f.pid_listed:
            out.append(_P(os.getpid(), f.proc))
        return out

    pynvml.nvmlDeviceGetComputeRunningProcesses = procs

    for name, mod in (("torch", torch), ("torch.cuda", cuda), ("torch.cuda.nccl", nccl), ("torch._dynamo", dyn),
                      ("torch._dynamo.utils", utils), ("torch.distributed", dist),
                      ("torch.distributed.distributed_c10d", c10d),
                      ("megatron.core.transformer.moe.fused_a2a", fa2a), ("pynvml", pynvml)):
        monkeypatch_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod


def _restore(saved: dict):
    for name, old in saved.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class _FP:
    def __init__(self, pid, used):
        self.pid, self.usedGpuMemory = pid, used


def _field(line: str, key: str) -> str:
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    raise AssertionError(f"{key} missing in {line!r}")


def test_three_level_account_and_deltas():
    saved: dict = {}
    _install_fakes(saved)
    try:
        # Load by path: importing the package would pull the whole bridge (megatron.core, HF) into this test.
        _path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "megatron", "bridge",
                             "models", "qwen_vl_ov2", "ov2_mem_probe.py")
        _spec = importlib.util.spec_from_file_location("ov2_mem_probe_under_test", _path)
        mp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(mp)
        G = 1024**3
        owner = types.SimpleNamespace()

        s1 = mp.device_mem_suffix(owner, _T(1, 8192), _T(4000, 1176), _T(3, 3))
        assert _field(s1, "dev_used") == f"{160 * 1024}MiB", s1          # 190 - 30
        assert _field(s1, "proc_used") == f"{150 * 1024}MiB", s1
        assert _field(s1, "nontorch_proc") == "10.00G", s1                 # 150 - 140
        assert _field(s1, "unattr") == "10.00G", s1                        # 160 - 150
        assert _field(s1, "d_dev") == "?" and _field(s1, "d_res") == "?", s1  # first sample: no delta
        assert _field(s1, "tokens") == "8192" and _field(s1, "patches") == "4000" and _field(s1, "nimg") == "3"
        assert _field(s1, "smax") == "8192" and _field(s1, "pmax") == "4000"
        assert _field(s1, "dyn") == "3" and _field(s1, "pidmatch") == "1" and _field(s1, "nproc") == "2"
        assert _field(s1, "alloc") == "130.00G" and _field(s1, "res") == "140.00G", s1
        assert _field(s1, "segs") == "40" and _field(s1, "inactive_split") == "3.00G", s1
        assert _field(s1, "retries") == "2" and _field(s1, "ooms") == "0", s1
        assert _field(s1, "pgs") == "3" and _field(s1, "hep") == "0" and _field(s1, "dep") == "0", s1
        assert _field(s1, "tp") == "?" and _field(s1, "etp") == "?", s1      # no megatron.core here
        assert _field(s1, "pid") == str(os.getpid()) and len(_field(s1, "t")) == 8, s1
        assert _field(s1, "triton") in ("?",) or _field(s1, "triton").isdigit(), s1
        # ENV line: knobs + runtime dispatcher (owner has no language_model -> '?')
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        e = mp.env_line(owner, 7)
        assert e.startswith("[MEMPROBE-ENV r7] pid=") and "NCCL_CUMEM_ENABLE='1'" in e, e
        assert "dispatcher=? backend=?" in e and "nccl=2.29.3" in e and f"dev_total={190 * 1024}MiB" in e, e
        cfg = types.SimpleNamespace(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep",
                                    num_moe_experts=256, moe_router_topk=8, expert_tensor_parallel_size=2,
                                    expert_model_parallel_size=8)
        e2 = mp.env_line(types.SimpleNamespace(language_model=types.SimpleNamespace(config=cfg)), 0)
        assert "dispatcher=flex backend=hybridep experts=256 topk=8 etp=2 ep=8" in e2, e2

        # ratchet click: device grows 2 GiB, process grows 2 GiB, torch reserved flat -> nontorch_proc +2
        _Fake.free -= 2 * G
        _Fake.proc += 2 * G
        _Fake.frames["ok"] = 4
        _Fake.inactive += 1 * G
        _Fake.pg_map["d"] = 4
        sys.modules["megatron.core.transformer.moe.fused_a2a"]._hybrid_ep_buffer = object()  # created now
        s2 = mp.device_mem_suffix(owner, _T(1, 4096), _T(1000, 1176), _T(1, 3))
        assert _field(s2, "d_inactive") == "+1.00G" and _field(s2, "pgs") == "4", s2
        assert _field(s2, "hep") == "1" and _field(s2, "d_alloc") == "+0.00G", s2
        assert _field(s2, "d_dev") == "+2.00G" and _field(s2, "d_proc") == "+2.00G", s2
        assert _field(s2, "d_res") == "+0.00G" and _field(s2, "d_nontorch") == "+2.00G", s2
        assert _field(s2, "d_unattr") == "+0.00G", s2
        assert _field(s2, "tokens") == "4096" and _field(s2, "smax") == "8192" and _field(s2, "pmax") == "4000", s2
        assert _field(s2, "dyn") == "4"

        # unattributed click: device grows 3 GiB, process unchanged
        _Fake.free -= 3 * G
        sys.modules["megatron.core.transformer.moe.fused_a2a"]._hybrid_ep_buffer = object()  # re-created
        s3 = mp.device_mem_suffix(owner, None, None, None)
        assert _field(s3, "hep") == "2" and _field(s3, "dep") == "0", s3

        # feed the three real lines (+ NCCL ALLOC lines of our pid) through the report tool
        pid = os.getpid()
        log = [
            f"[MEMPROBE r5] fwd#4 allocated=1.0G max_allocated=1.0G reserved=1.0G max_reserved=1.0G{s1}",
            f"host:{pid}:{pid + 1} [0] NCCL INFO transport/p2p.cc:123 Cuda Alloc Size {2 * G} pointer 0x1",
            f"host:{pid}:{pid + 1} [0] NCCL INFO comm 0xabc rank 5 nranks 16 cudaDev 1 ... Init COMPLETE",
            f"host:999:1000 [0] NCCL INFO CUMEM Alloc Size {7 * G} pointer 0x2",  # another pid: not ours
            f"[MEMPROBE r5] fwd#8 allocated=1.0G max_allocated=1.0G reserved=1.0G max_reserved=1.0G{s2}",
            f"[MEMPROBE r5] fwd#12 allocated=1.0G max_allocated=1.0G reserved=1.0G max_reserved=1.0G{s3}",
            "[MEMPROBE r6] fwd#4 allocated=1.0G max_allocated=1.0G reserved=1.0G max_reserved=1.0G",  # plain line
        ]
        _rp = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples", "models", "qwen",
                           "qwen35_vl_ov2", "gb200", "mem_probe_report.py")
        _rspec = importlib.util.spec_from_file_location("mem_probe_report_under_test", _rp)
        rp = importlib.util.module_from_spec(_rspec)
        _rspec.loader.exec_module(rp)
        rows = list(rp.parse(log))
        assert [r["fwd"] for r in rows] == [4, 8, 12], rows
        assert rows[1]["nccl_alloc"] == 2 * G and rows[1]["nccl_alloc_n"] == 1 and rows[1]["nccl_init"] == 1, rows[1]
        assert rows[2]["nccl_alloc"] == 0, rows[2]
        assert rp._owner(rows[1]) == ("nontorch_proc", 1.0), rp._owner(rows[1])
        assert rp._owner(rows[2]) == ("unattr", 1.0), rp._owner(rows[2])
        assert rp._owner(rows[0]) == ("?", 0.0)
        import io
        import contextlib
        buf = io.StringIO()
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"memprobe_selftest_{pid}.log")
        with open(tmp, "w") as fh:
            fh.write("\n".join(log) + "\n")
        try:
            with contextlib.redirect_stdout(buf):
                rc = rp.main([tmp, "--min-click", "1.0"])
        finally:
            os.unlink(tmp)
        out = buf.getvalue()
        assert rc == 0, out
        assert "CLICK fwd#8" in out and "-> nontorch_proc (100%)" in out and "nccl_alloc=2.00G/1" in out, out
        assert "CLICK fwd#12" in out and "-> unattr (100%)" in out and "HEP-REALLOC" in out, out
        assert "== rank 5 pid" in out and "rank 6" not in out, out
        import re
        assert re.search(r"unattr\s+3\.00 GiB\s+60\.0%", out) and re.search(r"nontorch_proc\s+2\.00 GiB\s+40\.0%", out), out
        assert _field(s3, "d_dev") == "+3.00G" and _field(s3, "d_proc") == "+0.00G", s3
        assert _field(s3, "d_unattr") == "+3.00G" and _field(s3, "d_nontorch") == "+0.00G", s3
        assert _field(s3, "tokens") == "?" and _field(s3, "patches") == "?" and _field(s3, "nimg") == "?", s3

        # container without hostPID: our pid is not in NVML's list but the GPU has exactly one process ->
        # use that entry, pidmatch=0
        _Fake.pid_listed = False
        s4 = mp.device_mem_suffix(owner, _T(1, 100), None, None)
        assert _field(s4, "proc_used") == f"{5 * 1024}MiB" and _field(s4, "pidmatch") == "0", s4
        assert _field(s4, "unattr") == "160.00G", s4
        # two foreign processes and not ours -> 0, pidmatch=0
        sys.modules["pynvml"].nvmlDeviceGetComputeRunningProcesses = lambda h: [_FP(1, G), _FP(2, G)]
        s4b = mp.device_mem_suffix(owner, _T(1, 100), None, None)
        assert _field(s4b, "proc_used") == "0MiB" and _field(s4b, "pidmatch") == "0", s4b
        assert _field(s4b, "unattr") == "165.00G", s4b

        # NVML broken from the start on a fresh owner -> proc fields '?', device/torch fields still present
        _Fake.pid_listed = True
        _Fake.handle_by_uuid_ok = False
        pynvml = sys.modules["pynvml"]
        pynvml.nvmlDeviceGetHandleByIndex = lambda i: (_ for _ in ()).throw(RuntimeError("no index"))
        owner2 = types.SimpleNamespace()
        s5 = mp.device_mem_suffix(owner2, _T(1, 7), _T(9, 1), _T(1, 3))
        assert _field(s5, "proc_used") == "?" and _field(s5, "nontorch_proc") == "?" and _field(s5, "unattr") == "?", s5
        assert _field(s5, "dev_used") == f"{165 * 1024}MiB", s5
        s6 = mp.device_mem_suffix(owner2, _T(1, 7), _T(9, 1), _T(1, 3))
        assert _field(s6, "d_dev") == "+0.00G" and _field(s6, "d_proc") == "?", s6
        assert getattr(owner2, mp._STATE_ATTR)["handle_err"] is True  # not retried every sample
    finally:
        _restore(saved)


if __name__ == "__main__":
    test_three_level_account_and_deltas()
    print("SELFTEST OK")
