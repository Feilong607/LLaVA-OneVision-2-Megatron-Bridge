#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a Qwen3.5 OV2 training SAVE (torch_dist) against its HF export, on CPU, family by family.

WHY THIS EXISTS -- the gap the other two checkers leave open:

  * ``validate_hf_export.py`` proves STRUCTURE (keys / shapes / shards / dtypes). Every value is unproven.
  * ``verify_export_parity.py`` proves EP-LAYOUT INVARIANCE: the same iteration exported twice at two
    expert-parallel sizes must be byte-identical. Both legs, however, build a TP=1/PP=1/EP=world model
    (``ov2_30b_export_ep8.py``) and run the SAME mapping registry, so anything identical at every EP -- a
    wrong key pairing, a wrong transform, a tensor silently dropped and randomly initialised, or a bad TP
    merge on load -- is common-mode and PASSES there.

The merged video stage saves at **TP4** (the validated s1.5 line was TP1), so its export exercises a
TP4 -> TP1 merge for the first time, and that merge is common-mode across both parity legs: it is exactly
what the EP differential cannot see. This tool closes the missing side by comparing the export against the
CHECKPOINT it came from.

WHAT IT DOES

  1. Loads selected SAVE tensors as GLOBAL (unsharded) tensors straight from the torch_dist files. A
     torch_dist checkpoint records global shapes plus per-chunk offsets, so the read is independent of the
     TP/EP the SAVE was written with -- and, by going through ``torch.distributed.checkpoint`` directly, it
     does not re-execute the mcore model build or the bridge mapping registry it is checking.
  2. Reads the matching tensors out of the HF export's safetensors (row slices; nothing large is
     materialised twice).
  3. Asks, per family: **is the HF tensor a known REARRANGEMENT of the SAVE tensor(s), and is it the
     canonical one?** The candidate set deliberately holds the canonical arrangement AND the classic traps
     (contiguous-vs-GQA-interleaved QKV, TP-rank-interleaved gate/up, swapped halves, transposes, a
     zero-centred norm exported without its +1, fused parts concatenated in the wrong order...). The report
     names the arrangement that matched.

     A trap arrangement matching is a FAIL with a diagnosis. No arrangement matching is a FAIL with the
     smallest max-abs-diff reached. A family whose keys resolve on neither side is UNRESOLVED -- never a
     pass -- and the row lists the sibling keys that DO exist on each side, so one run is enough to fix it.

NAMES AND SHAPES ARE DISCOVERED, NOT ASSUMED. Both key spaces are read first: which layers carry attention
vs Gated DeltaNet, how experts are stored (``mlp.experts.experts.linear_fc1.weightN`` in this build, a
stacked tensor or ``local_experts.N`` elsewhere), whether the GDN input projection is one fused matrix or
split per role (``in_proj.weight.{query,key,value,z,beta,alpha}``), and whether attention is gated -- the
q side of a Qwen3.5 attention layer carries its gate, so the q/k/v split is taken from the HF tensors' own
row counts rather than from head counts.

WHAT IT PROVES / DOES NOT PROVE

  Proves, for every family it checks: the exported values ARE the trained values up to the named
  rearrangement -- so a dropped, re-initialised, mispaired or mis-transformed tensor is caught even when it
  is wrong at every EP and every TP.

  Does NOT prove: that the two halves of a fused pair carry the semantics their names claim (bytes cannot
  tell ``gate`` from ``up``; what this CAN tell you is which arrangement the exporter used, which is the
  actionable half of that question); HF-vs-mcore logits; M-RoPE at inference; MTP semantics; or anything
  about families it did not sample.

USAGE (CPU pod or the export workspace; no GPU, no torchrun)

    python3 save_vs_hf_arrangement.py --save <iter_dir> --hf <hf_export_dir>
    python3 save_vs_hf_arrangement.py --save <iter_dir> --hf <hf_export_dir> --dump-keys     # discovery only
    python3 save_vs_hf_arrangement.py --save ... --hf ... --layers 0,3 --expert 0,255 --json report.json

Exit 0 = every checked family matched its canonical arrangement; 1 = a mismatch or a trap arrangement;
2 = usage / IO error; 3 = a family could not be resolved (``--allow-unresolved`` downgrades that to a
warning).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from itertools import permutations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger("save-vs-hf")

PASS, FAIL, UNRESOLVED, SKIP = "PASS", "FAIL", "UNRESOLVED", "SKIP"

_LLM = "language_model.decoder.layers."
_VIS = "vision_model.decoder.layers."


class Unresolvable(Exception):
    """A family's keys are absent on one side; the message carries the siblings that do exist."""


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Layout arithmetic. Pure functions over ints returning index lists, so every layout claim in this file
# is unit-testable without torch, DCP or a checkpoint (tests/unit_tests/models/ov2/test_save_vs_hf_layout.py).
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def qkv_rows_grouped_blocks(nq: int, nk: int, nv: int, groups: int) -> Tuple[List[int], List[int], List[int]]:
    """Row indices of q / k / v inside a fused mcore ``linear_qkv``, given each side's TOTAL row count.

    mcore interleaves by KV group -- each group's queries sit next to their K and V:
    ``[q-block, k-block, v-block][q-block, k-block, v-block]...``. Taking the block sizes from the row
    counts rather than from ``heads * head_dim`` is what makes this work for Qwen3.5's GATED attention,
    where the q side also carries a gate of the same width (q_proj is 2 * heads * head_dim rows).
    """
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    for name, n in (("nq", nq), ("nk", nk), ("nv", nv)):
        if n <= 0 or n % groups:
            raise ValueError(f"{name}={n} is not a positive multiple of groups={groups}")
    qg, kg, vg = nq // groups, nk // groups, nv // groups
    block = qg + kg + vg
    q: List[int] = []
    k: List[int] = []
    v: List[int] = []
    for g in range(groups):
        base = g * block
        q.extend(range(base, base + qg))
        k.extend(range(base + qg, base + qg + kg))
        v.extend(range(base + qg + kg, base + block))
    return q, k, v


def qkv_rows_grouped(num_heads: int, num_kv_heads: int, head_dim: int) -> Tuple[List[int], List[int], List[int]]:
    """The head-count form of :func:`qkv_rows_grouped_blocks` (ungated attention, one gate-free q per head)."""
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"bad geometry: heads={num_heads} kv={num_kv_heads} head_dim={head_dim}")
    if num_heads % num_kv_heads:
        raise ValueError(f"num_heads={num_heads} is not a multiple of num_kv_heads={num_kv_heads}")
    return qkv_rows_grouped_blocks(num_heads * head_dim, num_kv_heads * head_dim,
                                   num_kv_heads * head_dim, num_kv_heads)


def qkv_rows_gated_head_interleaved(nq: int, nk: int, nv: int, groups: int) -> Tuple[List[int], List[int], List[int]]:
    """Row indices when the layer is GATED and the export interleaves q with its gate per head.

    mcore lays a gated group out as ``[q-block, gate-block, k, v]`` (attention.py, ``split_arg_list`` =
    [q_per_group*hn, q_per_group*hn, hn, hn]), while an HF ``q_proj`` that is reshaped to
    ``[..., num_heads, 2*head_dim]`` and chunked on the last dim reads as ``[q_h0, gate_h0, q_h1, ...]``.
    This returns the SAVE rows in that HF order. Geometry is derived from the row counts alone:
    ``head_dim = nk // groups`` and ``heads_per_group = (nq // groups) // 2 // head_dim``.
    """
    if groups <= 0 or nk % groups or nv % groups or nq % groups:
        raise ValueError(f"nq={nq} nk={nk} nv={nv} do not split into groups={groups}")
    hd = nk // groups
    if hd <= 0 or nv // groups != hd:
        raise ValueError(f"k and v must contribute one head per group (nk={nk}, nv={nv}, groups={groups})")
    q_and_gate = nq // groups
    if q_and_gate % (2 * hd):
        raise ValueError(f"q rows per group ({q_and_gate}) is not 2 * head_dim ({hd}) * heads_per_group")
    npg = q_and_gate // (2 * hd)
    block = q_and_gate + 2 * hd
    q: List[int] = []
    k: List[int] = []
    v: List[int] = []
    for g in range(groups):
        base = g * block
        for j in range(npg):
            q.extend(range(base + j * hd, base + (j + 1) * hd))                       # q of head j
            q.extend(range(base + npg * hd + j * hd, base + npg * hd + (j + 1) * hd))  # its gate
        k.extend(range(base + q_and_gate, base + q_and_gate + hd))
        v.extend(range(base + q_and_gate + hd, base + block))
    return q, k, v


def qkv_rows_contiguous(num_heads: int, num_kv_heads: int, head_dim: int) -> Tuple[List[int], List[int], List[int]]:
    """The trap layout: all of q, then all of k, then all of v (what a naive row split produces)."""
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"bad geometry: heads={num_heads} kv={num_kv_heads} head_dim={head_dim}")
    nq, nk = num_heads * head_dim, num_kv_heads * head_dim
    return list(range(nq)), list(range(nq, nq + nk)), list(range(nq + nk, nq + 2 * nk))


def qkv_rows_contiguous_blocks(nq: int, nk: int, nv: int) -> Tuple[List[int], List[int], List[int]]:
    """The trap layout in row-count form: ``[all q][all k][all v]``."""
    for name, n in (("nq", nq), ("nk", nk), ("nv", nv)):
        if n <= 0:
            raise ValueError(f"{name}={n} must be positive")
    return list(range(nq)), list(range(nq, nq + nk)), list(range(nq + nk, nq + nk + nv))


def qkv_rows_head_interleaved(num_heads: int, head_dim: int) -> Tuple[List[int], List[int], List[int]]:
    """Per-head ``[q_i, k_i, v_i]`` triplets -- the MHA case of the grouped layout (the OV2 vision tower)."""
    return qkv_rows_grouped(num_heads, num_heads, head_dim)


def deinterleave_rows(total_rows: int, chunks: int) -> List[int]:
    """Rows that turn a TP-rank-interleaved ``[a_0, b_0, a_1, b_1, ...]`` stack into ``[a_0..a_n, b_0..b_n]``.

    This is the SwiGLU trap in index form: mcore holds ``linear_fc1`` as ``[gate; up]`` PER TP RANK, so a
    merge that simply concatenates rank blocks leaves gate and up interleaved in ``chunks`` pieces. Applying
    these indices to such a tensor restores the contiguous ``[gate; up]`` an HF export must carry -- so if
    the HF tensor equals the SAVE *after* this permutation, the export shipped the interleaved order.
    """
    if chunks <= 0:
        raise ValueError(f"chunks must be positive, got {chunks}")
    if total_rows % (2 * chunks):
        raise ValueError(f"total_rows={total_rows} is not divisible by 2*chunks={2 * chunks}")
    piece = total_rows // (2 * chunks)
    gate = [c * 2 * piece + i for c in range(chunks) for i in range(piece)]
    up = [c * 2 * piece + piece + i for c in range(chunks) for i in range(piece)]
    return gate + up


def swap_halves(total_rows: int) -> List[int]:
    """``[a; b] -> [b; a]`` (catches a fused pair written in the opposite order)."""
    if total_rows % 2:
        raise ValueError(f"total_rows={total_rows} is odd")
    half = total_rows // 2
    return list(range(half, total_rows)) + list(range(half))


def concat_offsets(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """``[(start, stop)]`` for each part of a row-concatenation."""
    out: List[Tuple[int, int]] = []
    cur = 0
    for s in sizes:
        out.append((cur, cur + s))
        cur += s
    return out


def sample_windows(n_rows: int, budget: int) -> List[Tuple[int, int]]:
    """Row windows to compare for a tall tensor: the head AND the tail.

    The tail matters: an off-by-one or a vocab-padding misalignment leaves the first rows identical and
    shifts everything after them, which a head-only sample cannot see.
    """
    if budget <= 0 or n_rows <= budget:
        return [(0, n_rows)]
    half = max(1, budget // 2)
    if n_rows <= 2 * half:
        return [(0, n_rows)]
    return [(0, half), (n_rows - half, n_rows)]


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Readers
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class SaveReader:
    """Reads GLOBAL tensors from a torch_dist (DCP) checkpoint directory, single process, CPU."""

    def __init__(self, iter_dir: Path, max_load_bytes: int):
        from torch.distributed.checkpoint import FileSystemReader

        self.dir = iter_dir
        self.max_load_bytes = max_load_bytes
        self._reader = FileSystemReader(str(iter_dir))
        self.meta = self._reader.read_metadata().state_dict_metadata
        self._cache: Dict[str, Any] = {}

    def keys(self) -> List[str]:
        return sorted(self.meta)

    def tensor_keys(self) -> List[str]:
        return [k for k in self.keys() if self.describe(k) is not None]

    def siblings(self, prefix: str, limit: int = 4) -> List[str]:
        return [k for k in self.keys() if k.startswith(prefix)][:limit]

    def describe(self, key: str) -> Optional[Tuple[Tuple[int, ...], Any]]:
        m = self.meta.get(key)
        size = getattr(m, "size", None)
        if size is None:
            return None
        return tuple(int(x) for x in size), getattr(getattr(m, "properties", None), "dtype", None)

    def nbytes(self, key: str) -> int:
        import torch

        desc = self.describe(key)
        if desc is None:
            return 0
        shape, dtype = desc
        n = 1
        for d in shape:
            n *= int(d)
        width = torch.empty(0, dtype=dtype).element_size() if dtype is not None else 2
        return n * width

    def load(self, key: str):
        """The whole global tensor (cached). DCP fills a plain tensor from every chunk that covers it."""
        if key in self._cache:
            return self._cache[key]
        import torch
        import torch.distributed.checkpoint as dcp

        desc = self.describe(key)
        if desc is None:
            raise KeyError(key)
        shape, dtype = desc
        nbytes = self.nbytes(key)
        if nbytes > self.max_load_bytes:
            raise MemoryError(
                f"{key} is {nbytes / 2**30:.2f} GiB > --max-load-gb ({self.max_load_bytes / 2**30:.2f} GiB); "
                f"raise the budget, or use load_dim0_index for a single slice"
            )
        buf = torch.empty(tuple(shape), dtype=dtype if dtype is not None else torch.float32)
        state = {key: buf}
        try:
            dcp.load(state, storage_reader=self._reader, no_dist=True)
        except TypeError:
            # Older torch has no `no_dist`; a single-process gloo group is the same single reader.
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
            dcp.load(state, storage_reader=self._reader)
        self._cache[key] = state[key]
        return self._cache[key]

    def load_dim0_index(self, key: str, index: int):
        """One dim-0 slice of a stacked tensor, without materialising the whole thing when it is huge."""
        desc = self.describe(key)
        if desc is None:
            raise KeyError(key)
        shape, dtype = desc
        if not shape:
            raise ValueError(f"{key} is a scalar; no dim-0 slice")
        n = int(shape[0])
        if not 0 <= index < n:
            raise IndexError(f"{key}: dim-0 index {index} out of range (0..{n - 1})")
        if self.nbytes(key) <= self.max_load_bytes:
            return self.load(key)[index]
        import torch
        import torch.distributed as dist
        from megatron.core import dist_checkpointing as mdc
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        if not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
        local = [1] + [int(d) for d in shape[1:]]
        buf = torch.empty(tuple(local), dtype=dtype if dtype is not None else torch.float32)
        st = ShardedTensor.from_rank_offsets(key, buf, (0, index, n), replica_id=0)
        mdc.load({key: st}, str(self.dir), validate_access_integrity=False)
        return st.data[0]


class HFReader:
    """Reads tensors (or slices) out of an HF export's safetensors shards."""

    def __init__(self, root: Path):
        from safetensors import safe_open

        self.root = root
        index = root / "model.safetensors.index.json"
        if index.is_file():
            with index.open() as fh:
                self.weight_map: Dict[str, str] = json.load(fh)["weight_map"]
        elif (root / "model.safetensors").is_file():
            with safe_open(str(root / "model.safetensors"), framework="pt") as fh:
                self.weight_map = {k: "model.safetensors" for k in fh.keys()}
        else:
            raise FileNotFoundError(f"no safetensors (index or single file) under {root}")

    def keys(self) -> List[str]:
        return sorted(self.weight_map)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def siblings(self, prefix: str, limit: int = 4) -> List[str]:
        return [k for k in self.keys() if k.startswith(prefix)][:limit]

    def shape(self, key: str) -> Tuple[int, ...]:
        from safetensors import safe_open

        with safe_open(str(self.root / self.weight_map[key]), framework="pt") as fh:
            return tuple(fh.get_slice(key).get_shape())

    def load(self, key: str, dim0_slice: Optional[Tuple[int, int]] = None, index0: Optional[int] = None):
        """Full tensor, rows ``[a, b)``, or element ``index0`` of dim 0 (packed ``[E, ...]`` experts)."""
        from safetensors import safe_open

        if key not in self.weight_map:
            raise KeyError(key)
        with safe_open(str(self.root / self.weight_map[key]), framework="pt") as fh:
            if index0 is not None:
                return fh.get_slice(key)[index0]
            if dim0_slice is not None:
                return fh.get_slice(key)[dim0_slice[0] : dim0_slice[1]]
            return fh.get_tensor(key)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def compare(a, b, atol: float, rtol: float) -> Tuple[bool, float]:
    """(match, max_abs_diff). Exact by default; NaN-vs-NaN counts as equal, a lone NaN as infinite."""
    import torch

    if tuple(a.shape) != tuple(b.shape):
        return False, float("inf")
    if a.dtype != b.dtype:
        a, b = a.float(), b.float()
    exact = atol == 0.0 and rtol == 0.0
    if exact:
        eq = torch.eq(a, b)
        if a.is_floating_point():
            eq = eq | (torch.isnan(a) & torch.isnan(b))
        if bool(eq.all()):
            return True, 0.0
    fa, fb = a.float(), b.float()
    diff = (fa - fb).abs()
    both_nan = torch.isnan(fa) & torch.isnan(fb)
    one_nan = torch.isnan(fa) ^ torch.isnan(fb)
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)
    diff = torch.where(one_nan, torch.full_like(diff, float("inf")), diff)
    match = False if exact else bool(torch.isclose(fa, fb, atol=atol, rtol=rtol, equal_nan=True).all())
    return match, (float(diff.max()) if diff.numel() else 0.0)


def explain_rows(save_tensor, hf_tensor, max_runs: int = 12) -> str:
    """Say where each HF row actually came from, by matching rows byte-for-byte against the SAVE.

    When no declared arrangement matches, guessing further is a waste of a push cycle: this reads the
    answer off the data instead. Rows are hashed (exact bytes, so bf16 is compared as stored), the HF ->
    SAVE row map is recovered, and consecutive stretches are compressed into runs. Rows that match nothing
    in the SAVE are reported separately -- those are the ones that cannot be a re-arrangement at all.
    """
    import hashlib

    import torch

    if save_tensor.ndim != 2 or hf_tensor.ndim != 2 or save_tensor.shape[1] != hf_tensor.shape[1]:
        return ""

    def row_hashes(t):
        flat = t.contiguous().view(torch.uint8).numpy()
        return [hashlib.blake2b(flat[i].tobytes(), digest_size=16).digest() for i in range(flat.shape[0])]

    index: Dict[bytes, int] = {}
    duplicates = 0
    for i, h in enumerate(row_hashes(save_tensor)):
        if h in index:
            duplicates += 1
        else:
            index[h] = i
    mapping = [index.get(h, -1) for h in row_hashes(hf_tensor)]
    unmatched = sum(1 for m in mapping if m < 0)

    runs: List[Tuple[int, int, int]] = []  # (hf_start, save_start, length)
    for i, m in enumerate(mapping):
        if runs and m >= 0 and runs[-1][1] >= 0 and m == runs[-1][1] + runs[-1][2] and \
                i == runs[-1][0] + runs[-1][2]:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        elif runs and m < 0 and runs[-1][1] < 0 and i == runs[-1][0] + runs[-1][2]:
            runs[-1] = (runs[-1][0], -1, runs[-1][2] + 1)
        else:
            runs.append((i, m, 1))
    shown = "; ".join(
        f"HF[{a}:{a + n}]={'SAVE[%d:%d]' % (b, b + n) if b >= 0 else 'NO MATCH'}" for a, b, n in runs[:max_runs])
    more = f" (+{len(runs) - max_runs} more runs)" if len(runs) > max_runs else ""
    extra = f"; {unmatched} HF rows match no SAVE row" if unmatched else ""
    dup = f"; {duplicates} duplicate SAVE rows (mapping may be ambiguous)" if duplicates else ""
    return f"row map: {shown}{more}{extra}{dup}"


class Result:
    """One family's verdict: the status, the arrangement that matched, and the keys behind it."""

    def __init__(self, family: str, status: str, arrangement: str = "", detail: str = "",
                 save_keys: Sequence[str] = (), hf_keys: Sequence[str] = (),
                 max_abs_diff: Optional[float] = None):
        self.family = family
        self.status = status
        self.arrangement = arrangement
        self.detail = detail
        self.save_keys = list(save_keys)
        self.hf_keys = list(hf_keys)
        self.max_abs_diff = max_abs_diff

    def as_dict(self) -> dict:
        return {
            "family": self.family, "status": self.status, "arrangement": self.arrangement,
            "detail": self.detail, "save_keys": self.save_keys, "hf_keys": self.hf_keys,
            "max_abs_diff": self.max_abs_diff,
        }


class Checker:
    """Runs one family at a time; any exception becomes an UNRESOLVED row instead of killing the run."""

    def __init__(self, save: SaveReader, hf: HFReader, atol: float, rtol: float, rows: int,
                 explain: bool = True, explain_max_rows: int = 65536):
        self.save = save
        self.hf = hf
        self.atol = atol
        self.rtol = rtol
        self.rows = rows
        self.explain = explain
        self.explain_max_rows = explain_max_rows
        self.results: List[Result] = []

    def run(self, family: str, fn: Callable[[], Result]) -> None:
        try:
            self.results.append(fn())
        except Unresolvable as exc:
            self.results.append(Result(family, UNRESOLVED, detail=str(exc)))
        except KeyError as exc:
            self.results.append(Result(family, UNRESOLVED, detail=f"missing key {exc}"))
        except MemoryError as exc:
            self.results.append(Result(family, UNRESOLVED, detail=str(exc)))
        except Exception as exc:  # noqa: BLE001 -- one broken family must not hide the others
            self.results.append(Result(family, UNRESOLVED, detail=f"{type(exc).__name__}: {exc}"))

    # -- key resolution ----------------------------------------------------------------------------
    def pick_save(self, candidates: Sequence[str], *, diag_prefix: str = "") -> str:
        for key in candidates:
            if self.save.describe(key) is not None:
                return key
        near = self.save.siblings(diag_prefix) if diag_prefix else []
        raise Unresolvable(f"no SAVE key among {list(candidates)[:3]}"
                           + (f"; SAVE has {near}" if near else ""))

    def pick_hf(self, candidates: Sequence[str], *, diag_prefix: str = "") -> str:
        for key in candidates:
            if self.hf.has(key):
                return key
        near = self.hf.siblings(diag_prefix) if diag_prefix else []
        raise Unresolvable(f"no HF key among {list(candidates)[:3]}"
                           + (f"; HF has {near}" if near else ""))

    def pick_hf_group(self, groups: Sequence[Sequence[str]], *, diag_prefix: str = "") -> List[str]:
        """First group whose every key exists (e.g. a packed tensor, else a gate/up pair)."""
        for group in groups:
            if all(self.hf.has(k) for k in group):
                return list(group)
        near = self.hf.siblings(diag_prefix) if diag_prefix else []
        raise Unresolvable(f"no HF key group among {[list(g)[:2] for g in groups][:3]}"
                           + (f"; HF has {near}" if near else ""))

    # -- verdict -----------------------------------------------------------------------------------
    def _verdict(self, family: str, save_keys: Sequence[str], hf_keys: Sequence[str],
                 candidates: Sequence[Tuple[str, bool, Any, Any]],
                 explain_pair: Optional[Tuple[Any, Any]] = None) -> Result:
        """``candidates`` = (name, is_canonical, save_side_tensor, hf_side_tensor); canonical ones first."""
        if not candidates:
            return Result(family, UNRESOLVED, detail="no comparable arrangement could be built",
                          save_keys=save_keys, hf_keys=hf_keys)
        best: Optional[float] = None
        for name, canonical, sv, hv in candidates:
            ok, diff = compare(sv, hv, self.atol, self.rtol)
            best = diff if best is None else min(best, diff)
            if ok:
                if canonical:
                    return Result(family, PASS, name, "matches the canonical arrangement",
                                  save_keys, hf_keys, 0.0)
                return Result(family, FAIL, name,
                              f"HF equals the NON-canonical arrangement '{name}': the export re-laid this "
                              f"family out the wrong way", save_keys, hf_keys, 0.0)
        detail = (f"no known arrangement reproduces the HF tensor from the SAVE "
                  f"(smallest max|diff| = {best:.6g}); the values themselves differ")
        if self.explain and explain_pair is not None:
            sv_raw, hv_raw = explain_pair
            if int(sv_raw.shape[0]) <= self.explain_max_rows and int(hv_raw.shape[0]) <= self.explain_max_rows:
                try:
                    told = explain_rows(sv_raw, hv_raw)
                except Exception as exc:  # noqa: BLE001 -- diagnostics must never mask the verdict
                    told = f"row map unavailable ({type(exc).__name__}: {exc})"
                if told:
                    detail += "; " + told
        return Result(family, FAIL, "none", detail, save_keys, hf_keys, best)

    # -- 1:1 families ------------------------------------------------------------------------------
    def identity_family(self, family: str, save_key: str, hf_key: str, *, plus_one: bool = False,
                        transpose_ok: bool = True) -> Result:
        """1:1 tensors. ``plus_one``: canonical is HF == SAVE + 1 (mcore zero-centred RMSNorm -> HF RMSNorm).

        Tall tensors are sampled head+tail (``--rows``); the SAVE is always trimmed to the export's row
        count first, because vocab-parallel padding legitimately makes the SAVE taller.
        """
        import torch

        if not self.hf.has(hf_key):
            raise KeyError(hf_key)
        hf_shape = self.hf.shape(hf_key)
        sv_full = self.save.load(save_key)
        if sv_full.ndim == 0 or not hf_shape:
            sv, hv, tag, whole = sv_full, self.hf.load(hf_key), "", True
        else:
            hf_rows = int(hf_shape[0])
            windows = sample_windows(hf_rows, self.rows)
            sv = torch.cat([sv_full[a:b] for a, b in windows], dim=0)
            sampled = windows != [(0, hf_rows)]
            hv = torch.cat([self.hf.load(hf_key, (a, b)) for a, b in windows], dim=0) if sampled \
                else self.hf.load(hf_key)
            tag = f"[rows {'+'.join(f'{a}:{b}' for a, b in windows)}]" if sampled else ""
            whole = not sampled and int(sv_full.shape[0]) == hf_rows
        cands: List[Tuple[str, bool, Any, Any]] = []
        if plus_one:
            cands.append((f"zero_centred_plus_one{tag}", True, (sv.float() + 1.0).to(hv.dtype), hv))
            cands.append((f"identity_no_offset{tag}", False, sv, hv))
        else:
            cands.append((f"identity{tag}", True, sv, hv))
            if sv.is_floating_point():
                cands.append((f"plus_one_offset{tag}", False, (sv.float() + 1.0).to(hv.dtype), hv))
        if transpose_ok and whole and sv.ndim == 2:
            cands.append(("transposed", False, sv.t().contiguous(), hv))
        result = self._verdict(family, [save_key], [hf_key], cands)
        if result.status == PASS and sv_full.ndim >= 1 and hf_shape and sv_full.shape[0] > hf_shape[0]:
            result.detail += (f"; SAVE has {int(sv_full.shape[0]) - int(hf_shape[0])} extra dim-0 rows "
                              f"(vocab padding) that the export legitimately drops")
        return result

    def alternates_family(self, family: str, save_key: str, hf_alternates: Sequence[str], *,
                          plus_one: bool = False, transpose_ok: bool = True,
                          diag_prefix: str = "") -> Result:
        """:meth:`identity_family` over a list of possible HF spellings."""
        hf_key = self.pick_hf(hf_alternates, diag_prefix=diag_prefix)
        return self.identity_family(family, save_key, hf_key, plus_one=plus_one, transpose_ok=transpose_ok)

    # -- fused / packed families -------------------------------------------------------------------
    def qkv_family(self, family: str, save_key: str, hf_q: str, hf_k: str, hf_v: str, groups: int,
                   ungated_q_rows: int = 0) -> Result:
        """Fused mcore ``linear_qkv`` vs the export's q / k / v.

        Block sizes come from the HF tensors themselves. Gating is detected by comparing the HF q rows with
        ``heads * head_dim``: mcore lays a gated group out as ``[q-block, gate-block, k, v]`` while an HF
        ``q_proj`` reshaped to ``[..., heads, 2*head_dim]`` reads as ``[q_h0, gate_h0, q_h1, ...]``, so the
        two differ by a within-group permutation. Both that arrangement and the un-permuted one are
        candidates; only the first is canonical, and the other is named rather than left unexplained.
        """
        import torch

        sv = self.save.load(save_key)
        q, k, v = self.hf.load(hf_q), self.hf.load(hf_k), self.hf.load(hf_v)
        nq, nk, nv = int(q.shape[0]), int(k.shape[0]), int(v.shape[0])
        hv = torch.cat([q, k, v], dim=0)
        if nq + nk + nv != int(sv.shape[0]):
            return Result(family, UNRESOLVED,
                          detail=(f"HF q/k/v rows {nq}+{nk}+{nv}={nq + nk + nv} do not add up to the fused "
                                  f"SAVE rows {int(sv.shape[0])}"),
                          save_keys=[save_key], hf_keys=[hf_q, hf_k, hf_v])
        gated = ungated_q_rows > 0 and nq == 2 * ungated_q_rows
        plan: List[Tuple[str, bool, Callable[[], Tuple[List[int], List[int], List[int]]]]] = []
        if gated:
            plan.append(("gated_per_head_q_gate", True,
                         lambda: qkv_rows_gated_head_interleaved(nq, nk, nv, groups)))
            plan.append(("gated_per_group_q_then_gate", False,
                         lambda: qkv_rows_grouped_blocks(nq, nk, nv, groups)))
        else:
            plan.append(("gqa_group_interleaved", True, lambda: qkv_rows_grouped_blocks(nq, nk, nv, groups)))
            if ungated_q_rows and nq != ungated_q_rows:
                plan.append(("gated_per_head_q_gate", False,
                             lambda: qkv_rows_gated_head_interleaved(nq, nk, nv, groups)))
        plan.append(("contiguous_q_k_v", False, lambda: qkv_rows_contiguous_blocks(nq, nk, nv)))

        cands: List[Tuple[str, bool, Any, Any]] = []
        for name, canonical, build in plan:
            try:
                idx = build()
            except ValueError:
                continue
            order = idx[0] + idx[1] + idx[2]
            if len(order) != int(sv.shape[0]):
                continue
            cands.append((name, canonical, sv[torch.as_tensor(order, dtype=torch.long)], hv))
        if not cands:
            return Result(family, UNRESOLVED,
                          detail=(f"no candidate layout fits rows {int(sv.shape[0])} with "
                                  f"q/k/v {nq}/{nk}/{nv} and groups={groups}"),
                          save_keys=[save_key], hf_keys=[hf_q, hf_k, hf_v])
        return self._verdict(family, [save_key], [hf_q, hf_k, hf_v], cands, explain_pair=(sv, hv))

    def fused_qkv_family(self, family: str, save_key: str, hf_key: str, num_heads: int, head_dim: int) -> Result:
        """OV2 vision tower: HF keeps ONE fused ``self_attn.qkv`` ([3d, d]); mcore interleaves per head."""
        import torch

        sv = self.save.load(save_key)
        hv = self.hf.load(hf_key)
        cands: List[Tuple[str, bool, Any, Any]] = []
        q, k, v = qkv_rows_head_interleaved(num_heads, head_dim)
        order = q + k + v
        if len(order) == int(sv.shape[0]):
            cands.append(("head_interleaved_to_contiguous", True,
                          sv[torch.as_tensor(order, dtype=torch.long)], hv))
        cands.append(("identity_no_relayout", False, sv, hv))
        return self._verdict(family, [save_key], [hf_key], cands)

    def gated_pair_family(self, family: str, save_key: str, hf_keys: Sequence[str], *,
                          hf_index0: Optional[int] = None, save_index0: Optional[int] = None,
                          tp_candidates: Sequence[int] = (2, 4, 8)) -> Result:
        """Fused gate/up matrices (routed experts, shared expert, dense MLP).

        ``hf_keys`` is either one packed tensor or a (gate, up) pair that is concatenated first. Canonical:
        the HF side equals the SAVE's global ``[gate; up]`` rows (transposed when the shapes say so). Traps:
        gate and up interleaved in TP-rank blocks -- what a merge that concatenated ``linear_fc1`` rank
        slices without honouring the swiglu split produces -- or the two halves swapped.
        """
        import torch

        sv = self.save.load_dim0_index(save_key, save_index0) if save_index0 is not None else self.save.load(save_key)
        if len(hf_keys) == 1:
            hv = self.hf.load(hf_keys[0], index0=hf_index0)
        else:
            hv = torch.cat([self.hf.load(k, index0=hf_index0) for k in hf_keys], dim=0)
        needs_t = sv.ndim == 2 and tuple(sv.shape) == tuple(reversed(tuple(hv.shape))) and sv.shape[0] != sv.shape[1]

        def shaped(t):
            return t.t().contiguous() if needs_t else t

        cands: List[Tuple[str, bool, Any, Any]] = [
            ("transposed_to_hf_layout" if needs_t else "identity", True, shaped(sv), hv)
        ]
        rows = int(sv.shape[0])
        for tp in tp_candidates:
            if rows % (2 * tp):
                continue
            idx = torch.as_tensor(deinterleave_rows(rows, tp), dtype=torch.long)
            cands.append((f"tp{tp}_rank_interleaved_gate_up", False, shaped(sv[idx]), hv))
        if rows % 2 == 0:
            idx = torch.as_tensor(swap_halves(rows), dtype=torch.long)
            cands.append(("halves_swapped", False, shaped(sv[idx]), hv))
        return self._verdict(family, [save_key], list(hf_keys), cands, explain_pair=(sv, hv))

    def matrix_family(self, family: str, save_key: str, hf_key: str, *, hf_index0: Optional[int] = None,
                      save_index0: Optional[int] = None) -> Result:
        """A plain matrix that may be stored per-expert on one side and packed on the other."""
        sv = self.save.load_dim0_index(save_key, save_index0) if save_index0 is not None else self.save.load(save_key)
        hv = self.hf.load(hf_key, index0=hf_index0)
        cands: List[Tuple[str, bool, Any, Any]] = []
        if tuple(sv.shape) == tuple(hv.shape):
            cands.append(("identity", True, sv, hv))
        if sv.ndim == 2 and tuple(sv.shape) == tuple(reversed(tuple(hv.shape))):
            cands.append(("transposed_to_hf_layout", True, sv.t().contiguous(), hv))
        if not cands:
            return Result(family, UNRESOLVED,
                          detail=f"SAVE{tuple(sv.shape)} vs HF{tuple(hv.shape)} are not shape-compatible",
                          save_keys=[save_key], hf_keys=[hf_key])
        return self._verdict(family, [save_key], [hf_key], cands, explain_pair=(sv, hv))

    def hf_concat_family(self, family: str, save_key: str, hf_keys: Sequence[str]) -> Result:
        """One fused SAVE matrix vs several HF matrices concatenated in a declared order."""
        import torch

        sv = self.save.load(save_key)
        parts = {k: self.hf.load(k) for k in hf_keys}
        sizes = [int(parts[k].shape[0]) for k in hf_keys]
        if sum(sizes) != int(sv.shape[0]):
            return Result(family, UNRESOLVED,
                          detail=(f"HF parts sum to {sum(sizes)} rows {concat_offsets(sizes)}, SAVE has "
                                  f"{int(sv.shape[0])}"),
                          save_keys=[save_key], hf_keys=list(hf_keys))
        cands: List[Tuple[str, bool, Any, Any]] = []
        for order in permutations(range(len(hf_keys))):
            names = [hf_keys[i] for i in order]
            label = "+".join(n.rsplit(".", 2)[-2] for n in names)
            cands.append((f"concat[{label}]", order == tuple(range(len(hf_keys))), sv,
                          torch.cat([parts[n] for n in names], dim=0)))
        cands.sort(key=lambda c: not c[1])
        return self._verdict(family, [save_key], list(hf_keys), cands,
                             explain_pair=(sv, torch.cat([parts[k] for k in hf_keys], dim=0)))

    def save_concat_family(self, family: str, save_keys: Sequence[str], hf_key: str) -> Result:
        """Several SAVE parts vs ONE fused HF tensor -- this build splits GDN in_proj / conv1d per role.

        The canonical order is the declared one; every other ordering of the same parts is a trap candidate,
        so a query/key swap is named instead of showing up as an unexplained mismatch.
        """
        import torch

        parts = {k: self.save.load(k) for k in save_keys}
        hv = self.hf.load(hf_key)
        sizes = [int(parts[k].shape[0]) for k in save_keys]
        if sum(sizes) != int(hv.shape[0]):
            return Result(family, UNRESOLVED,
                          detail=(f"SAVE parts sum to {sum(sizes)} rows {concat_offsets(sizes)}, HF has "
                                  f"{int(hv.shape[0])}"),
                          save_keys=list(save_keys), hf_keys=[hf_key])
        cands: List[Tuple[str, bool, Any, Any]] = []
        limit = 6 if len(save_keys) <= 3 else 2  # 3 parts -> all 6 orders; more -> canonical + reversed
        orders = list(permutations(range(len(save_keys))))[:limit] if len(save_keys) <= 3 else [
            tuple(range(len(save_keys))), tuple(reversed(range(len(save_keys))))]
        for order in orders:
            names = [save_keys[i] for i in order]
            label = "+".join(n.rsplit(".", 1)[-1] for n in names)
            cands.append((f"concat[{label}]", order == tuple(range(len(save_keys))),
                          torch.cat([parts[n] for n in names], dim=0), hv))
        cands.sort(key=lambda c: not c[1])
        return self._verdict(family, list(save_keys), [hf_key], cands,
                             explain_pair=(torch.cat([parts[k] for k in save_keys], dim=0), hv))

    def conv1d_family(self, family: str, save_keys: Sequence[str], hf_key: str) -> Result:
        """GDN depthwise conv: possibly split per role on the SAVE side, possibly squeezed on the HF side."""
        import torch

        hv = self.hf.load(hf_key)
        parts = [self.save.load(k) for k in save_keys]
        sv = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        cands: List[Tuple[str, bool, Any, Any]] = []
        label = "+".join(k.rsplit(".", 1)[-1] for k in save_keys) if len(save_keys) > 1 else "identity"
        if tuple(sv.shape) == tuple(hv.shape):
            cands.append((f"concat[{label}]" if len(save_keys) > 1 else "identity", True, sv, hv))
        if sv.ndim == 3 and hv.ndim == 2 and sv.shape[1] == 1:
            cands.append(("squeezed_channel_dim", True, sv.squeeze(1), hv))
        if sv.ndim == 2 and hv.ndim == 3 and hv.shape[1] == 1:
            cands.append(("unsqueezed_channel_dim", True, sv.unsqueeze(1), hv))
        if len(save_keys) == 3:
            rev = torch.cat(list(reversed(parts)), dim=0)
            if tuple(rev.shape) == tuple(hv.shape):
                cands.append(("concat[reversed]", False, rev, hv))
        if not cands:
            return Result(family, UNRESOLVED, detail=f"SAVE{tuple(sv.shape)} vs HF{tuple(hv.shape)}",
                          save_keys=list(save_keys), hf_keys=[hf_key])
        return self._verdict(family, list(save_keys), [hf_key], cands)

    def absent_family(self, family: str, save_key: str, hf_key: str) -> Result:
        """The tensor must be present on both sides or on neither (the merged line's vision final LN)."""
        in_save = self.save.describe(save_key) is not None
        in_hf = self.hf.has(hf_key)
        if in_save == in_hf:
            return Result(family, SKIP if not in_save else PASS,
                          "present_on_both" if in_save else "absent_on_both",
                          "the export carries this tensor exactly when the SAVE does", [save_key], [hf_key])
        return Result(family, FAIL, "present" if in_hf else "missing",
                      f"SAVE has it: {in_save}; HF has it: {in_hf} -- the export must carry this tensor if "
                      f"and only if the SAVE does", [save_key], [hf_key])


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Discovery -- what the SAVE actually contains; never assumed from the config
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _layer_suffixes(keys: Sequence[str], prefix: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    pat = re.compile(re.escape(prefix) + r"(\d+)\.(.+)$")
    for k in keys:
        m = pat.match(k)
        if m:
            out.setdefault(int(m.group(1)), []).append(m.group(2))
    return out


def _parts_of(suffixes: Sequence[str], stem: str) -> List[str]:
    """Named sub-tensors of ``stem`` (``in_proj.weight.query`` -> ``query``), sorted, excluding extra state."""
    pat = re.compile(re.escape(stem) + r"\.([A-Za-z_]+)$")
    return sorted({m.group(1) for m in (pat.match(s) for s in suffixes) if m})


def discover(save_keys: Sequence[str]) -> dict:
    """Which layers are attention / GDN / MoE, how experts and GDN projections are stored, what exists."""
    llm = _layer_suffixes(save_keys, _LLM)
    vis = _layer_suffixes(save_keys, _VIS)
    attention, gdn, moe = [], [], []
    expert_stem: Optional[str] = None
    expert_ids: List[int] = []
    expert_indexed = False
    in_proj_parts: List[str] = []
    conv1d_parts: List[str] = []
    in_proj_fused = False
    conv1d_fused = False
    shared_stem: Optional[str] = None
    for layer, suffixes in sorted(llm.items()):
        sset = set(suffixes)
        if "self_attention.linear_qkv.weight" in sset:
            attention.append(layer)
        if any(s.startswith("self_attention.in_proj.weight") for s in sset):
            gdn.append(layer)
            in_proj_parts = in_proj_parts or _parts_of(suffixes, "self_attention.in_proj.weight")
            conv1d_parts = conv1d_parts or _parts_of(suffixes, "self_attention.conv1d.weight")
            in_proj_fused = in_proj_fused or ("self_attention.in_proj.weight" in sset)
            conv1d_fused = conv1d_fused or ("self_attention.conv1d.weight" in sset)
        if "mlp.router.weight" in sset:
            moe.append(layer)
        for s in suffixes:
            m = re.match(r"(mlp\.experts\.(?:experts\.)?(?:local_experts\.(\d+)\.)?linear_fc1)\.weight(\d*)$", s)
            if m:
                expert_stem = expert_stem or m.group(1).replace(f"local_experts.{m.group(2)}.", "local_experts.{e}.")
                if m.group(3):
                    expert_indexed = True
                    expert_ids.append(int(m.group(3)))
                elif m.group(2):
                    expert_ids.append(int(m.group(2)))
            if shared_stem is None and s.startswith("mlp.shared_experts."):
                shared_stem = "mlp.shared_experts."
    return {
        "attention_layers": attention,
        "gdn_layers": gdn,
        "moe_layers": moe,
        "vision_layers": sorted(vis),
        "expert_stem": expert_stem,
        "expert_indexed": expert_indexed,
        "expert_ids": sorted(set(expert_ids))[:4],
        "expert_count": len(set(expert_ids)),
        "gdn_in_proj_parts": in_proj_parts,
        "gdn_in_proj_fused": in_proj_fused,
        "gdn_conv1d_parts": conv1d_parts,
        "gdn_conv1d_fused": conv1d_fused,
        "shared_expert_stem": shared_stem,
        "has_mtp": any(".mtp." in k for k in save_keys),
        "has_vision_final_ln": any(k.startswith("vision_model.decoder.final_layernorm.") for k in save_keys),
    }


def expert_save_key(layer: int, expert: int, stem: Optional[str], indexed: bool) -> Tuple[str, Optional[int]]:
    """(SAVE key for this expert's fc1 stem, dim-0 index) for the discovered expert storage style."""
    stem = stem or "mlp.experts.linear_fc1"
    body = stem.replace("{e}", str(expert))
    if indexed:
        return f"{_LLM}{layer}.{body}.weight{expert}", None
    if "local_experts" in stem:
        return f"{_LLM}{layer}.{body}.weight", None
    return f"{_LLM}{layer}.{body}.weight", expert


def hf_dims(root: Path) -> dict:
    """Geometry from the export's own config.json (composite -> text_config / vision_config)."""
    with (root / "config.json").open() as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config", cfg)
    vision = cfg.get("vision_config", {}) or {}
    hidden = int(text.get("hidden_size", 0) or 0)
    heads = int(text.get("num_attention_heads", 0) or 0)
    kv = int(text.get("num_key_value_heads", heads or 1) or 1)
    head_dim = int(text.get("head_dim", (hidden // heads) if heads else 0) or 0)
    v_hidden = int(vision.get("hidden_size", 0) or 0)
    v_heads = int(vision.get("num_attention_heads", vision.get("num_heads", 0)) or 0)
    return {
        "hidden": hidden, "num_heads": heads, "num_kv_heads": kv, "head_dim": head_dim,
        "num_experts": int(text.get("num_experts", 0) or 0),
        "vision_hidden": v_hidden, "vision_heads": v_heads,
        "vision_head_dim": (v_hidden // v_heads) if v_heads else 0,
        "mtp_layers": int(text.get("mtp_num_hidden_layers", 0) or 0),
        "vision_post_layernorm": bool(vision.get("use_post_layernorm", False)),
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def build_and_run(checker: Checker, found: dict, dims: dict, layers: Sequence[int], experts: Sequence[int],
                  vision_layers: Sequence[int]) -> None:
    """Queue every family for the requested layers/experts; each one is independent and self-reporting."""
    hf_llm = "model.language_model."
    hfl = hf_llm + "layers."
    hv = "model.visual."
    c = checker

    c.run("embed_tokens", lambda: c.identity_family(
        "embed_tokens", "language_model.embedding.word_embeddings.weight", hf_llm + "embed_tokens.weight",
        transpose_ok=False))
    c.run("lm_head", lambda: c.identity_family(
        "lm_head", "language_model.output_layer.weight", "lm_head.weight", transpose_ok=False))
    c.run("final_norm", lambda: c.identity_family(
        "final_norm", "language_model.decoder.final_layernorm.weight", hf_llm + "norm.weight"))

    for L in layers:
        sa = f"{_LLM}{L}.self_attention."
        if L in found["attention_layers"]:
            c.run(f"attn{L}_qkv", lambda L=L, sa=sa: c.qkv_family(
                f"attn{L}_qkv", sa + "linear_qkv.weight",
                c.pick_hf([f"{hfl}{L}.self_attn.q_proj.weight"], diag_prefix=f"{hfl}{L}.self_attn."),
                f"{hfl}{L}.self_attn.k_proj.weight", f"{hfl}{L}.self_attn.v_proj.weight",
                dims["num_kv_heads"] or 1, dims["num_heads"] * dims["head_dim"]))
            c.run(f"attn{L}_o_proj", lambda L=L, sa=sa: c.identity_family(
                f"attn{L}_o_proj", sa + "linear_proj.weight", f"{hfl}{L}.self_attn.o_proj.weight"))
            c.run(f"attn{L}_q_norm", lambda L=L, sa=sa: c.identity_family(
                f"attn{L}_q_norm", sa + "q_layernorm.weight", f"{hfl}{L}.self_attn.q_norm.weight"))
            c.run(f"attn{L}_k_norm", lambda L=L, sa=sa: c.identity_family(
                f"attn{L}_k_norm", sa + "k_layernorm.weight", f"{hfl}{L}.self_attn.k_norm.weight"))
            c.run(f"attn{L}_input_ln", lambda L=L, sa=sa: c.identity_family(
                f"attn{L}_input_ln", sa + "linear_qkv.layer_norm_weight", f"{hfl}{L}.input_layernorm.weight"))

        if L in found["gdn_layers"]:
            la = f"{hfl}{L}.linear_attn."
            parts = found["gdn_in_proj_parts"]
            # This build splits the GDN input projection per role; older ones keep one fused matrix.
            if parts:
                qkv_parts = [p for p in ("query", "key", "value") if p in parts]
                if len(qkv_parts) == 3:
                    c.run(f"gdn{L}_in_proj_qkv", lambda L=L, sa=sa, qp=qkv_parts: c.save_concat_family(
                        f"gdn{L}_in_proj_qkv", [sa + f"in_proj.weight.{p}" for p in qp],
                        c.pick_hf([f"{hfl}{L}.linear_attn.in_proj_qkv.weight",
                                   f"{hfl}{L}.linear_attn.in_proj_qkvz.weight"], diag_prefix=la)))
                for part, hf_names in (("z", ["in_proj_z.weight"]),
                                       ("beta", ["in_proj_b.weight", "in_proj_ba.weight"]),
                                       ("alpha", ["in_proj_a.weight"])):
                    if part not in parts:
                        continue
                    c.run(f"gdn{L}_in_proj_{part}", lambda L=L, sa=sa, part=part, hn=hf_names:
                          c.alternates_family(f"gdn{L}_in_proj_{part}", sa + f"in_proj.weight.{part}",
                                              [f"{hfl}{L}.linear_attn.{n}" for n in hn], diag_prefix=la))
            elif found["gdn_in_proj_fused"]:
                c.run(f"gdn{L}_in_proj", lambda L=L, sa=sa: c.hf_concat_family(
                    f"gdn{L}_in_proj", sa + "in_proj.weight",
                    [f"{hfl}{L}.linear_attn.in_proj_qkv.weight", f"{hfl}{L}.linear_attn.in_proj_z.weight",
                     f"{hfl}{L}.linear_attn.in_proj_b.weight", f"{hfl}{L}.linear_attn.in_proj_a.weight"]))
            conv_parts = [p for p in ("query", "key", "value") if p in found["gdn_conv1d_parts"]]
            conv_keys = [sa + f"conv1d.weight.{p}" for p in conv_parts] if conv_parts else [sa + "conv1d.weight"]
            c.run(f"gdn{L}_conv1d", lambda L=L, ck=conv_keys: c.conv1d_family(
                f"gdn{L}_conv1d", ck, f"{hfl}{L}.linear_attn.conv1d.weight"))
            # mcore keeps a zero-centred RMSNorm here while the HF class expects the standard one -> +1.
            c.run(f"gdn{L}_out_norm", lambda L=L, sa=sa: c.identity_family(
                f"gdn{L}_out_norm", sa + "out_norm.weight", f"{hfl}{L}.linear_attn.norm.weight", plus_one=True))
            c.run(f"gdn{L}_out_proj", lambda L=L, sa=sa: c.identity_family(
                f"gdn{L}_out_proj", sa + "out_proj.weight", f"{hfl}{L}.linear_attn.out_proj.weight"))
            c.run(f"gdn{L}_A_log", lambda L=L, sa=sa: c.identity_family(
                f"gdn{L}_A_log", sa + "A_log", f"{hfl}{L}.linear_attn.A_log"))
            c.run(f"gdn{L}_dt_bias", lambda L=L, sa=sa: c.identity_family(
                f"gdn{L}_dt_bias", sa + "dt_bias", f"{hfl}{L}.linear_attn.dt_bias"))
            c.run(f"gdn{L}_input_ln", lambda L=L, sa=sa: c.identity_family(
                f"gdn{L}_input_ln", sa + "in_proj.layer_norm_weight", f"{hfl}{L}.input_layernorm.weight"))

        if L in found["moe_layers"]:
            c.run(f"moe{L}_router", lambda L=L: c.identity_family(
                f"moe{L}_router", f"{_LLM}{L}.mlp.router.weight", f"{hfl}{L}.mlp.gate.weight"))
            c.run(f"moe{L}_pre_mlp_ln", lambda L=L: c.identity_family(
                f"moe{L}_pre_mlp_ln", f"{_LLM}{L}.pre_mlp_layernorm.weight",
                f"{hfl}{L}.post_attention_layernorm.weight"))
            ex = f"{hfl}{L}.mlp.experts"
            for e in experts:
                fc1, idx = expert_save_key(L, e, found["expert_stem"], found["expert_indexed"])
                fc2 = fc1.replace("linear_fc1", "linear_fc2")
                c.run(f"moe{L}_e{e}_gate_up", lambda L=L, e=e, fc1=fc1, idx=idx, ex=ex: c.gated_pair_family(
                    f"moe{L}_e{e}_gate_up", fc1,
                    c.pick_hf_group([[f"{ex}.gate_up_proj"], [f"{ex}.gate_up_proj.weight"],
                                     [f"{ex}.{e}.gate_up_proj.weight"],
                                     [f"{ex}.{e}.gate_proj.weight", f"{ex}.{e}.up_proj.weight"]],
                                    diag_prefix=ex),
                    hf_index0=e if c.hf.has(f"{ex}.gate_up_proj") or c.hf.has(f"{ex}.gate_up_proj.weight") else None,
                    save_index0=idx))
                c.run(f"moe{L}_e{e}_down", lambda L=L, e=e, fc2=fc2, idx=idx, ex=ex: c.matrix_family(
                    f"moe{L}_e{e}_down", fc2,
                    c.pick_hf([f"{ex}.down_proj", f"{ex}.down_proj.weight", f"{ex}.{e}.down_proj.weight"],
                              diag_prefix=ex),
                    hf_index0=e if c.hf.has(f"{ex}.down_proj") or c.hf.has(f"{ex}.down_proj.weight") else None,
                    save_index0=idx))
            if found["shared_expert_stem"]:
                sh = f"{_LLM}{L}.{found['shared_expert_stem']}"
                hsh = f"{hfl}{L}.mlp.shared_expert"
                c.run(f"moe{L}_shared_gate_up", lambda L=L, sh=sh, hsh=hsh: c.gated_pair_family(
                    f"moe{L}_shared_gate_up", sh + "linear_fc1.weight",
                    c.pick_hf_group([[hsh + ".gate_up_proj.weight"],
                                     [hsh + ".gate_proj.weight", hsh + ".up_proj.weight"],
                                     [hsh + "s.gate_up_proj.weight"]], diag_prefix=f"{hfl}{L}.mlp.shared")))
                c.run(f"moe{L}_shared_down", lambda L=L, sh=sh, hsh=hsh: c.alternates_family(
                    f"moe{L}_shared_down", sh + "linear_fc2.weight",
                    [hsh + ".down_proj.weight", hsh + "s.down_proj.weight"],
                    diag_prefix=f"{hfl}{L}.mlp.shared"))
                c.run(f"moe{L}_shared_gate", lambda L=L, sh=sh: c.alternates_family(
                    f"moe{L}_shared_gate", sh + "gate_weight",
                    [f"{hfl}{L}.mlp.shared_expert_gate.weight", f"{hfl}{L}.mlp.shared_expert.gate.weight"],
                    diag_prefix=f"{hfl}{L}.mlp.shared"))

    for L in vision_layers:
        if L not in found["vision_layers"]:
            continue
        c.run(f"vis{L}_qkv", lambda L=L: c.fused_qkv_family(
            f"vis{L}_qkv", f"{_VIS}{L}.self_attention.linear_qkv.weight",
            f"{hv}encoder.layers.{L}.self_attn.qkv.weight", dims["vision_heads"], dims["vision_head_dim"]))
        c.run(f"vis{L}_proj", lambda L=L: c.identity_family(
            f"vis{L}_proj", f"{_VIS}{L}.self_attention.linear_proj.weight",
            f"{hv}encoder.layers.{L}.self_attn.proj.weight"))
        c.run(f"vis{L}_fc1", lambda L=L: c.identity_family(
            f"vis{L}_fc1", f"{_VIS}{L}.mlp.linear_fc1.weight", f"{hv}encoder.layers.{L}.mlp.fc1.weight"))
        c.run(f"vis{L}_fc2", lambda L=L: c.identity_family(
            f"vis{L}_fc2", f"{_VIS}{L}.mlp.linear_fc2.weight", f"{hv}encoder.layers.{L}.mlp.fc2.weight"))
        c.run(f"vis{L}_ln1", lambda L=L: c.identity_family(
            f"vis{L}_ln1", f"{_VIS}{L}.self_attention.linear_qkv.layer_norm_weight",
            f"{hv}encoder.layers.{L}.layer_norm1.weight"))

    c.run("vision_pre_ln", lambda: c.identity_family(
        "vision_pre_ln", "vision_model.pre_layernorm.weight", hv + "layernorm_pre.weight"))
    c.run("adapter_ln", lambda: c.identity_family(
        "adapter_ln", "adapter.layernorm.weight", hv + "merger.ln_q.weight"))
    c.run("adapter_fc1", lambda: c.identity_family(
        "adapter_fc1", "adapter.linear_fc1.weight", hv + "merger.mlp.0.weight"))
    c.run("adapter_fc2", lambda: c.identity_family(
        "adapter_fc2", "adapter.linear_fc2.weight", hv + "merger.mlp.2.weight"))

    # The merged video line trains WITHOUT the vision final LayerNorm (OV2_MTP_LAYERS=0 -> mcore never
    # builds it); the export must agree with the SAVE either way.
    c.run("vision_final_ln", lambda: c.absent_family(
        "vision_final_ln", "vision_model.decoder.final_layernorm.weight", hv + "layernorm_post.weight")
        if not found["has_vision_final_ln"] else c.identity_family(
        "vision_final_ln", "vision_model.decoder.final_layernorm.weight", hv + "layernorm_post.weight"))


def _int_list(text: str) -> List[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x != ""]


def main() -> int:
    """Parse arguments, discover both key spaces, run the families and report."""
    ap = argparse.ArgumentParser(
        description="Compare a Qwen3.5 OV2 torch_dist SAVE against its HF export (CPU, arrangement-aware).")
    ap.add_argument("--save", required=True, type=Path, help="training SAVE iter dir (holds .metadata)")
    ap.add_argument("--hf", required=True, type=Path, help="HF export dir (config.json + safetensors)")
    ap.add_argument("--layers", default="", help="LLM layer ids (default: first attention + first GDN layer)")
    ap.add_argument("--expert", default="0", help="expert ids per MoE layer (default 0)")
    ap.add_argument("--vision-layers", default="0", help="vision layer ids (default 0)")
    ap.add_argument("--rows", type=int, default=4096,
                    help="row budget for tall 1:1 tensors; sampled head+tail (0 = whole tensor)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (default 0 = bit-exact)")
    ap.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (default 0 = bit-exact)")
    ap.add_argument("--max-load-gb", type=float, default=8.0, help="per-tensor SAVE load budget (GiB)")
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the byte-level row map printed when no arrangement matches")
    ap.add_argument("--dump-keys", action="store_true", help="print both key spaces plus discovery, then exit")
    ap.add_argument("--grep", default="", help="with --dump-keys: only print keys containing this substring")
    ap.add_argument("--allow-unresolved", action="store_true", help="exit 0 even if a family was UNRESOLVED")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not (args.save / ".metadata").is_file():
        logger.error("FATAL: %s has no .metadata (not a torch_dist checkpoint dir)", args.save)
        return 2
    if not (args.hf / "config.json").is_file():
        logger.error("FATAL: %s has no config.json (not an HF export dir)", args.hf)
        return 2

    try:
        save = SaveReader(args.save, int(args.max_load_gb * (1024 ** 3)))
        hf = HFReader(args.hf)
    except Exception as exc:  # noqa: BLE001
        logger.error("FATAL: cannot open inputs: %r", exc)
        return 2

    dims = hf_dims(args.hf)
    found = discover(save.keys())
    logger.info("[save-vs-hf] SAVE %s (%d keys)", args.save, len(save.keys()))
    logger.info("[save-vs-hf] HF   %s (%d tensors)", args.hf, len(hf.keys()))
    logger.info("[save-vs-hf] dims %s", json.dumps(dims))
    logger.info("[save-vs-hf] found %s", json.dumps(
        {k: (v[:8] + ["..."] if isinstance(v, list) and len(v) > 8 else v) for k, v in found.items()}))

    if args.dump_keys:
        logger.info("---- SAVE keys ----")
        for k in save.keys():
            if args.grep and args.grep not in k:
                continue
            desc = save.describe(k)
            logger.info("  %s %s", k, tuple(desc[0]) if desc else "(bytes)")
        logger.info("---- HF keys ----")
        for k in hf.keys():
            if args.grep and args.grep not in k:
                continue
            logger.info("  %s %s", k, hf.shape(k))
        return 0

    layers = _int_list(args.layers)
    if not layers:
        layers = sorted(set(
            ([found["attention_layers"][0]] if found["attention_layers"] else [])
            + ([found["gdn_layers"][0]] if found["gdn_layers"] else [])
            + ([found["moe_layers"][0]] if found["moe_layers"] else [])))
    experts = _int_list(args.expert)
    vision_layers = _int_list(args.vision_layers)
    logger.info("[save-vs-hf] checking layers=%s experts=%s vision_layers=%s", layers, experts, vision_layers)

    checker = Checker(save, hf, args.atol, args.rtol, args.rows, explain=not args.no_explain)
    build_and_run(checker, found, dims, layers, experts, vision_layers)

    counts = {PASS: 0, FAIL: 0, UNRESOLVED: 0, SKIP: 0}
    logger.info("")
    logger.info("%-24s %-11s %-38s %s", "FAMILY", "STATUS", "ARRANGEMENT", "DETAIL")
    for r in checker.results:
        counts[r.status] += 1
        logger.info("%-24s %-11s %-38s %s", r.family, r.status, r.arrangement or "-", r.detail)
    logger.info("")
    logger.info("[save-vs-hf] %d PASS  %d FAIL  %d UNRESOLVED  %d SKIP",
                counts[PASS], counts[FAIL], counts[UNRESOLVED], counts[SKIP])
    logger.info("[save-vs-hf] SCOPE: compares the export against the SAVE for the families above. It does not "
                "prove HF-vs-mcore logits, M-RoPE at inference, MTP semantics, or unsampled families.")

    verdict = "FAIL" if counts[FAIL] else (
        "UNRESOLVED" if counts[UNRESOLVED] and not args.allow_unresolved else "PASS")
    if args.json:
        args.json.write_text(json.dumps({
            "save": str(args.save), "hf": str(args.hf), "dims": dims, "found": found,
            "layers": layers, "experts": experts, "vision_layers": vision_layers,
            "atol": args.atol, "rtol": args.rtol, "rows": args.rows,
            "counts": counts, "verdict": verdict,
            "results": [r.as_dict() for r in checker.results],
        }, indent=2, sort_keys=True))
        logger.info("[save-vs-hf] report -> %s", args.json)

    if counts[FAIL]:
        return 1
    if counts[UNRESOLVED] and not args.allow_unresolved:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
