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
  3. Asks, per family: **is the HF tensor a known REARRANGEMENT of the SAVE tensor, and is it the canonical
     one?** The candidate set deliberately holds the canonical arrangement AND the classic traps
     (contiguous-vs-GQA-interleaved QKV, TP-rank-interleaved gate/up, transposed matrices, a zero-centred
     norm exported without its +1, parts concatenated in the wrong order...). The report names the
     arrangement that matched.

     A trap arrangement matching is a FAIL with a diagnosis. No arrangement matching is a FAIL with the
     smallest max-abs-diff reached. A family whose keys resolve on neither side is UNRESOLVED -- never a
     pass.

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
    python3 save_vs_hf_arrangement.py --save ... --hf ... --layers 0,3 --expert 0,17 --json report.json

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


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Layout arithmetic. Pure functions over ints returning index lists, so every layout claim in this file
# is unit-testable without torch, DCP or a checkpoint (tests/unit_tests/models/ov2/test_save_vs_hf_layout.py).
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def qkv_rows_grouped(num_heads: int, num_kv_heads: int, head_dim: int) -> Tuple[List[int], List[int], List[int]]:
    """Row indices of q / k / v inside mcore's fused ``linear_qkv.weight``.

    mcore interleaves by KV group so each group's queries sit next to their K and V:
    ``[q_1..q_n, k_1, v_1][q_1..q_n, k_2, v_2]...`` with ``n = num_heads // num_kv_heads``. This is the
    canonical layout; a merge that concatenated the three projections instead gives ``qkv_rows_contiguous``.
    """
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"bad geometry: heads={num_heads} kv={num_kv_heads} head_dim={head_dim}")
    if num_heads % num_kv_heads:
        raise ValueError(f"num_heads={num_heads} is not a multiple of num_kv_heads={num_kv_heads}")
    per_group = num_heads // num_kv_heads
    q: List[int] = []
    k: List[int] = []
    v: List[int] = []
    block = (per_group + 2) * head_dim
    for g in range(num_kv_heads):
        base = g * block
        q.extend(range(base, base + per_group * head_dim))
        k.extend(range(base + per_group * head_dim, base + (per_group + 1) * head_dim))
        v.extend(range(base + (per_group + 1) * head_dim, base + (per_group + 2) * head_dim))
    return q, k, v


def qkv_rows_contiguous(num_heads: int, num_kv_heads: int, head_dim: int) -> Tuple[List[int], List[int], List[int]]:
    """The trap layout: all of q, then all of k, then all of v (what a naive row split produces)."""
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"bad geometry: heads={num_heads} kv={num_kv_heads} head_dim={head_dim}")
    nq, nk = num_heads * head_dim, num_kv_heads * head_dim
    return list(range(nq)), list(range(nq, nq + nk)), list(range(nq + nk, nq + 2 * nk))


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
    """``[(start, stop)]`` for each part of a row-concatenation (GDN's 4-way ``in_proj`` split)."""
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
        """One dim-0 slice of a stacked tensor, without materialising the whole thing.

        Uses mcore's sharded-tensor load (the same mechanism ``verify_consistency.py`` relies on) to read
        only that slice; falls back to a full load when mcore is unavailable and the tensor fits the budget.
        """
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

    def __init__(self, save: SaveReader, hf: HFReader, atol: float, rtol: float, rows: int):
        self.save = save
        self.hf = hf
        self.atol = atol
        self.rtol = rtol
        self.rows = rows
        self.results: List[Result] = []

    def run(self, family: str, fn: Callable[[], Result]) -> None:
        try:
            self.results.append(fn())
        except KeyError as exc:
            self.results.append(Result(family, UNRESOLVED, detail=f"missing key {exc}"))
        except MemoryError as exc:
            self.results.append(Result(family, UNRESOLVED, detail=str(exc)))
        except Exception as exc:  # noqa: BLE001 -- one broken family must not hide the others
            self.results.append(Result(family, UNRESOLVED, detail=f"{type(exc).__name__}: {exc}"))

    def _verdict(self, family: str, save_keys: Sequence[str], hf_keys: Sequence[str],
                 candidates: Sequence[Tuple[str, bool, Any, Any]]) -> Result:
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
        return Result(family, FAIL, "none",
                      f"no known arrangement reproduces the HF tensor from the SAVE "
                      f"(smallest max|diff| = {best:.6g}); the values themselves differ",
                      save_keys, hf_keys, best)

    # -- 1:1 families ------------------------------------------------------------------------------
    def identity_family(self, family: str, save_key: str, hf_key: str, *, plus_one: bool = False,
                        transpose_ok: bool = True) -> Result:
        """1:1 tensors. ``plus_one``: canonical is HF == SAVE + 1 (mcore zero-centred RMSNorm -> HF RMSNorm).

        Tall tensors are sampled head+tail (``--rows``); a sampled family never tries the transpose
        candidate, because a row window of a transpose is not the transpose of a row window.
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
            # The SAVE's dim 0 can be LONGER than the export's (vocab-parallel padding), so the SAVE is
            # always trimmed to the export's rows before comparing -- never compared whole against a
            # shorter tensor, which would read as a mismatch when nothing is wrong.
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
        # A SAVE that is TALLER than the export is normal for vocab-parallel padding; say so explicitly
        # rather than letting a reader assume the extra rows were checked.
        if result.status == PASS and sv_full.ndim >= 1 and hf_shape and sv_full.shape[0] > hf_shape[0]:
            result.detail += (f"; SAVE has {int(sv_full.shape[0]) - int(hf_shape[0])} extra dim-0 rows "
                              f"(vocab padding) that the export legitimately drops")
        return result

    # -- fused / packed families -------------------------------------------------------------------
    def qkv_family(self, family: str, save_key: str, hf_q: str, hf_k: str, hf_v: str,
                   num_heads: int, num_kv_heads: int, head_dim: int) -> Result:
        import torch

        sv = self.save.load(save_key)
        hv = torch.cat([self.hf.load(hf_q), self.hf.load(hf_k), self.hf.load(hf_v)], dim=0)
        cands: List[Tuple[str, bool, Any, Any]] = []
        for name, canonical, idx in (
            ("gqa_group_interleaved", True, qkv_rows_grouped(num_heads, num_kv_heads, head_dim)),
            ("contiguous_q_k_v", False, qkv_rows_contiguous(num_heads, num_kv_heads, head_dim)),
        ):
            order = idx[0] + idx[1] + idx[2]
            if len(order) != int(sv.shape[0]):
                continue
            cands.append((name, canonical, sv[torch.as_tensor(order, dtype=torch.long)], hv))
        if not cands:
            return Result(family, UNRESOLVED, detail=(
                f"fused rows {tuple(sv.shape)} do not match heads={num_heads} kv={num_kv_heads} "
                f"head_dim={head_dim}"), save_keys=[save_key], hf_keys=[hf_q, hf_k, hf_v])
        return self._verdict(family, [save_key], [hf_q, hf_k, hf_v], cands)

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

    def gated_pair_family(self, family: str, save_key: str, hf_key: str, *, hf_index0: Optional[int] = None,
                          save_index0: Optional[int] = None,
                          tp_candidates: Sequence[int] = (2, 4, 8)) -> Result:
        """Fused gate/up matrices (routed experts, shared expert, dense MLP).

        Canonical: the HF tensor is the SAVE's global ``[gate; up]`` rows unchanged (or transposed into HF's
        layout when the shapes say so). Traps: gate and up interleaved in TP-rank blocks -- what a merge
        that concatenated ``linear_fc1`` rank slices without honouring the swiglu split produces -- or the
        two halves swapped.
        """
        import torch

        sv = self.save.load_dim0_index(save_key, save_index0) if save_index0 is not None else self.save.load(save_key)
        hv = self.hf.load(hf_key, index0=hf_index0)
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
        return self._verdict(family, [save_key], [hf_key], cands)

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
        return self._verdict(family, [save_key], [hf_key], cands)

    def concat_split_family(self, family: str, save_key: str, hf_keys: Sequence[str]) -> Result:
        """GDN ``in_proj``: one fused SAVE matrix vs several HF matrices concatenated in a declared order.

        Canonical = the order the Qwen3.5 GDN mapping declares (qkv, z, b, a). Every other ordering of the
        same parts is a trap candidate, so a swapped pair is named instead of showing up as noise.
        """
        import torch

        sv = self.save.load(save_key)
        parts = {k: self.hf.load(k) for k in hf_keys}
        sizes = [int(parts[k].shape[0]) for k in hf_keys]
        if sum(sizes) != int(sv.shape[0]):
            return Result(family, UNRESOLVED,
                          detail=(f"parts sum to {sum(sizes)} rows {concat_offsets(sizes)}, SAVE has "
                                  f"{int(sv.shape[0])}"),
                          save_keys=[save_key], hf_keys=list(hf_keys))
        cands: List[Tuple[str, bool, Any, Any]] = []
        for order in permutations(range(len(hf_keys))):
            names = [hf_keys[i] for i in order]
            label = "+".join(n.rsplit(".", 2)[-2] for n in names)
            cands.append((f"concat[{label}]", order == tuple(range(len(hf_keys))), sv,
                          torch.cat([parts[n] for n in names], dim=0)))
        cands.sort(key=lambda c: not c[1])  # canonical order first
        return self._verdict(family, [save_key], list(hf_keys), cands)

    def conv1d_family(self, family: str, save_key: str, hf_key: str) -> Result:
        sv = self.save.load(save_key)
        hv = self.hf.load(hf_key)
        cands: List[Tuple[str, bool, Any, Any]] = []
        if tuple(sv.shape) == tuple(hv.shape):
            cands.append(("identity", True, sv, hv))
        if sv.ndim == 3 and hv.ndim == 2 and sv.shape[1] == 1:
            cands.append(("squeezed_channel_dim", True, sv.squeeze(1), hv))
        if sv.ndim == 2 and hv.ndim == 3 and hv.shape[1] == 1:
            cands.append(("unsqueezed_channel_dim", True, sv.unsqueeze(1), hv))
        if not cands:
            return Result(family, UNRESOLVED, detail=f"SAVE{tuple(sv.shape)} vs HF{tuple(hv.shape)}",
                          save_keys=[save_key], hf_keys=[hf_key])
        return self._verdict(family, [save_key], [hf_key], cands)

    def absent_family(self, family: str, save_key: str, hf_key: str) -> Result:
        """The tensor must be present on both sides or on neither (the merged line's vision final LN)."""
        in_save = self.save.describe(save_key) is not None
        in_hf = self.hf.has(hf_key)
        if in_save == in_hf:
            state = "present on both" if in_save else "absent_on_both"
            return Result(family, SKIP if not in_save else PASS, state,
                          "the export carries this tensor exactly when the SAVE does", [save_key], [hf_key])
        return Result(family, FAIL, "present" if in_hf else "missing",
                      f"SAVE has it: {in_save}; HF has it: {in_hf} -- the export must carry this tensor if "
                      f"and only if the SAVE does", [save_key], [hf_key])


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Discovery -- what the SAVE actually contains; never assumed from the config
# ──────────────────────────────────────────────────────────────────────────────────────────────────
_LLM = "language_model.decoder.layers."
_VIS = "vision_model.decoder.layers."


def _layer_ids(keys: Sequence[str], prefix: str, marker: str) -> List[int]:
    pat = re.compile(re.escape(prefix) + r"(\d+)\." + re.escape(marker) + r"$")
    return sorted({int(m.group(1)) for m in (pat.match(k) for k in keys) if m})


def discover(save_keys: Sequence[str]) -> dict:
    """Which layers are attention / GDN / MoE, how experts are stored, what the SAVE carries at all."""
    per_expert = re.compile(re.escape(_LLM) + r"(\d+)\.mlp\.experts\.linear_fc1\.weight(\d+)$")
    sequential = re.compile(re.escape(_LLM) + r"(\d+)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$")
    stacked = re.compile(re.escape(_LLM) + r"(\d+)\.mlp\.experts\.linear_fc1\.weight$")
    style: Optional[str] = None
    experts: List[int] = []
    for k in save_keys:
        m = per_expert.match(k)
        if m:
            style = style or "per_expert_suffix"
            experts.append(int(m.group(2)))
            continue
        m = sequential.match(k)
        if m:
            style = style or "local_experts"
            experts.append(int(m.group(2)))
            continue
        if stacked.match(k):
            style = style or "stacked"
    return {
        "attention_layers": _layer_ids(save_keys, _LLM, "self_attention.linear_qkv.weight"),
        "gdn_layers": _layer_ids(save_keys, _LLM, "self_attention.in_proj.weight"),
        "moe_layers": _layer_ids(save_keys, _LLM, "mlp.router.weight"),
        "vision_layers": _layer_ids(save_keys, _VIS, "self_attention.linear_qkv.weight"),
        "expert_style": style,
        "expert_ids": sorted(set(experts)),
        "has_shared_experts": any(".mlp.shared_experts." in k for k in save_keys),
        "has_mtp": any(".mtp." in k for k in save_keys),
        "has_vision_final_ln": any(k.startswith("vision_model.decoder.final_layernorm.") for k in save_keys),
    }


def expert_save_keys(layer: int, expert: int, style: Optional[str]) -> Tuple[str, str, Optional[int]]:
    """(fc1_key, fc2_key, save_dim0_index) for the discovered expert storage style."""
    base = f"{_LLM}{layer}.mlp.experts."
    if style == "per_expert_suffix":
        return f"{base}linear_fc1.weight{expert}", f"{base}linear_fc2.weight{expert}", None
    if style == "local_experts":
        return (f"{base}local_experts.{expert}.linear_fc1.weight",
                f"{base}local_experts.{expert}.linear_fc2.weight", None)
    return f"{base}linear_fc1.weight", f"{base}linear_fc2.weight", expert


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
    v_heads = int(vision.get("num_attention_heads", 0) or 0)
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

    checker.run("embed_tokens", lambda: checker.identity_family(
        "embed_tokens", "language_model.embedding.word_embeddings.weight", hf_llm + "embed_tokens.weight",
        transpose_ok=False))
    checker.run("lm_head", lambda: checker.identity_family(
        "lm_head", "language_model.output_layer.weight", "lm_head.weight", transpose_ok=False))
    checker.run("final_norm", lambda: checker.identity_family(
        "final_norm", "language_model.decoder.final_layernorm.weight", hf_llm + "norm.weight"))

    for L in layers:
        if L in found["attention_layers"]:
            checker.run(f"attn{L}_qkv", lambda L=L: checker.qkv_family(
                f"attn{L}_qkv", f"{_LLM}{L}.self_attention.linear_qkv.weight",
                f"{hfl}{L}.self_attn.q_proj.weight", f"{hfl}{L}.self_attn.k_proj.weight",
                f"{hfl}{L}.self_attn.v_proj.weight",
                dims["num_heads"], dims["num_kv_heads"], dims["head_dim"]))
            checker.run(f"attn{L}_o_proj", lambda L=L: checker.identity_family(
                f"attn{L}_o_proj", f"{_LLM}{L}.self_attention.linear_proj.weight",
                f"{hfl}{L}.self_attn.o_proj.weight"))
            checker.run(f"attn{L}_q_norm", lambda L=L: checker.identity_family(
                f"attn{L}_q_norm", f"{_LLM}{L}.self_attention.q_layernorm.weight",
                f"{hfl}{L}.self_attn.q_norm.weight"))
            checker.run(f"attn{L}_k_norm", lambda L=L: checker.identity_family(
                f"attn{L}_k_norm", f"{_LLM}{L}.self_attention.k_layernorm.weight",
                f"{hfl}{L}.self_attn.k_norm.weight"))
            checker.run(f"attn{L}_input_ln", lambda L=L: checker.identity_family(
                f"attn{L}_input_ln", f"{_LLM}{L}.self_attention.linear_qkv.layer_norm_weight",
                f"{hfl}{L}.input_layernorm.weight"))

        if L in found["gdn_layers"]:
            checker.run(f"gdn{L}_in_proj", lambda L=L: checker.concat_split_family(
                f"gdn{L}_in_proj", f"{_LLM}{L}.self_attention.in_proj.weight",
                [f"{hfl}{L}.linear_attn.in_proj_qkv.weight", f"{hfl}{L}.linear_attn.in_proj_z.weight",
                 f"{hfl}{L}.linear_attn.in_proj_b.weight", f"{hfl}{L}.linear_attn.in_proj_a.weight"]))
            checker.run(f"gdn{L}_conv1d", lambda L=L: checker.conv1d_family(
                f"gdn{L}_conv1d", f"{_LLM}{L}.self_attention.conv1d.weight",
                f"{hfl}{L}.linear_attn.conv1d.weight"))
            # mcore keeps a zero-centred RMSNorm here while the HF class expects the standard one -> +1.
            checker.run(f"gdn{L}_out_norm", lambda L=L: checker.identity_family(
                f"gdn{L}_out_norm", f"{_LLM}{L}.self_attention.out_norm.weight",
                f"{hfl}{L}.linear_attn.norm.weight", plus_one=True))
            checker.run(f"gdn{L}_out_proj", lambda L=L: checker.identity_family(
                f"gdn{L}_out_proj", f"{_LLM}{L}.self_attention.out_proj.weight",
                f"{hfl}{L}.linear_attn.out_proj.weight"))
            checker.run(f"gdn{L}_A_log", lambda L=L: checker.identity_family(
                f"gdn{L}_A_log", f"{_LLM}{L}.self_attention.A_log", f"{hfl}{L}.linear_attn.A_log"))
            checker.run(f"gdn{L}_dt_bias", lambda L=L: checker.identity_family(
                f"gdn{L}_dt_bias", f"{_LLM}{L}.self_attention.dt_bias", f"{hfl}{L}.linear_attn.dt_bias"))
            checker.run(f"gdn{L}_input_ln", lambda L=L: checker.identity_family(
                f"gdn{L}_input_ln", f"{_LLM}{L}.self_attention.in_proj.layer_norm_weight",
                f"{hfl}{L}.input_layernorm.weight"))

        if L in found["moe_layers"]:
            checker.run(f"moe{L}_router", lambda L=L: checker.identity_family(
                f"moe{L}_router", f"{_LLM}{L}.mlp.router.weight", f"{hfl}{L}.mlp.gate.weight"))
            checker.run(f"moe{L}_pre_mlp_ln", lambda L=L: checker.identity_family(
                f"moe{L}_pre_mlp_ln", f"{_LLM}{L}.pre_mlp_layernorm.weight",
                f"{hfl}{L}.post_attention_layernorm.weight"))
            for e in experts:
                fc1, fc2, save_idx = expert_save_keys(L, e, found["expert_style"])
                checker.run(f"moe{L}_e{e}_gate_up", lambda L=L, e=e, fc1=fc1, si=save_idx:
                            checker.gated_pair_family(
                                f"moe{L}_e{e}_gate_up", fc1, f"{hfl}{L}.mlp.experts.gate_up_proj",
                                hf_index0=e, save_index0=si))
                checker.run(f"moe{L}_e{e}_down", lambda L=L, e=e, fc2=fc2, si=save_idx:
                            checker.matrix_family(
                                f"moe{L}_e{e}_down", fc2, f"{hfl}{L}.mlp.experts.down_proj",
                                hf_index0=e, save_index0=si))
            if found["has_shared_experts"]:
                checker.run(f"moe{L}_shared_gate_up", lambda L=L: checker.gated_pair_family(
                    f"moe{L}_shared_gate_up", f"{_LLM}{L}.mlp.shared_experts.linear_fc1.weight",
                    f"{hfl}{L}.mlp.shared_expert.gate_up_proj.weight"))
                checker.run(f"moe{L}_shared_down", lambda L=L: checker.matrix_family(
                    f"moe{L}_shared_down", f"{_LLM}{L}.mlp.shared_experts.linear_fc2.weight",
                    f"{hfl}{L}.mlp.shared_expert.down_proj.weight"))

    for L in vision_layers:
        if L not in found["vision_layers"]:
            continue
        checker.run(f"vis{L}_qkv", lambda L=L: checker.fused_qkv_family(
            f"vis{L}_qkv", f"{_VIS}{L}.self_attention.linear_qkv.weight",
            f"{hv}encoder.layers.{L}.self_attn.qkv.weight", dims["vision_heads"], dims["vision_head_dim"]))
        checker.run(f"vis{L}_proj", lambda L=L: checker.identity_family(
            f"vis{L}_proj", f"{_VIS}{L}.self_attention.linear_proj.weight",
            f"{hv}encoder.layers.{L}.self_attn.proj.weight"))
        checker.run(f"vis{L}_fc1", lambda L=L: checker.identity_family(
            f"vis{L}_fc1", f"{_VIS}{L}.mlp.linear_fc1.weight", f"{hv}encoder.layers.{L}.mlp.fc1.weight"))
        checker.run(f"vis{L}_fc2", lambda L=L: checker.identity_family(
            f"vis{L}_fc2", f"{_VIS}{L}.mlp.linear_fc2.weight", f"{hv}encoder.layers.{L}.mlp.fc2.weight"))
        checker.run(f"vis{L}_ln1", lambda L=L: checker.identity_family(
            f"vis{L}_ln1", f"{_VIS}{L}.self_attention.linear_qkv.layer_norm_weight",
            f"{hv}encoder.layers.{L}.layer_norm1.weight"))

    checker.run("vision_pre_ln", lambda: checker.identity_family(
        "vision_pre_ln", "vision_model.pre_layernorm.weight", hv + "layernorm_pre.weight"))
    checker.run("adapter_ln", lambda: checker.identity_family(
        "adapter_ln", "adapter.layernorm.weight", hv + "merger.ln_q.weight"))
    checker.run("adapter_fc1", lambda: checker.identity_family(
        "adapter_fc1", "adapter.linear_fc1.weight", hv + "merger.mlp.0.weight"))
    checker.run("adapter_fc2", lambda: checker.identity_family(
        "adapter_fc2", "adapter.linear_fc2.weight", hv + "merger.mlp.2.weight"))

    # The merged video line trains WITHOUT the vision final LayerNorm (OV2_MTP_LAYERS=0 -> mcore never
    # builds it); the export must agree with the SAVE either way.
    checker.run("vision_final_ln", lambda: checker.absent_family(
        "vision_final_ln", "vision_model.decoder.final_layernorm.weight", hv + "layernorm_post.weight")
        if not found["has_vision_final_ln"] else checker.identity_family(
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
    ap.add_argument("--dump-keys", action="store_true", help="print both key spaces plus discovery, then exit")
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
        {k: (v[:12] + ["..."] if isinstance(v, list) and len(v) > 12 else v) for k, v in found.items()}))

    if args.dump_keys:
        logger.info("---- SAVE keys ----")
        for k in save.keys():
            desc = save.describe(k)
            logger.info("  %s %s", k, tuple(desc[0]) if desc else "(bytes)")
        logger.info("---- HF keys ----")
        for k in hf.keys():
            logger.info("  %s %s", k, hf.shape(k))
        return 0

    layers = _int_list(args.layers)
    if not layers:
        layers = ([found["attention_layers"][0]] if found["attention_layers"] else []) + \
                 ([found["gdn_layers"][0]] if found["gdn_layers"] else [])
        layers = sorted(set(layers + ([found["moe_layers"][0]] if found["moe_layers"] else [])))
    experts = _int_list(args.expert)
    vision_layers = _int_list(args.vision_layers)
    logger.info("[save-vs-hf] checking layers=%s experts=%s vision_layers=%s", layers, experts, vision_layers)

    checker = Checker(save, hf, args.atol, args.rtol, args.rows)
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
