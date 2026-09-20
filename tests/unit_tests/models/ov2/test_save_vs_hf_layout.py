# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout arithmetic and key-space discovery behind ``save_vs_hf_arrangement.py``.

Pure integer and string reasoning -- no torch, no GPU, no checkpoint -- so the claims the comparator makes
about mcore's layouts can be checked on any machine. The end-to-end behaviour (real DCP files, safetensors,
injected defects) is covered by ``save_vs_hf_arrangement_selftest.py``, which needs torch.

Runnable both ways::

    pytest tests/unit_tests/models/ov2/test_save_vs_hf_layout.py
    python3 tests/unit_tests/models/ov2/test_save_vs_hf_layout.py
"""

import importlib.util
from pathlib import Path

import numpy as np


try:
    import pytest
except ImportError:  # standalone run on a machine without pytest -- only `raises` is used below
    import contextlib
    import types

    @contextlib.contextmanager
    def _raises(expected):
        try:
            yield
        except expected:
            return
        raise AssertionError(f"expected {expected.__name__}")

    pytest = types.SimpleNamespace(raises=_raises)


_TOOL = (Path(__file__).resolve().parents[4] / "examples" / "models" / "qwen" / "qwen35_vl_ov2" /
         "convert" / "save_vs_hf_arrangement.py")
_LLM = "language_model.decoder.layers."


def _load_module():
    spec = importlib.util.spec_from_file_location("save_vs_hf_arrangement", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_module()


# ── QKV ───────────────────────────────────────────────────────────────────────────────────────────
def test_grouped_qkv_matches_the_documented_mcore_layout():
    """4 heads / 2 KV groups / head_dim 2 -> [q q k v][q q k v] with 2 query heads per group."""
    q, k, v = M.qkv_rows_grouped(4, 2, 2)
    assert q == [0, 1, 2, 3, 8, 9, 10, 11]
    assert k == [4, 5, 12, 13]
    assert v == [6, 7, 14, 15]


def test_grouped_qkv_is_a_partition_of_the_fused_rows():
    for heads, kv, hd in ((4, 2, 2), (64, 8, 128), (16, 16, 4), (12, 4, 64)):
        q, k, v = M.qkv_rows_grouped(heads, kv, hd)
        rows = (heads + 2 * kv) * hd
        assert sorted(q + k + v) == list(range(rows))
        assert len(q) == heads * hd and len(k) == kv * hd and len(v) == kv * hd


def test_block_form_handles_qwen35_gated_attention():
    """The real merged SAVE: 16 heads, 2 KV groups, head_dim 256, and q carries its gate -> 9216 rows."""
    nq, nk, nv = 2 * 16 * 256, 2 * 256, 2 * 256
    assert nq + nk + nv == 9216
    q, k, v = M.qkv_rows_grouped_blocks(nq, nk, nv, 2)
    assert sorted(q + k + v) == list(range(9216))
    assert len(q) == 8192 and len(k) == 512 and len(v) == 512
    # Group 0 owns the first block: half the q(+gate) rows, then its k, then its v.
    assert q[:3] == [0, 1, 2] and k[0] == 4096 and v[0] == 4096 + 256
    assert q[4096] == 4608  # group 1 starts after the first (4096 + 256 + 256) block


def test_block_form_agrees_with_the_head_count_form_when_ungated():
    assert M.qkv_rows_grouped_blocks(4 * 2, 2 * 2, 2 * 2, 2) == M.qkv_rows_grouped(4, 2, 2)


def test_contiguous_qkv_is_the_other_partition_and_differs_under_gqa():
    q, k, v = M.qkv_rows_contiguous(4, 2, 2)
    assert q == list(range(8)) and k == [8, 9, 10, 11] and v == [12, 13, 14, 15]
    assert M.qkv_rows_contiguous(4, 2, 2) != M.qkv_rows_grouped(4, 2, 2)
    assert M.qkv_rows_contiguous(8, 8, 4) != M.qkv_rows_grouped(8, 8, 4)


def test_contiguous_block_form():
    q, k, v = M.qkv_rows_contiguous_blocks(16, 4, 4)
    assert q == list(range(16)) and k == [16, 17, 18, 19] and v == [20, 21, 22, 23]
    assert M.qkv_rows_contiguous_blocks(16, 4, 4) != M.qkv_rows_grouped_blocks(16, 4, 4, 2)


def test_gated_head_interleave_small_case():
    """4 heads / 2 groups / head_dim 2, gated: each group is [q-block, gate-block, k, v] in the SAVE."""
    q, k, v = M.qkv_rows_gated_head_interleaved(16, 4, 4, 2)
    # group 0 occupies rows 0..11: q heads at 0-1 / 2-3, their gates at 4-5 / 6-7, then k, then v
    assert q[:8] == [0, 1, 4, 5, 2, 3, 6, 7]
    assert k[:2] == [8, 9] and v[:2] == [10, 11]
    # group 1 repeats the pattern one block (12 rows) later
    assert q[8:] == [12, 13, 16, 17, 14, 15, 18, 19]
    assert k[2:] == [20, 21] and v[2:] == [22, 23]


def test_gated_head_interleave_covers_the_real_geometry():
    """16 heads, 2 KV groups, head_dim 256, q carrying its gate -> a partition of all 9216 rows."""
    q, k, v = M.qkv_rows_gated_head_interleaved(8192, 512, 512, 2)
    assert sorted(q + k + v) == list(range(9216))
    assert len(q) == 8192 and len(k) == 512 and len(v) == 512
    # The first head's gate sits a whole q-block (npg * head_dim = 8 * 256) after its query rows.
    assert q[:2] == [0, 1] and q[256] == 2048


def test_gated_interleave_differs_from_the_plain_group_block():
    assert M.qkv_rows_gated_head_interleaved(16, 4, 4, 2) != M.qkv_rows_grouped_blocks(16, 4, 4, 2)


def test_gated_head_interleave_validates_its_geometry():
    with pytest.raises(ValueError):
        M.qkv_rows_gated_head_interleaved(16, 4, 4, 3)     # rows do not split into 3 groups
    with pytest.raises(ValueError):
        M.qkv_rows_gated_head_interleaved(16, 4, 8, 2)     # k and v disagree on head_dim
    with pytest.raises(ValueError):
        M.qkv_rows_gated_head_interleaved(12, 8, 8, 2)     # q rows per group are not 2 * head_dim * heads


def test_grouped_indices_recover_per_head_order():
    """Rows tagged by (head, kind) come back grouped by kind, heads ascending -- what HF q/k/v expect."""
    heads, kv, hd = 6, 3, 4
    per_group = heads // kv
    tags = []
    for g in range(kv):
        for h in range(per_group):
            tags += [("q", g * per_group + h)] * hd
        tags += [("k", g)] * hd
        tags += [("v", g)] * hd
    tags = np.array(tags, dtype=object)
    q, k, v = M.qkv_rows_grouped(heads, kv, hd)
    assert [t[0] for t in tags[q]] == ["q"] * (heads * hd)
    assert [t[1] for t in tags[q]] == [h for h in range(heads) for _ in range(hd)]
    assert [t[0] for t in tags[k]] == ["k"] * (kv * hd)
    assert [t[1] for t in tags[v]] == [g for g in range(kv) for _ in range(hd)]


def test_head_interleaved_is_the_mha_case():
    assert M.qkv_rows_head_interleaved(3, 2) == M.qkv_rows_grouped(3, 3, 2)


def test_qkv_geometry_is_validated():
    with pytest.raises(ValueError):
        M.qkv_rows_grouped(5, 2, 4)  # heads not a multiple of kv heads
    with pytest.raises(ValueError):
        M.qkv_rows_grouped(4, 0, 4)
    with pytest.raises(ValueError):
        M.qkv_rows_contiguous(4, 2, 0)
    with pytest.raises(ValueError):
        M.qkv_rows_grouped_blocks(9, 4, 4, 2)  # q rows not divisible by the group count


# ── SwiGLU / TP interleave ────────────────────────────────────────────────────────────────────────
def test_deinterleave_rows_small_case():
    assert M.deinterleave_rows(12, 2) == [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]


def test_deinterleave_undoes_a_tp_rank_concatenation():
    """Build [gate;up] per rank, concatenate the ranks, and check the indices restore [gate;up].

    This is the failure the merged (TP4) stage newly exposes: rank blocks concatenated without honouring
    the swiglu split leave gate and up interleaved, and every EP export shows the same interleave.
    """
    for tp in (1, 2, 4, 8):
        rows = 48
        half = rows // 2
        gate = [("gate", i) for i in range(half)]
        up = [("up", i) for i in range(half)]
        per_rank = half // tp
        interleaved = []
        for r in range(tp):
            interleaved += gate[r * per_rank : (r + 1) * per_rank]
            interleaved += up[r * per_rank : (r + 1) * per_rank]
        restored = [interleaved[i] for i in M.deinterleave_rows(rows, tp)]
        assert restored == gate + up


def test_deinterleave_is_a_permutation():
    for tp in (1, 2, 4):
        idx = M.deinterleave_rows(48, tp)
        assert sorted(idx) == list(range(48))


def test_deinterleave_rejects_indivisible_shapes():
    with pytest.raises(ValueError):
        M.deinterleave_rows(12, 5)
    with pytest.raises(ValueError):
        M.deinterleave_rows(12, 0)


def test_swap_halves():
    assert M.swap_halves(6) == [3, 4, 5, 0, 1, 2]
    with pytest.raises(ValueError):
        M.swap_halves(7)


# ── concatenation + sampling ──────────────────────────────────────────────────────────────────────
def test_concat_offsets():
    assert M.concat_offsets([10, 4, 2, 2]) == [(0, 10), (10, 14), (14, 16), (16, 18)]
    assert M.concat_offsets([]) == []


def test_sample_windows_covers_head_and_tail():
    assert M.sample_windows(100, 0) == [(0, 100)]
    assert M.sample_windows(100, 200) == [(0, 100)]
    windows = M.sample_windows(1000, 100)
    assert windows == [(0, 50), (950, 1000)]
    assert sum(b - a for a, b in windows) == 100
    assert windows[0][1] <= windows[1][0]


def test_sample_windows_degrades_to_one_window_when_the_tensor_is_short():
    assert M.sample_windows(10, 20) == [(0, 10)]
    assert M.sample_windows(10, 10) == [(0, 10)]


# ── discovery ─────────────────────────────────────────────────────────────────────────────────────
def test_discover_reads_the_real_merged_save_key_space():
    """The shapes this build actually writes: split GDN projections, doubled experts., shared experts."""
    keys = [
        f"{_LLM}0.self_attention.in_proj.weight.query",
        f"{_LLM}0.self_attention.in_proj.weight.key",
        f"{_LLM}0.self_attention.in_proj.weight.value",
        f"{_LLM}0.self_attention.in_proj.weight.z",
        f"{_LLM}0.self_attention.in_proj.weight.beta",
        f"{_LLM}0.self_attention.in_proj.weight.alpha",
        f"{_LLM}0.self_attention.conv1d.weight.query",
        f"{_LLM}0.self_attention.conv1d.weight.key",
        f"{_LLM}0.self_attention.conv1d.weight.value",
        f"{_LLM}0.mlp.router.weight",
        f"{_LLM}0.mlp.experts.experts.linear_fc1.weight0",
        f"{_LLM}0.mlp.experts.experts.linear_fc1.weight1",
        f"{_LLM}0.mlp.shared_experts.linear_fc1.weight",
        f"{_LLM}3.self_attention.linear_qkv.weight",
        f"{_LLM}3.mlp.router.weight",
        "vision_model.decoder.layers.0.self_attention.linear_qkv.weight",
    ]
    found = M.discover(keys)
    assert found["attention_layers"] == [3]
    assert found["gdn_layers"] == [0]
    assert found["moe_layers"] == [0, 3]
    assert found["vision_layers"] == [0]
    assert found["gdn_in_proj_parts"] == ["alpha", "beta", "key", "query", "value", "z"]
    assert found["gdn_conv1d_parts"] == ["key", "query", "value"]
    assert found["gdn_in_proj_fused"] is False
    assert found["expert_stem"] == "mlp.experts.experts.linear_fc1"
    assert found["expert_indexed"] is True
    assert found["expert_ids"] == [0, 1]
    assert found["shared_expert_stem"] == "mlp.shared_experts."
    assert found["has_vision_final_ln"] is False


def test_discover_still_handles_a_fused_gdn_projection():
    found = M.discover([f"{_LLM}1.self_attention.in_proj.weight"])
    assert found["gdn_layers"] == [1]
    assert found["gdn_in_proj_fused"] is True
    assert found["gdn_in_proj_parts"] == []


def test_discover_recognises_the_other_expert_storage_styles():
    single = M.discover([f"{_LLM}0.mlp.experts.linear_fc1.weight7"])
    assert single["expert_stem"] == "mlp.experts.linear_fc1" and single["expert_indexed"] is True
    stacked = M.discover([f"{_LLM}0.mlp.experts.linear_fc1.weight"])
    assert stacked["expert_stem"] == "mlp.experts.linear_fc1" and stacked["expert_indexed"] is False
    seq = M.discover([f"{_LLM}0.mlp.experts.local_experts.3.linear_fc1.weight"])
    assert "local_experts" in (seq["expert_stem"] or "") and seq["expert_ids"] == [3]


def test_expert_save_key_follows_the_discovered_style():
    key, index = M.expert_save_key(2, 5, "mlp.experts.experts.linear_fc1", True)
    assert key.endswith("mlp.experts.experts.linear_fc1.weight5") and index is None
    key, index = M.expert_save_key(2, 5, "mlp.experts.linear_fc1", False)
    assert key.endswith("mlp.experts.linear_fc1.weight") and index == 5
    key, index = M.expert_save_key(2, 5, "mlp.experts.local_experts.{e}.linear_fc1", False)
    assert key.endswith("local_experts.5.linear_fc1.weight") and index is None


def test_vision_final_layernorm_is_detected_when_present():
    assert M.discover(["vision_model.decoder.final_layernorm.weight"])["has_vision_final_ln"] is True


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
