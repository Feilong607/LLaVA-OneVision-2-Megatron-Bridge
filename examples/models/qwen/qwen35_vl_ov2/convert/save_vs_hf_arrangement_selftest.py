#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end self-test for ``save_vs_hf_arrangement.py``: synthetic SAVE + HF export, injected defects.

Runs anywhere torch + safetensors exist (CPU pod, export workspace) -- no GPU, no checkpoint, no model.
It builds a tiny but structurally faithful pair (attention layer, GDN layer, MoE experts, vision layer,
adapter), then re-runs the comparator against deliberately broken exports and asserts that each defect is
caught AND named. A checker that cannot demonstrate its own failure modes is not evidence.

    python3 save_vs_hf_arrangement_selftest.py            # all cases
    python3 save_vs_hf_arrangement_selftest.py --case qkv_contiguous
    python3 save_vs_hf_arrangement_selftest.py --sharded   # write the SAVE as 4 dim-0 shards (needs mcore)
    python3 save_vs_hf_arrangement_selftest.py --keep /tmp/sv   # keep the fixtures for inspection

Exit 0 = every case behaved as specified; 1 = a case did not; 2 = the environment is missing torch or
safetensors (nothing was proved -- do not read that as a pass).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


HERE = Path(__file__).resolve().parent
TOOL = HERE / "save_vs_hf_arrangement.py"

# Geometry: small, but every divisibility the real layouts depend on holds.
V, H = 32, 8                      # vocab, hidden
NH, NKV, HD = 4, 2, 2             # llm heads / kv heads / head dim  -> fused qkv rows = (4+2*2)*2 = 16
FFN, NE = 6, 2                    # expert ffn, experts             -> fused fc1 rows = 12
VH, VHEADS, VFFN = 6, 3, 10       # vision hidden / heads / ffn     -> fused vision qkv rows = 18
MERGER = 12
ATTN_LAYER, GDN_LAYER = 0, 1

_LLM = "language_model.decoder.layers."
_VIS = "vision_model.decoder.layers."
HFL = "model.language_model.layers."
HV = "model.visual."


def _t(*shape: int):
    import torch

    # Deterministic, distinct per shape+call so a mis-paired tensor cannot match by accident.
    _t.counter = getattr(_t, "counter", 0) + 1
    g = torch.Generator().manual_seed(1234 + _t.counter)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)


def build_save() -> Dict[str, "object"]:
    """The synthetic training SAVE, in mcore key space and mcore layouts."""
    sd = {
        "language_model.embedding.word_embeddings.weight": _t(V + 4, H),  # +4 = vocab padding
        "language_model.output_layer.weight": _t(V, H),
        "language_model.decoder.final_layernorm.weight": _t(H),
        # attention layer
        f"{_LLM}{ATTN_LAYER}.self_attention.linear_qkv.weight": _t((NH + 2 * NKV) * HD, H),
        f"{_LLM}{ATTN_LAYER}.self_attention.linear_proj.weight": _t(H, NH * HD),
        f"{_LLM}{ATTN_LAYER}.self_attention.q_layernorm.weight": _t(HD),
        f"{_LLM}{ATTN_LAYER}.self_attention.k_layernorm.weight": _t(HD),
        f"{_LLM}{ATTN_LAYER}.self_attention.linear_qkv.layer_norm_weight": _t(H),
        # GDN layer
        f"{_LLM}{GDN_LAYER}.self_attention.in_proj.weight": _t(10 + 4 + 2 + 2, H),
        f"{_LLM}{GDN_LAYER}.self_attention.conv1d.weight": _t(10, 1, 4),
        f"{_LLM}{GDN_LAYER}.self_attention.out_norm.weight": _t(4),
        f"{_LLM}{GDN_LAYER}.self_attention.out_proj.weight": _t(H, 4),
        f"{_LLM}{GDN_LAYER}.self_attention.A_log": _t(2),
        f"{_LLM}{GDN_LAYER}.self_attention.dt_bias": _t(2),
        f"{_LLM}{GDN_LAYER}.self_attention.in_proj.layer_norm_weight": _t(H),
        # vision + adapter
        f"{_VIS}0.self_attention.linear_qkv.weight": _t(3 * VHEADS * (VH // VHEADS), VH),
        f"{_VIS}0.self_attention.linear_proj.weight": _t(VH, VH),
        f"{_VIS}0.mlp.linear_fc1.weight": _t(VFFN, VH),
        f"{_VIS}0.mlp.linear_fc2.weight": _t(VH, VFFN),
        f"{_VIS}0.self_attention.linear_qkv.layer_norm_weight": _t(VH),
        "vision_model.pre_layernorm.weight": _t(VH),
        "adapter.layernorm.weight": _t(VH),
        "adapter.linear_fc1.weight": _t(MERGER, VH),
        "adapter.linear_fc2.weight": _t(H, MERGER),
    }
    for layer in (ATTN_LAYER, GDN_LAYER):
        sd[f"{_LLM}{layer}.mlp.router.weight"] = _t(NE, H)
        sd[f"{_LLM}{layer}.pre_mlp_layernorm.weight"] = _t(H)
        for e in range(NE):
            sd[f"{_LLM}{layer}.mlp.experts.linear_fc1.weight{e}"] = _t(2 * FFN, H)
            sd[f"{_LLM}{layer}.mlp.experts.linear_fc2.weight{e}"] = _t(H, FFN)
    return sd


def build_hf(save: Dict[str, "object"]) -> Dict[str, "object"]:
    """The CORRECT export derived from that SAVE -- the arrangement the comparator calls canonical."""
    import torch

    sys.path.insert(0, str(HERE))
    from save_vs_hf_arrangement import qkv_rows_grouped, qkv_rows_head_interleaved

    q, k, v = qkv_rows_grouped(NH, NKV, HD)
    fused = save[f"{_LLM}{ATTN_LAYER}.self_attention.linear_qkv.weight"]
    vq, vk, vv = qkv_rows_head_interleaved(VHEADS, VH // VHEADS)
    vfused = save[f"{_VIS}0.self_attention.linear_qkv.weight"]

    hf = {
        "model.language_model.embed_tokens.weight":
            save["language_model.embedding.word_embeddings.weight"][:V].clone(),
        "lm_head.weight": save["language_model.output_layer.weight"].clone(),
        "model.language_model.norm.weight": save["language_model.decoder.final_layernorm.weight"].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.q_proj.weight": fused[torch.as_tensor(q)].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.k_proj.weight": fused[torch.as_tensor(k)].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.v_proj.weight": fused[torch.as_tensor(v)].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.o_proj.weight":
            save[f"{_LLM}{ATTN_LAYER}.self_attention.linear_proj.weight"].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.q_norm.weight":
            save[f"{_LLM}{ATTN_LAYER}.self_attention.q_layernorm.weight"].clone(),
        f"{HFL}{ATTN_LAYER}.self_attn.k_norm.weight":
            save[f"{_LLM}{ATTN_LAYER}.self_attention.k_layernorm.weight"].clone(),
        f"{HFL}{ATTN_LAYER}.input_layernorm.weight":
            save[f"{_LLM}{ATTN_LAYER}.self_attention.linear_qkv.layer_norm_weight"].clone(),
        f"{HV}encoder.layers.0.self_attn.qkv.weight": vfused[torch.as_tensor(vq + vk + vv)].clone(),
        f"{HV}encoder.layers.0.self_attn.proj.weight":
            save[f"{_VIS}0.self_attention.linear_proj.weight"].clone(),
        f"{HV}encoder.layers.0.mlp.fc1.weight": save[f"{_VIS}0.mlp.linear_fc1.weight"].clone(),
        f"{HV}encoder.layers.0.mlp.fc2.weight": save[f"{_VIS}0.mlp.linear_fc2.weight"].clone(),
        f"{HV}encoder.layers.0.layer_norm1.weight":
            save[f"{_VIS}0.self_attention.linear_qkv.layer_norm_weight"].clone(),
        f"{HV}layernorm_pre.weight": save["vision_model.pre_layernorm.weight"].clone(),
        f"{HV}merger.ln_q.weight": save["adapter.layernorm.weight"].clone(),
        f"{HV}merger.mlp.0.weight": save["adapter.linear_fc1.weight"].clone(),
        f"{HV}merger.mlp.2.weight": save["adapter.linear_fc2.weight"].clone(),
    }
    # GDN: in_proj splits in the declared order; out_norm gains the +1 (zero-centred -> standard RMSNorm).
    g = save[f"{_LLM}{GDN_LAYER}.self_attention.in_proj.weight"]
    for name, (a, b) in zip(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
                            ((0, 10), (10, 14), (14, 16), (16, 18))):
        hf[f"{HFL}{GDN_LAYER}.linear_attn.{name}.weight"] = g[a:b].clone()
    hf[f"{HFL}{GDN_LAYER}.linear_attn.conv1d.weight"] = \
        save[f"{_LLM}{GDN_LAYER}.self_attention.conv1d.weight"].clone()
    hf[f"{HFL}{GDN_LAYER}.linear_attn.norm.weight"] = \
        (save[f"{_LLM}{GDN_LAYER}.self_attention.out_norm.weight"].float() + 1.0).to(torch.bfloat16)
    hf[f"{HFL}{GDN_LAYER}.linear_attn.out_proj.weight"] = \
        save[f"{_LLM}{GDN_LAYER}.self_attention.out_proj.weight"].clone()
    hf[f"{HFL}{GDN_LAYER}.linear_attn.A_log"] = save[f"{_LLM}{GDN_LAYER}.self_attention.A_log"].clone()
    hf[f"{HFL}{GDN_LAYER}.linear_attn.dt_bias"] = save[f"{_LLM}{GDN_LAYER}.self_attention.dt_bias"].clone()
    hf[f"{HFL}{GDN_LAYER}.input_layernorm.weight"] = \
        save[f"{_LLM}{GDN_LAYER}.self_attention.in_proj.layer_norm_weight"].clone()
    for layer in (ATTN_LAYER, GDN_LAYER):
        hf[f"{HFL}{layer}.mlp.gate.weight"] = save[f"{_LLM}{layer}.mlp.router.weight"].clone()
        hf[f"{HFL}{layer}.post_attention_layernorm.weight"] = \
            save[f"{_LLM}{layer}.pre_mlp_layernorm.weight"].clone()
        hf[f"{HFL}{layer}.mlp.experts.gate_up_proj"] = torch.stack(
            [save[f"{_LLM}{layer}.mlp.experts.linear_fc1.weight{e}"] for e in range(NE)])
        hf[f"{HFL}{layer}.mlp.experts.down_proj"] = torch.stack(
            [save[f"{_LLM}{layer}.mlp.experts.linear_fc2.weight{e}"] for e in range(NE)])
    return hf


def write_save(sd: Dict[str, "object"], out: Path, shards: int = 1) -> None:
    """Write the synthetic SAVE as a torch_dist checkpoint.

    ``shards>1`` writes each 2-D tensor as that many dim-0 chunks through mcore's ShardedTensor -- the same
    shape a TP-sharded save has on disk -- so the comparator's global assembly is exercised, not assumed.
    """
    import torch.distributed.checkpoint as dcp

    out.mkdir(parents=True, exist_ok=True)
    if shards <= 1:
        dcp.save(dict(sd), storage_writer=dcp.FileSystemWriter(str(out)), no_dist=True)
        return

    import torch.distributed as dist
    from megatron.core import dist_checkpointing as mdc
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    if not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    sharded = {}
    for key, tensor in sd.items():
        n = shards if (tensor.ndim >= 1 and int(tensor.shape[0]) % shards == 0) else 1
        if n == 1:
            sharded[key] = ShardedTensor.from_rank_offsets(key, tensor.clone(), replica_id=0)
            continue
        piece = int(tensor.shape[0]) // n
        for i in range(n):
            chunk = tensor[i * piece : (i + 1) * piece].clone()
            sharded[f"{key}#chunk{i}"] = ShardedTensor.from_rank_offsets(key, chunk, (0, i, n), replica_id=0)
    mdc.save(sharded, str(out))


def write_hf(tensors: Dict[str, "object"], out: Path, *, vision_post_layernorm: bool = False) -> None:
    """Write the synthetic export: one safetensors file plus the config.json the tool reads."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({
        "model_type": "llava_onevision2_moe",
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_moe_text", "hidden_size": H, "num_attention_heads": NH,
            "num_key_value_heads": NKV, "head_dim": HD, "num_experts": NE, "mtp_num_hidden_layers": 0,
        },
        "vision_config": {
            "hidden_size": VH, "num_attention_heads": VHEADS, "use_post_layernorm": vision_post_layernorm,
        },
    }, indent=2))


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Defect injectors: each returns a mutated copy of the correct export.
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def defect_qkv_contiguous(hf, save):
    """Defect: q/k/v split off the fused matrix as three contiguous blocks instead of per KV group."""
    import torch

    sys.path.insert(0, str(HERE))
    from save_vs_hf_arrangement import qkv_rows_contiguous

    q, k, v = qkv_rows_contiguous(NH, NKV, HD)
    fused = save[f"{_LLM}{ATTN_LAYER}.self_attention.linear_qkv.weight"]
    hf = dict(hf)
    hf[f"{HFL}{ATTN_LAYER}.self_attn.q_proj.weight"] = fused[torch.as_tensor(q)].clone()
    hf[f"{HFL}{ATTN_LAYER}.self_attn.k_proj.weight"] = fused[torch.as_tensor(k)].clone()
    hf[f"{HFL}{ATTN_LAYER}.self_attn.v_proj.weight"] = fused[torch.as_tensor(v)].clone()
    return hf


def defect_gate_up_interleaved(hf, save):
    """Defect: one expert's gate/up shipped in TP-rank-interleaved order (the SwiGLU merge trap)."""
    import torch

    sys.path.insert(0, str(HERE))
    from save_vs_hf_arrangement import deinterleave_rows

    idx = torch.as_tensor(deinterleave_rows(2 * FFN, 2), dtype=torch.long)
    hf = dict(hf)
    # Only expert 0 is interleaved: the remaining experts must stay PASS, which is what proves the
    # comparator localises a defect instead of flagging the whole family.
    hf[f"{HFL}{ATTN_LAYER}.mlp.experts.gate_up_proj"] = torch.stack(
        [save[f"{_LLM}{ATTN_LAYER}.mlp.experts.linear_fc1.weight{e}"][idx] if e == 0
         else save[f"{_LLM}{ATTN_LAYER}.mlp.experts.linear_fc1.weight{e}"] for e in range(NE)])
    return hf


def defect_out_norm_no_offset(hf, save):
    """Defect: the GDN out-norm exported without the +1 that turns zero-centred gamma into HF's."""
    hf = dict(hf)
    hf[f"{HFL}{GDN_LAYER}.linear_attn.norm.weight"] = \
        save[f"{_LLM}{GDN_LAYER}.self_attention.out_norm.weight"].clone()
    return hf


def defect_expert_reinitialised(hf, save):
    """Defect: one expert matrix replaced by noise -- the "loads fine, is not the trained model" case."""
    import torch

    hf = dict(hf)
    victim = hf[f"{HFL}{ATTN_LAYER}.mlp.experts.down_proj"].clone()
    g = torch.Generator().manual_seed(99)
    victim[0] = torch.randn(victim.shape[1], victim.shape[2], generator=g).to(victim.dtype)
    hf[f"{HFL}{ATTN_LAYER}.mlp.experts.down_proj"] = victim
    return hf


def defect_gdn_parts_swapped(hf, save):
    """Defect: two GDN in_proj parts concatenated in the wrong order."""
    hf = dict(hf)
    a = f"{HFL}{GDN_LAYER}.linear_attn.in_proj_b.weight"
    b = f"{HFL}{GDN_LAYER}.linear_attn.in_proj_a.weight"
    hf[a], hf[b] = hf[b].clone(), hf[a].clone()
    return hf


def defect_vision_post_ln_present(hf, save):
    """Defect: the export carries a vision final LayerNorm the SAVE never trained."""
    import torch

    hf = dict(hf)
    g = torch.Generator().manual_seed(7)
    hf[f"{HV}layernorm_post.weight"] = torch.randn(VH, generator=g).to(torch.bfloat16)
    return hf


CASES: List[Tuple[str, object, str, str, int]] = [
    # (name, injector or None, family that must flip, expected arrangement substring, expected exit code)
    ("clean", None, "", "", 0),
    ("qkv_contiguous", defect_qkv_contiguous, f"attn{ATTN_LAYER}_qkv", "contiguous_q_k_v", 1),
    ("gate_up_interleaved", defect_gate_up_interleaved, f"moe{ATTN_LAYER}_e0_gate_up",
     "rank_interleaved_gate_up", 1),
    ("out_norm_no_offset", defect_out_norm_no_offset, f"gdn{GDN_LAYER}_out_norm", "identity_no_offset", 1),
    ("expert_reinitialised", defect_expert_reinitialised, f"moe{ATTN_LAYER}_e0_down", "none", 1),
    ("gdn_parts_swapped", defect_gdn_parts_swapped, f"gdn{GDN_LAYER}_in_proj", "concat[", 1),
    ("vision_post_ln_present", defect_vision_post_ln_present, "vision_final_ln", "present", 1),
]


def run_tool(save_dir: Path, hf_dir: Path, report: Path) -> Tuple[int, dict]:
    """Run the comparator as a subprocess and return (exit code, parsed report)."""
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--save", str(save_dir), "--hf", str(hf_dir),
         "--layers", f"{ATTN_LAYER},{GDN_LAYER}", "--expert", "0,1", "--vision-layers", "0",
         "--json", str(report)],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:] if proc.stdout else "")
    if proc.returncode not in (0, 1, 3):
        sys.stderr.write(proc.stderr[-4000:])
    data = json.loads(report.read_text()) if report.is_file() else {}
    return proc.returncode, data


def main() -> int:
    """Build the fixtures, run every case, and check each defect was caught AND named."""
    ap = argparse.ArgumentParser(description="Self-test for save_vs_hf_arrangement.py")
    ap.add_argument("--case", default="", help="run one case by name (default: all)")
    ap.add_argument("--sharded", action="store_true",
                    help="write the SAVE as 4 dim-0 shards via mcore (exercises global assembly)")
    ap.add_argument("--keep", type=Path, default=None, help="keep fixtures in this directory")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: environment lacks torch/safetensors ({exc!r}); nothing was proved", file=sys.stderr)
        return 2

    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="save-vs-hf-selftest."))
    root.mkdir(parents=True, exist_ok=True)
    print(f"[selftest] fixtures under {root} (sharded={args.sharded})")

    save_sd = build_save()
    save_dir = root / "save" / "iter_0000005"
    try:
        write_save(save_sd, save_dir, shards=4 if args.sharded else 1)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not write the synthetic SAVE: {exc!r}", file=sys.stderr)
        return 1
    correct = build_hf(save_sd)

    failures: List[str] = []
    for name, injector, family, arrangement, want_rc in CASES:
        if args.case and args.case != name:
            continue
        hf_dir = root / f"hf_{name}"
        tensors = correct if injector is None else injector(correct, save_sd)
        write_hf(tensors, hf_dir, vision_post_layernorm=(name == "vision_post_ln_present"))
        rc, report = run_tool(save_dir, hf_dir, root / f"report_{name}.json")
        rows = {r["family"]: r for r in report.get("results", [])}
        print(f"[selftest] case={name} rc={rc} counts={report.get('counts')}")

        if rc != want_rc:
            failures.append(f"{name}: exit {rc}, expected {want_rc}")
        if name == "clean":
            bad = [f"{k}={v['status']}" for k, v in rows.items() if v["status"] not in ("PASS", "SKIP")]
            if bad:
                failures.append(f"clean: these families did not pass: {bad}")
            continue
        row = rows.get(family)
        if row is None:
            failures.append(f"{name}: family {family} missing from the report")
        elif row["status"] != "FAIL":
            failures.append(f"{name}: {family} was {row['status']}, expected FAIL")
        elif arrangement not in (row["arrangement"] or ""):
            failures.append(f"{name}: {family} named arrangement '{row['arrangement']}', "
                            f"expected one containing '{arrangement}'")
        # Only the injected family may flip: a checker that fails everything proves nothing.
        collateral = [k for k, v in rows.items() if v["status"] == "FAIL" and k != family]
        if collateral:
            failures.append(f"{name}: collateral FAILs on {collateral}")

    if args.keep is None:
        shutil.rmtree(root, ignore_errors=True)

    print("")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("[selftest] OK: every case behaved as specified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
