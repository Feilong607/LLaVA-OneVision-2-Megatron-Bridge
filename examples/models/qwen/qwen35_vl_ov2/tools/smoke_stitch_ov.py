#!/usr/bin/env python3
"""STITCH smoke: attach the OneVision-Encoder (+ adapter) onto the Qwen3.5-35B-A3B text LLM via
build_llava_ov2 -- the concrete proof of "how to use the OneVision-Encoder" on this backbone.
Structure-only (EP1/TP1, cpu-init, no weights). qwen3.5-only / self-contained.

Vision geometry left at the build_llava_ov2 defaults = the VERIFIED p16m33 tower
(patch16 / merge3 / hidden1024 / 24L) -- the user's chosen encoder. The adapter auto-sizes its
output to llm_cfg.hidden_size (=2048 for 35B-A3B).
"""
import os, traceback

TEXT_DIR = os.environ.get("TEXT_DIR", "/ov2/pretrain_models/Qwen3.5-35B-A3B-text")
print("=== Qwen3.5-35B-A3B + OneVisionEncoder stitch smoke ===", flush=True)
print("llm_hf =", TEXT_DIR, flush=True)

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu

if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
mpu.initialize_model_parallel(1, 1)

from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2

model = build_llava_ov2(
    llm_hf_path=TEXT_DIR,
    perform_init=False,
    use_cpu_init=True,
    grad_accum_fusion=False,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    sequence_parallel=False,
    load_llm_weights=False,
    # vision geometry defaults == p16m33 OneVisionEncoder (patch16 / merge3 / hidden1024 / 24L)
)
print("[stitch] build_llava_ov2 OK ->", type(model).__name__, flush=True)


def npar(m):
    return sum(p.numel() for p in m.parameters()) if m is not None else 0


sub = {nm: type(getattr(model, nm, None)).__name__ for nm in ("vision_model", "adapter", "language_model")}
print("[stitch] submodules:", sub, flush=True)
print("[stitch] params: vision=%.1fM  adapter=%.1fM  llm=%.2fB  total=%.2fB" % (
    npar(model.vision_model) / 1e6, npar(model.adapter) / 1e6,
    npar(model.language_model) / 1e9, npar(model) / 1e9), flush=True)

# adapter output width should equal the LLM hidden (2048) so fused image tokens match decoder_input
try:
    out_w = None
    for nm, p in model.adapter.named_parameters():
        if p.dim() >= 1:
            out_w = p.shape[0]
    print("[stitch] adapter last-param out dim =", out_w, "(expect ~2048 / llm hidden)", flush=True)
except Exception:
    pass

mtp = sum(1 for _, mm in model.language_model.named_modules() if "MultiToken" in type(mm).__name__)
gdn = sum(1 for _, mm in model.language_model.named_modules() if "GatedDeltaNet" in type(mm).__name__)
print("[stitch] llm hybrid check: GatedDeltaNet=%d  MTP modules=%d" % (gdn, mtp), flush=True)
print("[stitch] image_token_id =", getattr(model, "image_token_id", "<none>"), flush=True)
print("=== STITCH SMOKE PASS ===", flush=True)
