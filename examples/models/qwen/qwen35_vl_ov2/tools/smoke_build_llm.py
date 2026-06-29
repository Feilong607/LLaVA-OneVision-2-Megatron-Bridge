#!/usr/bin/env python3
"""GATE smoke: does the Qwen3.5-35B-A3B *text* LLM (GatedDeltaNet hybrid + 256-expert MoE + MTP)
build in this mcore via AutoBridge? Runs against the text-only HF dir produced by
extract_qwen35_text.py. qwen3.5-only / self-contained -- imports nothing from the qwen3 (30B) code.

Stage A (no GPU/dist needed): AutoBridge.from_hf_pretrained + to_megatron_provider(load_weights=False),
print the provider fields that prove the bridge READ qwen3.5 correctly (GDN variant, 256 experts, MTP).
Stage B (best-effort, needs torchrun): structure-only provide() at TP1/EP1/PP1, cpu-init, no real init.
"""
import os, sys, traceback

SRC = os.environ.get("TEXT_DIR", "/ov2/pretrain_models/Qwen3.5-35B-A3B-text")
print("=== Qwen3.5-35B-A3B text-LLM build smoke ===\nsrc =", SRC, flush=True)

from megatron.bridge import AutoBridge
bridge = AutoBridge.from_hf_pretrained(SRC)
print("[A] from_hf_pretrained OK", flush=True)
prov = bridge.to_megatron_provider(load_weights=False)
print("[A] to_megatron_provider OK -> provider fields:", flush=True)
for f in ["num_layers", "hidden_size", "num_attention_heads", "num_query_groups", "kv_channels",
          "num_moe_experts", "moe_router_topk", "moe_ffn_hidden_size", "moe_shared_expert_intermediate_size",
          "experimental_attention_variant", "linear_attention_freq", "linear_num_key_heads",
          "linear_num_value_heads", "linear_key_head_dim", "mtp_num_layers", "vocab_size"]:
    print("    %-42s = %r" % (f, getattr(prov, f, "<absent>")), flush=True)

# Stage B: structure-only provide() (best-effort; needs torch.distributed)
try:
    import torch, torch.distributed as dist
    import megatron.core.parallel_state as mpu
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    mpu.initialize_model_parallel(1, 1)
    prov.tensor_model_parallel_size = 1
    prov.pipeline_model_parallel_size = 1
    for k, v in [("expert_model_parallel_size", 1), ("use_cpu_initialization", True),
                 ("moe_router_dtype", "fp32"), ("gradient_accumulation_fusion", False)]:
        if hasattr(prov, k):
            setattr(prov, k, v)
    # mirror llava_ov2._fill_init: AutoBridge(load_weights=False) leaves init_method* = None,
    # which trips not_none() when building MoE experts / embedding. Fill them, then no-init.
    from megatron.core.utils import init_method_normal, scaled_init_method_normal
    _std = getattr(prov, "init_method_std", 0.02) or 0.02
    if getattr(prov, "init_method", None) is None:
        prov.init_method = init_method_normal(_std)
    if getattr(prov, "output_layer_init_method", None) is None:
        prov.output_layer_init_method = scaled_init_method_normal(_std, prov.num_layers)
    for _n in dir(prov):
        if _n.endswith("_init_method") and _n != "output_layer_init_method" and getattr(prov, _n, None) is None:
            try:
                setattr(prov, _n, prov.init_method)
            except Exception:
                pass
    if hasattr(prov, "perform_initialization"):
        prov.perform_initialization = False
    m = prov.provide(pre_process=True, post_process=True)
    n = sum(p.numel() for p in m.parameters())
    print("[B] provide() OK -> %s, params=%.2fB" % (type(m).__name__, n / 1e9), flush=True)
    # confirm the hybrid attention + MTP actually instantiated
    kinds = {}
    for nm, mod in m.named_modules():
        t = type(mod).__name__
        if any(s in t for s in ("DeltaNet", "Attention", "MTP", "MultiToken", "Mamba", "Linear_attn")):
            kinds[t] = kinds.get(t, 0) + 1
    print("[B] attn/MTP module types:", kinds, flush=True)
    print("=== SMOKE PASS ===", flush=True)
except Exception as e:
    traceback.print_exc()
    print("[B] provide() FAILED:", repr(e), flush=True)
    print("=== STAGE-A PASSED, STAGE-B FAILED (see trace) ===", flush=True)
