#!/usr/bin/env python3
"""Definitive audit: does qwen35_bridge.load_hf_weights load ALL the Qwen3.5 text-LLM params?
Build the model, fill every param with a NaN sentinel, run the bridge HF-weight load, then any param
still all-NaN = NOT loaded = random in the frozen LLM (the suspected cause of the stage-1 loss plateau).
CPU/EP1 — does NOT use the training GPUs (only fla's import-time triton probe touches a GPU)."""
import os, re, sys
from collections import Counter
import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)  # CPU, single-rank
import megatron.core.parallel_state as mpu
if not mpu.model_parallel_is_initialized():
    mpu.initialize_model_parallel(1, 1)  # TP1 PP1 EP1 (all experts on this rank)

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
from megatron.core.utils import init_method_normal, scaled_init_method_normal

p = _ov2_backbone_paths("qwen3.5-35b-a3b")
SRC = p["llm_hf"]
print("[audit] llm_hf =", SRC, flush=True)
bridge = AutoBridge.from_hf_pretrained(SRC)
prov = bridge.to_megatron_provider(load_weights=False)
prov.tensor_model_parallel_size = 1
prov.pipeline_model_parallel_size = 1
if hasattr(prov, "expert_model_parallel_size"): prov.expert_model_parallel_size = 1
if hasattr(prov, "use_cpu_initialization"): prov.use_cpu_initialization = True
if hasattr(prov, "moe_router_dtype"): prov.moe_router_dtype = "fp32"
if hasattr(prov, "gradient_accumulation_fusion"): prov.gradient_accumulation_fusion = False
_std = getattr(prov, "init_method_std", 0.02) or 0.02
if getattr(prov, "init_method", None) is None: prov.init_method = init_method_normal(_std)
if getattr(prov, "output_layer_init_method", None) is None: prov.output_layer_init_method = scaled_init_method_normal(_std, prov.num_layers)
for _n in dir(prov):
    if _n.endswith("_init_method") and _n != "output_layer_init_method" and getattr(prov, _n, None) is None:
        try: setattr(prov, _n, prov.init_method)
        except Exception: pass
if hasattr(prov, "perform_initialization"): prov.perform_initialization = False
m = prov.provide(pre_process=True, post_process=True)
print("[audit] model built (%.2fB params)" % (sum(x.numel() for x in m.parameters())/1e9), flush=True)

with torch.no_grad():
    for prm in m.parameters():
        prm.fill_(float("nan"))
print("[audit] filled NaN sentinel; loading HF weights ...", flush=True)
bridge.load_hf_weights([m])
print("[audit] load_hf_weights done; scanning ...", flush=True)

total = loaded = 0
unloaded, partial = [], []
for name, prm in m.named_parameters():
    total += 1
    n = torch.isnan(prm)
    if bool(n.all()):
        unloaded.append((name, tuple(prm.shape)))
    elif bool(n.any()):
        partial.append((name, tuple(prm.shape)))
    else:
        loaded += 1
print("\n[audit] TOTAL=%d  LOADED=%d  UNLOADED(all-NaN)=%d  PARTIAL(some-NaN)=%d" % (
    total, loaded, len(unloaded), len(partial)), flush=True)

def fam(names):
    c = Counter()
    for nm, _ in names:
        c[re.sub(r"\.\d+\.", ".*.", nm)] += 1
    return c
if unloaded:
    print("=== UNLOADED families (RANDOM in frozen LLM) ===", flush=True)
    for k, c in sorted(fam(unloaded).items()): print("   x%-4d %s" % (c, k), flush=True)
if partial:
    print("=== PARTIAL families ===", flush=True)
    for k, c in sorted(fam(partial).items()): print("   x%-4d %s" % (c, k), flush=True)
print("\n[audit] VERDICT:", "ALL LOADED -- LLM weights are complete (plateau is NOT a load gap)" if not unloaded and not partial
      else "INCOMPLETE LOAD -- frozen LLM has random params (likely the plateau cause)", flush=True)
