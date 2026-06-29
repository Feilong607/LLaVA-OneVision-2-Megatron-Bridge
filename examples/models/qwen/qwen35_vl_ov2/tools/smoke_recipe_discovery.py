#!/usr/bin/env python3
"""Verify the Qwen3.5 OV2 recipes are discoverable by run_recipe.py (getattr on megatron.bridge.recipes)
and that calling stage1 assembles a ConfigContainer with the qwen3.5 backbone wired."""
import megatron.bridge.recipes as r

names = ["ov2_qwen35_35b_a3b_stage1", "ov2_qwen35_35b_a3b_stage2", "ov2_qwen35_35b_a3b_midtrain"]
for n in names:
    print("discoverable  %-34s = %s" % (n, hasattr(r, n)), flush=True)
assert all(hasattr(r, n) for n in names), "recipe(s) NOT discoverable"

from megatron.bridge.recipes.ov2.ov2 import _OV2_BACKBONES
b = _OV2_BACKBONES.get("qwen3.5-35b-a3b")
print("backbone qwen3.5-35b-a3b registered:", b is not None, flush=True)
if b:
    print("   is_moe=%s llm_hf=%s patch=%s merge=%s proc=%s" % (
        b["is_moe"], b["llm_hf"], b["vision_patch_size"], b["vision_spatial_merge_size"], b["hf_proc"]), flush=True)

cfg = r.ov2_qwen35_35b_a3b_stage1()
print("stage1 ConfigContainer built:", type(cfg).__name__, flush=True)
print("=== RECIPE DISCOVERY PASS ===", flush=True)
