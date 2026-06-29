import megatron.bridge.recipes as r
from megatron.bridge.recipes.ov2.ov2 import _OV2_BACKBONES
print("backbone qwen3.5 mrope_section =", _OV2_BACKBONES["qwen3.5-35b-a3b"].get("mrope_section"), flush=True)
print("backbone qwen3-4b  mrope_section =", _OV2_BACKBONES["qwen3-4b"].get("mrope_section", "<absent=None>"), flush=True)
print("backbone qwen3-30b mrope_section =", _OV2_BACKBONES["qwen3-30b-a3b"].get("mrope_section", "<absent=None>"), flush=True)
cfg = r.ov2_qwen35_35b_a3b_stage1()
print("qwen35 cfg.model.mrope_section =", getattr(cfg.model, "mrope_section", "MISSING"), flush=True)
cfg30 = r.ov2_30b_a3b_p16m33_stage1()
print("30B   cfg.model.mrope_section =", getattr(cfg30.model, "mrope_section", "MISSING"), flush=True)
print("=== MROPE WIRING CHECK DONE ===", flush=True)
