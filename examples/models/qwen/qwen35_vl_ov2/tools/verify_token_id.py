import inspect
from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2
print("build_llava_ov2 accepts image_token_id:", "image_token_id" in inspect.signature(build_llava_ov2).parameters, flush=True)

import megatron.bridge.recipes.ov2.ov2_qwen35  # noqa: registers the backbone
from megatron.bridge.recipes.ov2.ov2 import _OV2_BACKBONES
print("qwen3.5-35b-a3b image_token_id =", _OV2_BACKBONES["qwen3.5-35b-a3b"].get("image_token_id"), flush=True)
print("qwen3-4b image_token_id        =", _OV2_BACKBONES["qwen3-4b"].get("image_token_id", "<absent -> default 151655>"), flush=True)
print("qwen3-30b-a3b image_token_id   =", _OV2_BACKBONES["qwen3-30b-a3b"].get("image_token_id", "<absent -> default 151655>"), flush=True)

from transformers import AutoTokenizer
def show(name, path):
    try:
        t = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        print("%-26s image_pad=%s vis_start=%s vis_end=%s" % (
            name, t.convert_tokens_to_ids("<|image_pad|>"),
            t.convert_tokens_to_ids("<|vision_start|>"), t.convert_tokens_to_ids("<|vision_end|>")), flush=True)
    except Exception as e:
        print("%-26s tokenizer load failed: %r" % (name, e), flush=True)

show("Qwen3.5 (-text)", "/ov2/pretrain_models/Qwen3.5-35B-A3B-text")
show("Qwen2.5-VL 30B proc", "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model")
# the ACTUAL processor task_encoder loads (hf_proc) -> its tokenizer is what IMG_PAD_ID resolves from.
# This is the model<->data agreement proof (must equal the model image_token_id 248056).
show("qwen35 hf_proc (auto_model)", "/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model")
print("=== EXPECT: qwen35 -text AND hf_proc both 248056/248053/248054 ; 30B 151655/151652/151653 ===", flush=True)
