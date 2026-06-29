#!/usr/bin/env python3
"""Extract a text-only Qwen3_5MoeForCausalLM HF dir from the Qwen3.5-35B-A3B VLM dir.

WHY: /ov2/pretrain_models/Qwen3.5-35B-A3B on disk is `Qwen3_5MoeForConditionalGeneration`
(a VLM: `model.language_model.*` text + `model.visual.*` native vision + `mtp.*` + `lm_head`).
The OV2 OneVision-Encoder path needs the TEXT model so that:
  - AutoBridge.from_hf_pretrained() routes to the `qwen3_5_moe_text` bridge (qwen35_bridge.py),
    NOT the `qwen3_5_moe` VLM bridge (which would rebuild Qwen's native vision tower), and
  - build_llava_ov2() stitches the OneVisionEncoder + adapter in place of that native tower.

This writes a text-only HF dir:
  - config.json : text_config promoted to top level (its model_type is already
                  `qwen3_5_moe_text`), architectures=["Qwen3_5MoeForCausalLM"];
                  vision_config + the image/video token ids are dropped (text model doesn't use them).
  - tokenizer   : copied verbatim.
  - --weights   : safetensors re-keyed  model.language_model.* -> model.* ;
                  model.visual.* dropped ; lm_head.weight + mtp.* kept ; index rebuilt.

NO GPU. Config-only by DEFAULT (fast; enough for the AutoBridge structure-build smoke).
Pass --weights for the full ~70GB re-key (needed for real weight loading / training).

This file is qwen3.5-only and self-contained -- it does NOT import or modify any qwen3 (30B) code.
"""
import argparse
import glob
import json
import os
import shutil

# tokenizer / aux files copied verbatim if present
_AUX_FILES = [
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "generation_config.json",
]

_LM_PREFIX = "model.language_model."
_VISION_PREFIX = "model.visual."


def remap_key(k):
    """HF key -> text-model key, or None to drop."""
    if k.startswith(_VISION_PREFIX):
        return None                                  # native Qwen vision tower -> OneVisionEncoder replaces it
    if k.startswith(_LM_PREFIX):
        return "model." + k[len(_LM_PREFIX):]        # model.language_model.X -> model.X
    return k                                         # lm_head.weight, mtp.* -> keep as-is


def write_config(src, dst):
    cfg = json.load(open(os.path.join(src, "config.json")))
    assert "text_config" in cfg, "expected a VLM config with a text_config block"
    tc = dict(cfg["text_config"])                    # promote verbatim (self-contained text-model config)
    tc["architectures"] = ["Qwen3_5MoeForCausalLM"]
    tc.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    tc.setdefault("torch_dtype", tc.get("dtype", cfg.get("torch_dtype", "bfloat16")))
    if "transformers_version" in cfg:
        tc["transformers_version"] = cfg["transformers_version"]
    json.dump(tc, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print("[config] model_type=%s arch=%s layers=%s experts=%s hidden=%s mtp_layers=%s vocab=%s" % (
        tc.get("model_type"), tc["architectures"], tc.get("num_hidden_layers"),
        tc.get("num_experts"), tc.get("hidden_size"), tc.get("mtp_num_hidden_layers"),
        tc.get("vocab_size")))


def copy_tokenizer(src, dst):
    n = 0
    for f in _AUX_FILES:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, f))
            n += 1
    print("[tokenizer] copied %d aux files" % n)


def remap_weights(src, dst):
    from safetensors import safe_open
    from safetensors.torch import save_file
    idxs = glob.glob(os.path.join(src, "*.index.json"))
    assert idxs, "no *.index.json in %s" % src
    wm = json.load(open(idxs[0]))["weight_map"]
    by_shard = {}
    for k, sf in wm.items():
        by_shard.setdefault(sf, []).append(k)
    new_wm, kept, dropped = {}, 0, 0
    for sf, keys in sorted(by_shard.items()):
        tensors = {}
        with safe_open(os.path.join(src, sf), framework="pt") as f:
            for k in keys:
                nk = remap_key(k)
                if nk is None:
                    dropped += 1
                    continue
                tensors[nk] = f.get_tensor(k)
                kept += 1
        if not tensors:
            print("[weights] %s -> (all-vision, skipped)" % sf)
            continue
        save_file(tensors, os.path.join(dst, sf), metadata={"format": "pt"})
        for nk in tensors:
            new_wm[nk] = sf
        print("[weights] %s -> %d tensors" % (sf, len(tensors)))
    json.dump({"metadata": {}, "weight_map": new_wm},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    print("[weights] kept=%d dropped(vision)=%d shards_with_text=%d" % (kept, dropped, len(set(new_wm.values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/ov2/pretrain_models/Qwen3.5-35B-A3B")
    ap.add_argument("--dst", default="/ov2/pretrain_models/Qwen3.5-35B-A3B-text")
    ap.add_argument("--weights", action="store_true", help="also re-key the ~70GB safetensors (needed for training)")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    print("src=%s\ndst=%s\nweights=%s" % (args.src, args.dst, args.weights))
    write_config(args.src, args.dst)
    copy_tokenizer(args.src, args.dst)
    if args.weights:
        remap_weights(args.src, args.dst)
    else:
        print("[weights] SKIPPED (config-only). Re-run with --weights for the full re-key.")
    print("DONE ->", args.dst)


if __name__ == "__main__":
    main()
