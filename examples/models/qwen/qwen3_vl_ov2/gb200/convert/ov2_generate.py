#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OV2 (LLaVA-OneVision-2) VLM generation from a Bridge torch_dist checkpoint.

⚠️  NEW / UNVALIDATED CODE — read before first run.
    OV2 has NO HF model class and NO `.generate()`: `llava_ov2.LlavaOnevision2` exposes only the
    TRAINING `forward(images, image_grid_thw, input_ids, position_ids=None, ..., labels=None, ...)`.
    This driver wraps that forward in a GREEDY decode loop with FULL recompute per step (no KV cache
    -> O(n^2); fine for short smoke outputs, slow for long generations). It cannot ride
    examples/conversion/hf_to_megatron_generate_vlm.py because that needs
    `AutoBridge.from_hf_pretrained(<full VLM>)`, which OV2 has nothing to dispatch for.

    Validate on first run — the make-or-break is the processor->forward image plumbing:
      * `images=pixel_values` (bf16) + `image_grid_thw` must be exactly what the OV2 vision tower
        was trained on (it IS — we reuse the SAME hf_proc the Energon task encoder uses).
      * the processor's inserted image-token id must equal the model's `self.image_token_id`
        (IMAGE_TOKEN_ID, 248056). If the hf_proc tokenizer uses a different placeholder, the splice
        in llava_ov2.forward (`input_ids == self.image_token_id`) finds 0 slots and asserts.
      * logits are returned only when `labels=None`; at TP>1 the output is vocab-parallel — use TP1
        (the verified layout) for this driver unless you add a vocab all-gather. PP=1 / CP=1 only.

    Verified OV2 layout: TP1 / PP1 / EP8. Launch via inference.sh (sets the torchrun world).
"""
import argparse
import os

import torch
import torch.distributed as dist


def _resolve_iter_dir(path):
    """Accept a Bridge ckpt root (latest_checkpointed_iteration.txt) or a direct dist-ckpt dir."""
    if os.path.isfile(os.path.join(path, ".metadata")):
        return path
    tag = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tag):
        with open(tag) as f:
            it = f.read().strip()
        cand = os.path.join(path, "iter_{:07d}".format(int(it)))
        if os.path.isfile(os.path.join(cand, ".metadata")):
            return cand
    if os.path.isdir(path):
        for d in sorted((d for d in os.listdir(path) if d.startswith("iter_")), reverse=True):
            if os.path.isfile(os.path.join(path, d, ".metadata")):
                return os.path.join(path, d)
    return path


def main():
    ap = argparse.ArgumentParser(description="OV2 VLM greedy generation from a torch_dist ckpt.")
    ap.add_argument("--backbone", default="qwen3-30b-a3b-p16m33")
    ap.add_argument("--megatron_ckpt", required=True, help="Bridge torch_dist OV2 ckpt (root or iter_/ dir)")
    ap.add_argument("--image", default=None, help="image path or URL (omit for text-only)")
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--etp", type=int, default=1)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank = dist.get_rank()

    def log(m):
        if rank == 0:
            print("[ov2-gen] {}".format(m), flush=True)

    from megatron.core import dist_checkpointing, parallel_state
    if args.tp != 1 and rank == 0:
        log("WARNING: TP>1 returns vocab-parallel logits; this greedy driver assumes gathered logits "
            "(TP1). Add a vocab all-gather before argmax for TP>1, or run TP1 (the verified layout).")
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=1,                 # OV2: PP=1 only
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=(args.etp or None),
        )

    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2
    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    p = _ov2_backbone_paths(args.backbone)

    # 1) structure-only OV2 at the target parallelism, then load weights (torch_dist auto-reshards).
    log("build {} (TP{}/EP{}) + load {}".format(args.backbone, args.tp, args.ep, args.megatron_ckpt))
    model = build_llava_ov2(
        llm_hf_path=p["llm_hf"],
        perform_init=False,
        load_llm_weights=False,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=(args.etp or None),
        sequence_parallel=False,
        patch_size=p["vision_patch_size"],
        spatial_merge_size=p["vision_spatial_merge_size"],
        vision_hidden_size=p["vision_hidden_size"],
        vision_num_layers=p["vision_num_layers"],
        vision_model_name=p["vision_model_name"],
    )
    src = _resolve_iter_dir(args.megatron_ckpt)
    loaded = dist_checkpointing.load(model.sharded_state_dict(), src)
    res = model.load_state_dict(loaded, strict=False)
    _miss = [k for k in getattr(res, "missing_keys", []) if not k.endswith("_extra_state")]
    if _miss:
        raise SystemExit("[ov2-gen] {} model params UNLOADED (ckpt/arch mismatch): {}".format(len(_miss), _miss[:10]))
    model = model.cuda().eval()

    # 2) encode image+prompt with the SAME OV2 HF processor the task encoder uses.
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(p["hf_proc"], trust_remote_code=True)
    if args.image:
        from PIL import Image
        if args.image.startswith("http"):
            import requests
            img = Image.open(requests.get(args.image, stream=True).raw).convert("RGB")
        else:
            img = Image.open(args.image).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=[img], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], return_tensors="pt")

    input_ids = enc["input_ids"].cuda()
    pixel_values = enc.get("pixel_values")
    image_grid_thw = enc.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.cuda()
    eos = proc.tokenizer.eos_token_id

    # 3) greedy decode — full-recompute per step. forward(labels=None) -> logits [b, s, vocab].
    log("generate (max_new_tokens={}) ...".format(args.max_new_tokens))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(args.max_new_tokens):
            logits = model(
                images=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                position_ids=None,          # OV2/Qwen3 LLM derives positions (incl. M-RoPE) internally
                labels=None,                # None -> logits (not loss)
            )
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if eos is not None and int(nxt.item()) == int(eos):
                break

    out = proc.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    log("OUTPUT:\n" + out)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
