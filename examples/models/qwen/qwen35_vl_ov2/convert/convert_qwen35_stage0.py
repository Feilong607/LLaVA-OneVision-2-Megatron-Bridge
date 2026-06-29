#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Build the Qwen3.5-35B-A3B OV2 **stage_0** combined base ckpt (Bridge torch_dist, EP8) for stage-1.

qwen3.5-only / self-contained. Unlike the 30B `convert_ov2_checkpoint.py from_base` (which loads the
LLM from an AIAK mcore base), Qwen3.5 has NO AIAK base -> the LLM comes from HF:
  1) build_llava_ov2(load_llm_weights=True) streams the qwen3_5_moe_text weights from the -text HF dir
     (extract_qwen35_text.py output) into the freshly-built EP8 model.
  2) load_hf_encoder_into_tower() grafts the OneVision p16m33 vision tower weights.
  3) _init_ov2_adapter() initializes a fresh merge3 adapter (stage-1 trains it).
  4) save_megatron_model(torch_dist) writes a Bridge-loadable EP8 ckpt usable as pretrained_checkpoint.

Run (single 8-GPU node): torchrun --standalone --nproc_per_node=8 convert_qwen35_stage0.py
On A100-18 the docker `--gpus` hook is broken -> run the container with `--privileged -v /dev:/dev`.
"""
import argparse
import os

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3.5-35b-a3b")
    ap.add_argument("--vision_hf", default="/ov2/pretrain_models/lmms-lab/onevision_encoder_patch16_0507-tf57",
                    help="HF OneVisionEncoder (p16m33) dir to graft into the vision tower")
    ap.add_argument("--out", default="/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=8)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % args.ep != 0 or world < args.ep:
        raise SystemExit("[qwen35-stage0] world=%d must be a multiple of EP=%d and >= EP (use NPROC=8 for EP8)." % (world, args.ep))

    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=args.ep,
        )

    def log(m):
        if rank == 0:
            print("[qwen35-stage0] %s" % m, flush=True)

    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2, load_hf_encoder_into_tower
    from megatron.bridge.models.qwen_vl_ov2.ov2_provider import _init_ov2_adapter
    from megatron.bridge.training.model_load_save import save_megatron_model

    p = _ov2_backbone_paths(args.backbone)
    log("backbone=%s TP%d/EP%d llm_hf=%s img_tok=%s" % (args.backbone, args.tp, args.ep, p["llm_hf"], p.get("image_token_id")))
    log("vision_hf=%s  out=%s" % (args.vision_hf, args.out))
    assert os.path.isdir(args.vision_hf), "vision_hf dir not found: %s" % args.vision_hf
    assert os.path.isfile(os.path.join(p["llm_hf"], "config.json")), "llm_hf config missing: %s" % p["llm_hf"]

    # 1) build EP8 model + stream the qwen3.5 LLM weights from the -text HF dir
    model = build_llava_ov2(
        llm_hf_path=p["llm_hf"],
        load_llm_weights=True,
        perform_init=False,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        sequence_parallel=False,
        image_token_id=p.get("image_token_id", 151655),
        adapter_init_scale=p.get("adapter_init_scale", 1.0),
        mrope_section=p.get("mrope_section", None),
        patch_size=p["vision_patch_size"],
        spatial_merge_size=p["vision_spatial_merge_size"],
        vision_hidden_size=p["vision_hidden_size"],
        vision_num_layers=p["vision_num_layers"],
        vision_model_name=p["vision_model_name"],
    )
    log("build_llava_ov2 OK (qwen3.5 LLM weights streamed from HF); image_token_id=%s" % getattr(model, "image_token_id", "?"))

    # 2) graft the OneVision p16m33 vision tower
    r = load_hf_encoder_into_tower(model.vision_model, args.vision_hf)
    log("vision graft: loaded=%s missing=%d unexpected=%d" % (r["loaded"], len(r["missing"]), len(r["unexpected"])))
    if r["missing"]:
        log("  WARN vision MISSING(12): %s" % r["missing"][:12])
    if r["unexpected"]:
        log("  WARN vision UNEXPECTED(12): %s" % r["unexpected"][:12])

    # 3) fresh adapter (stage-1 trains it)
    _init_ov2_adapter(model.adapter)
    log("adapter freshly initialized (fc1 std, fc2 scaled)")

    # 4) save torch_dist EP8
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
    dist.barrier()
    save_megatron_model([model], args.out, ckpt_format="torch_dist", hf_tokenizer_path=p["llm_hf"])
    log("SAVED torch_dist EP%d stage_0 -> %s (use as pretrained_checkpoint)" % (args.ep, args.out))


if __name__ == "__main__":
    main()
