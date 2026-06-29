#!/usr/bin/env python3
"""Injection audit: is the vision->adapter prefix injected at the right COUNT and SCALE for Qwen3.5?
Builds OV2 like convert_qwen35_stage0 (LLM loaded + graft OneVision + fresh adapter, EP8), runs a
real-ish image through vision+adapter, and compares:
  (1) #visual tokens (adapter out) vs grid.prod//merge^2 (placeholder count)  [training asserts ==, so expect match]
  (2) adapter-output per-token L2 norm vs language_model.embedding() per-token L2 norm
A big norm ratio => visual prefix is an OUTLIER vs the text-embedding distribution the frozen LLM expects."""
import os, numpy as np, torch, torch.distributed as dist
from PIL import Image

def main():
    if not dist.is_initialized(): dist.init_process_group("nccl")
    lr = int(os.environ.get("LOCAL_RANK",0)); torch.cuda.set_device(lr)
    rank = dist.get_rank(); world = dist.get_world_size()
    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world)
    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2, load_hf_encoder_into_tower
    from megatron.bridge.models.qwen_vl_ov2.ov2_provider import _init_ov2_adapter
    from transformers import AutoProcessor, AutoTokenizer
    def log(m):
        if rank==0: print("[inject-audit]", m, flush=True)
    p = _ov2_backbone_paths("qwen3.5-35b-a3b")
    proc_path = os.environ.get("OV2_HF_PROC_QWEN35_P16M33", p.get("hf_proc"))
    vision_hf = os.environ.get("VISION_HF","/ov2/pretrain_models/lmms-lab/onevision_encoder_patch16_0507-tf57")
    merge = int(p["vision_spatial_merge_size"])
    log("build OV2 (LLM loaded EP%d) llm=%s merge=%d" % (world, p["llm_hf"], merge))
    model = build_llava_ov2(llm_hf_path=p["llm_hf"], load_llm_weights=True, perform_init=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world,
        sequence_parallel=False, image_token_id=p.get("image_token_id",151655),
        patch_size=p["vision_patch_size"], spatial_merge_size=merge,
        vision_hidden_size=p["vision_hidden_size"], vision_num_layers=p["vision_num_layers"],
        vision_model_name=p["vision_model_name"])
    r = load_hf_encoder_into_tower(model.vision_model, vision_hf)
    log("vision graft: loaded=%s missing=%d unexpected=%d" % (r["loaded"], len(r["missing"]), len(r["unexpected"])))
    _init_ov2_adapter(model.adapter)
    model = model.cuda().eval()
    dt = next(model.vision_model.parameters()).dtype
    # real-ish synthetic image (gradient+transpose) -> processor -> pixel_values+grid
    a = np.tile(np.linspace(0,255,448).astype(np.uint8),(448,1))
    img = Image.fromarray(np.stack([a, a.T, (a//2 + a.T//2).astype(np.uint8)], -1)).convert("RGB")
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=False, local_files_only=True)
    out = proc(images=[img], text=["<|vision_start|><|image_pad|><|vision_end|>a photo"], return_tensors="pt")
    pv = out["pixel_values"].to(dt).cuda(); grid = out["image_grid_thw"].cuda()
    n_expect = int(grid.prod().item()) // (merge*merge)
    n_pad = int((out["input_ids"]==p.get("image_token_id",151655)).sum().item())
    log("grid_thw=%s prod=%d -> expect visual=%d ; processor #<image_pad>=%d" % (grid.tolist(), int(grid.prod()), n_expect, n_pad))
    with torch.no_grad():
        ve = model.vision_model(pv, grid_thw=grid, patch_positions=None)
        ie = model.adapter(ve, patch_positions=None)
    log("vision out=%s adapter out=%s" % (tuple(ve.shape), tuple(ie.shape)))
    log("COUNT: expect=%d adapter_feats=%d processor_pads=%d  match=%s" % (n_expect, ie.shape[0], n_pad, (n_expect==ie.shape[0])))
    tok = AutoTokenizer.from_pretrained(p["llm_hf"], trust_remote_code=True)
    cap = tok("a photo of a dog playing in the park near a red car and two people on a bench", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        te = model.language_model.embedding(input_ids=cap, position_ids=None)
    tef = te.reshape(-1, te.shape[-1]).float(); ief = ie.reshape(-1, ie.shape[-1]).float()
    def st(x):
        nn = x.norm(dim=-1); return (nn.mean().item(), nn.std().item(), x.mean().item(), x.std().item(), x.abs().max().item())
    t = st(tef); v = st(ief)
    log("TEXT   emb/token: L2 mean=%.3f std=%.3f | elem mean=%.4f std=%.4f max|=%.3f" % t)
    log("VISUAL adp/token: L2 mean=%.3f std=%.3f | elem mean=%.4f std=%.4f max|=%.3f" % v)
    log("SCALE RATIO visualL2/textL2 = %.3f   (>>1 or <<1 => outlier prefix)" % (v[0]/t[0]))
    log("PROBE DONE")
    dist.barrier()

if __name__ == "__main__":
    main()
