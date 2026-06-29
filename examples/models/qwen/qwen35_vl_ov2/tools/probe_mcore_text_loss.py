#!/usr/bin/env python3
"""Decisive probe: does the MCORE-converted Qwen3.5 LLM (the exact frozen LLM stage-1 uses)
compute the same pure-text loss as the healthy HF model (~1.12 on prose)?
Builds via the SAME path as convert_qwen35_stage0.py (build_llava_ov2 load_llm_weights=True, EP8),
then forwards prose through model.language_model and computes next-token CE on rank 0.
  mcore loss ~1.12  -> conversion+GDN/MoE compute faithful -> the +1.1 plateau is the vision/adapter path.
  mcore loss high   -> mcore conversion/compute broke the frozen LLM -> root cause.
Run: torchrun --standalone --nproc_per_node=8 probe_mcore_text_loss.py"""
import os, torch, torch.distributed as dist, torch.nn.functional as F

PROSE = [
  "The mitochondria is the powerhouse of the cell. It generates most of the cell supply of adenosine triphosphate, used as a source of chemical energy.",
  "In 1969, the Apollo 11 mission successfully landed the first humans on the Moon. Neil Armstrong became the first person to step onto the lunar surface.",
  "Machine learning is a subfield of artificial intelligence that focuses on building systems that learn from data. Deep neural networks have driven much of the recent progress.",
  "Water is composed of two hydrogen atoms and one oxygen atom. At standard temperature and pressure, it exists as a clear, colorless liquid.",
  "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect against nomadic invasions.",
]

def main():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr)
    rank = dist.get_rank(); world = dist.get_world_size()
    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world)
    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2
    from transformers import AutoTokenizer
    def log(m):
        if rank == 0: print("[mcore-probe] %s" % m, flush=True)
    p = _ov2_backbone_paths("qwen3.5-35b-a3b")
    log("building LLM (load_llm_weights=True, EP%d) from %s" % (world, p["llm_hf"]))
    model = build_llava_ov2(
        llm_hf_path=p["llm_hf"], load_llm_weights=True, perform_init=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        expert_model_parallel_size=world, sequence_parallel=False,
        image_token_id=p.get("image_token_id", 151655),
        patch_size=p["vision_patch_size"], spatial_merge_size=p["vision_spatial_merge_size"],
        vision_hidden_size=p["vision_hidden_size"], vision_num_layers=p["vision_num_layers"],
        vision_model_name=p["vision_model_name"],
    )
    lm = model.language_model.cuda().eval()
    log("build OK; running text forward")
    tok = AutoTokenizer.from_pretrained(p["llm_hf"], trust_remote_code=True)
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for t in PROSE:
            ids = tok(t, return_tensors="pt").input_ids.cuda()
            S = ids.shape[1]
            pos = torch.arange(S, device=ids.device, dtype=torch.long).unsqueeze(0)
            try:
                out = lm(input_ids=ids, position_ids=pos, attention_mask=None, labels=None)
            except Exception as e:
                log("forward(attn=None) failed: %r -> retry causal mask" % e)
                am = torch.tril(torch.ones(1,1,S,S, device=ids.device, dtype=torch.bool))
                out = lm(input_ids=ids, position_ids=pos, attention_mask=~am, labels=None)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            # logits [b,s,V]; next-token CE
            lg = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
            tg = ids[:, 1:].reshape(-1)
            ce = F.cross_entropy(lg, tg, reduction="mean").item()
            n = S - 1; tot += ce*n; ntok += n
            log("  ce=%.3f n=%d :: %s" % (ce, n, t[:48]))
    if rank == 0:
        print("[mcore-probe] MEAN per-token CE = %.4f over %d tokens  (HF ref on same prose = 1.12)" % (tot/ntok, ntok), flush=True)
        print("[mcore-probe] PROBE DONE", flush=True)
    dist.barrier()

if __name__ == "__main__":
    main()
