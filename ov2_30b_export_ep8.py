"""EP8 multi-GPU export: OV2 30B mcore ckpt -> HF, via load_megatron_model (collective) + save_hf_pretrained.
Run: torchrun --nproc_per_node=8 ov2_30b_export_ep8.py"""
import os, torch, torch.distributed as dist
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))   # THE FIX: per-rank device
if not dist.is_initialized():
    dist.init_process_group("nccl")
RANK = dist.get_rank()
def log(m):
    if RANK == 0: print("[ep8-export] " + m, flush=True)

from megatron.bridge import AutoBridge
CFG  = os.environ.get("CFG",  "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model")
CKPT = os.environ.get("CKPTA","/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon")
HF   = os.environ.get("HF",   "/ov2/feilong/gb200/_rt30b/hf_export")

log(f"AutoBridge.from_auto_config(CKPT={CKPT}, CFG={CFG})  # config-only bridge: no source-HF-weights lookup")
bridge = AutoBridge.from_auto_config(CKPT, CFG, trust_remote_code=True)
log(f"load_megatron_model({CKPT}) @ EP8/TP1/PP1 (collective)")
model = bridge.load_megatron_model(
    CKPT,
    mp_overrides=dict(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                      expert_model_parallel_size=8, expert_tensor_parallel_size=1),
    wrap_with_ddp=False,
)
# load_megatron_model returns a list (PP/vp stages); save_hf_pretrained expects the list
log("save_hf_pretrained -> " + HF)
bridge.save_hf_pretrained(model, HF)
dist.barrier()
log("EP8 EXPORT DONE")
