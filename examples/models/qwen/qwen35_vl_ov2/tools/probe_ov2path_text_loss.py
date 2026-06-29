#!/usr/bin/env python3
"""Does feeding decoder_input (the OV2 fuse path) change Qwen3.5 text loss vs the input_ids path?
Both pass the 1D-arange position_ids the OV2 forward uses for mtp. 2 variants on identical text:
  A) input_ids + pos1d, NO decoder_input        [= the 1.12 mcore reference]
  B) input_ids + pos1d + decoder_input=emb       [EXACT OV2 forward main path]
A==B==~1.12 -> decoder_input/mrope path fine -> position is NOT the gap. B high -> the OV2 path mishandles it."""
import os, torch, torch.distributed as dist, torch.nn.functional as F
PROSE = [
  "The mitochondria is the powerhouse of the cell. It generates most of the cell supply of adenosine triphosphate.",
  "In 1969, the Apollo 11 mission successfully landed the first humans on the Moon.",
  "Machine learning is a subfield of artificial intelligence that focuses on building systems that learn from data.",
]
def main():
    if not dist.is_initialized(): dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",0)))
    rank=dist.get_rank(); world=dist.get_world_size()
    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(1,1,expert_model_parallel_size=world)
    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2
    from transformers import AutoTokenizer
    def log(m):
        if rank==0: print("[ov2path]",m,flush=True)
    p=_ov2_backbone_paths("qwen3.5-35b-a3b")
    model=build_llava_ov2(llm_hf_path=p["llm_hf"], load_llm_weights=True, perform_init=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world,
        sequence_parallel=False, image_token_id=p.get("image_token_id",151655),
        patch_size=p["vision_patch_size"], spatial_merge_size=p["vision_spatial_merge_size"],
        vision_hidden_size=p["vision_hidden_size"], vision_num_layers=p["vision_num_layers"], vision_model_name=p["vision_model_name"])
    lm=model.language_model.cuda().eval()
    log("mtp_process=%s" % getattr(lm,"mtp_process",None))
    tok=AutoTokenizer.from_pretrained(p["llm_hf"], trust_remote_code=True)
    def ce_of(out, ids):
        lg = out if isinstance(out,torch.Tensor) else out[0]
        if lg.dim()==3 and lg.shape[0]==ids.shape[1] and lg.shape[1]==ids.shape[0]:
            lg=lg.transpose(0,1)
        lg=lg[:,:-1,:].float().reshape(-1,lg.shape[-1]); tg=ids[:,1:].reshape(-1)
        return F.cross_entropy(lg,tg).item()
    accA=accB=0.0; n=0
    with torch.no_grad():
        for t in PROSE:
            ids=tok(t,return_tensors="pt").input_ids.cuda(); S=ids.shape[1]
            pos1d=torch.arange(S,device=ids.device,dtype=torch.long).unsqueeze(0)
            emb=lm.embedding(input_ids=ids, position_ids=None)
            oA=lm(input_ids=ids, position_ids=pos1d, attention_mask=None, labels=None)
            oB=lm(input_ids=ids, position_ids=pos1d, attention_mask=None, decoder_input=emb, labels=None)
            a,b=ce_of(oA,ids),ce_of(oB,ids); accA+=a;accB+=b;n+=1
            log("  A(input_ids)=%.3f  B(decoder_input=OV2path)=%.3f :: %s" % (a,b,t[:40]))
    log("MEAN  A=%.4f  B=%.4f   (ref 1.12; B is the exact OV2 main-path call)" % (accA/n,accB/n))
    log("PROBE DONE")
    dist.barrier()
if __name__=="__main__": main()
