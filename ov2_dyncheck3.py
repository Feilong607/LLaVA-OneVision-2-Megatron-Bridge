import sys, types, importlib.abc, importlib.machinery
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PFX=("transformers.models.ernie4_5_vl_moe","transformers.models.glm4v","transformers.models.qwen3_omni_moe","transformers.models.qwen3_next")
    def find_spec(self,n,path=None,t=None):
        return importlib.machinery.ModuleSpec(n,self) if any(n==p or n.startswith(p+".") for p in self.PFX) else None
    def create_module(self,spec):
        m=types.ModuleType(spec.name); m.__path__=[]; m.__dict__["__getattr__"]=lambda x:type(x,(object,),{}); return m
    def exec_module(self,m): pass
sys.meta_path.append(_Stub())
from megatron.bridge import AutoBridge
from transformers import AutoConfig
import megatron.bridge.models.qwen_vl_ov2.ov2_bridge as ob

def allstr(m):
    out=set()
    for v in vars(m).values():
        if isinstance(v,str): out.add(v)
        elif isinstance(v,(list,tuple)):
            out|={x for x in v if isinstance(x,str)}
        elif isinstance(v,dict):
            out|={x for x in v.values() if isinstance(x,str)}
    return out

for nm,p in [("4B dense","/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"),
             ("30B moe","/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model")]:
    cfg=AutoConfig.from_pretrained(p, trust_remote_code=True)
    b=ob.LlavaOnevision2MoEBridge(); b.hf_config=cfg
    maps=getattr(b.mapping_registry(),"mappings",None) or list(b.mapping_registry())
    megs=set(); hfs=set()
    for m in maps:
        mp=getattr(m,"megatron_param",None)
        if isinstance(mp,str): megs.add(mp)
        hfs|=allstr(m)
    # LLM MLP megatron targets present?
    dense_fc1 = any(x=="language_model.decoder.layers.*.mlp.linear_fc1.weight" for x in megs)
    expert_fc1= any("mlp.experts" in x and "linear_fc1" in x for x in megs)
    gate_hf   = any(x.endswith("mlp.gate_proj.weight") for x in hfs)         # dense gate
    egate_hf  = any(x.endswith("mlp.experts.*.gate_proj.weight") for x in hfs)  # moe expert gate
    print(f"{nm}: dense_linear_fc1={dense_fc1} expert_linear_fc1={expert_fc1} | dense_gate_proj_hf={gate_hf} expert_gate_proj_hf={egate_hf}")
print("DONE")
