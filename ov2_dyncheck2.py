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
print("import megatron.bridge: OK")
names=[str(s) for s in AutoBridge.list_supported_models()]
print("OV2 registered:", [s for s in names if "onevision2" in s.lower()] or "NONE")

from transformers import AutoConfig
import megatron.bridge.models.qwen_vl_ov2.ov2_bridge as ob
P4="/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"
P30="/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model"
for nm,p in [("4B dense",P4),("30B moe",P30)]:
    cfg=AutoConfig.from_pretrained(p, trust_remote_code=True)
    sup=AutoBridge.supports(cfg)
    # build the mapping for THIS config's branch (set hf_config like the dispatch does)
    b=ob.LlavaOnevision2MoEBridge(); b.hf_config=cfg
    reg=b.mapping_registry(); maps=getattr(reg,"mappings",None) or list(reg)
    hfs=set()
    for m in maps:
        for a in ("hf_param","q","k","v","gate","up"):
            x=getattr(m,a,None)
            if isinstance(x,str): hfs.add(x)
    has_expert=any("experts" in h for h in hfs)
    has_dense =any(h.endswith("mlp.gate_proj.weight") for h in hfs)
    has_router=any(h.endswith("mlp.gate.weight") for h in hfs)
    print(f"  {nm}: supports={sup} | mapping entries={len(maps)} | expert_maps={has_expert} dense_maps={has_dense} router={has_router}")
print("DONE")
