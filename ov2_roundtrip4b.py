import os, sys, types, importlib.abc, importlib.machinery, traceback
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PFX=("transformers.models.ernie4_5_vl_moe","transformers.models.glm4v","transformers.models.qwen3_omni_moe","transformers.models.qwen3_next")
    def find_spec(self,n,path=None,t=None):
        return importlib.machinery.ModuleSpec(n,self) if any(n==p or n.startswith(p+".") for p in self.PFX) else None
    def create_module(self,spec):
        m=types.ModuleType(spec.name); m.__path__=[]; m.__dict__["__getattr__"]=lambda x:type(x,(object,),{}); return m
    def exec_module(self,m): pass
sys.meta_path.append(_Stub())

import torch, torch.distributed as dist
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29561", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
dist.init_process_group("gloo")
torch.cuda.set_device(0)
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(1,1)

from megatron.bridge import AutoBridge
P4="/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"
print("== from_hf_pretrained(4B) ==")
bridge = AutoBridge.from_hf_pretrained(P4, trust_remote_code=True)
print("  bridge:", type(bridge).__name__)
print("== build mcore provider + model (structure) ==")
prov = bridge.to_megatron_provider(load_weights=False)
prov.tensor_model_parallel_size=1; prov.pipeline_model_parallel_size=1
prov.finalize()
model = prov.provide_distributed_model(wrap_with_ddp=False)
print("  model built:", type(model[0] if isinstance(model,list) else model).__name__)
print("== load_hf_weights (THE import-mapping test) ==")
bridge.load_hf_weights(model)
print("  load_hf_weights OK  <-- every mapped weight loaded")
print("== export_hf_weights (export-mapping test) ==")
mdl = model[0] if isinstance(model,list) else model
n=0; sample=[]
for name,t in bridge.export_hf_weights(mdl, cpu=True):
    n+=1
    if n<=4: sample.append((name, tuple(t.shape)))
print("  exported tensors:", n)
for s in sample: print("    ", s)
print("ROUNDTRIP-4B DONE OK")
