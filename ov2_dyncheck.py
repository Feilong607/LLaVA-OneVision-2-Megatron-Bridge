import sys, types, importlib.abc, importlib.machinery
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PFX = ("transformers.models.ernie4_5_vl_moe","transformers.models.glm4v",
           "transformers.models.qwen3_omni_moe","transformers.models.qwen3_next")
    def find_spec(self, name, path=None, target=None):
        if any(name==p or name.startswith(p+".") for p in self.PFX):
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__path__=[]
        m.__dict__["__getattr__"] = lambda n: type(n,(object,),{})
        return m
    def exec_module(self, m): pass
sys.meta_path.append(_Stub())

print("== import megatron.bridge (does the wiring break it?) ==")
from megatron.bridge import AutoBridge
print("  import OK")

print("== is the OV2 bridge REGISTERED now? ==")
names=[]
try: names=[str(s) for s in AutoBridge.list_supported_models()]
except Exception as e: print("  list_supported_models err:", e)
print("  OV2 entries in registry:", [s for s in names if "nevision" in s.lower() or "ov2" in s.lower()] or "NONE")

print("== dispatch: AutoBridge.supports() on real configs ==")
from transformers import AutoConfig
for nm,p in [("4B dense","/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"),
             ("30B moe","/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model")]:
    try:
        cfg=AutoConfig.from_pretrained(p, trust_remote_code=True)
        print(f"  {nm}: architectures={getattr(cfg,'architectures',None)} model_type={getattr(cfg,'model_type',None)!r} -> supports={AutoBridge.supports(cfg)}")
    except Exception as e:
        print(f"  {nm}: ERR {type(e).__name__}: {e}")

print("== mapping_registry builds? ==")
import megatron.bridge.models.qwen_vl_ov2.ov2_bridge as ob
reg=ob.LlavaOnevision2MoEBridge().mapping_registry()
maps=getattr(reg,"mappings",None) or list(reg)
print("  mapping_registry OK:", len(maps), "entries")
print("DONE")
