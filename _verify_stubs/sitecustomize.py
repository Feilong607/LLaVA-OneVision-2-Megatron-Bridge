import sys, types, importlib.abc, importlib.machinery
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # generic stub for optional deps missing in this image: modelopt.* + a few new transformers.models.*
    PFX = ("modelopt", "diffusers",
           "transformers.models.ernie4_5_vl_moe","transformers.models.glm4v",
           "transformers.models.qwen3_omni_moe","transformers.models.qwen3_next")
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if any(name==p or name.startswith(p+".") for p in self.PFX) or (top=="modelopt"):
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__path__=[]
        m.__dict__["__getattr__"] = lambda n: (type(n,(object,),{}) if n[:1].isupper() else (lambda *a, **k: False))
        return m
    def exec_module(self, m): pass
# append (LAST) so any REAL installed module is found first; we only catch genuine ImportErrors
sys.meta_path.append(_Stub())
