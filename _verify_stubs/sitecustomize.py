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
        def _ga(n):
            # introspection-safe: inspect.getmodule()/getsourcefile() walk sys.modules and call
            # __file__.endswith(...) on EVERY module during any torch custom-op registration. A stub
            # whose __file__ is a function crashes there ('function' has no attribute 'endswith').
            # Raise AttributeError for dunders -> hasattr(stub,'__file__')==False -> inspect skips the stub.
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)
            return type(n,(object,),{}) if n[:1].isupper() else (lambda *a, **k: None)
        m.__dict__["__getattr__"] = _ga
        return m
    def exec_module(self, m): pass
# append (LAST) so any REAL installed module is found first; we only catch genuine ImportErrors
sys.meta_path.append(_Stub())

# --- botocore<->boto3 rename compat (root-fix the version-skew ImportError in this image) ---
# Old boto3 does `from botocore.docs.utils import DocumentModifiedShape`; newer botocore renamed
# it to DocumentedShape (and vice-versa). This image mixes boto3(dist-packages)+botocore(venv),
# so the names disagree. Runs at interpreter startup (sitecustomize) BEFORE any `import boto3`,
# so the name exists by the time boto3 imports it. No-op if botocore absent or already consistent.
try:
    import botocore.docs.utils as _bdu
    for _missing, _src in (("DocumentModifiedShape", "DocumentedShape"),
                           ("DocumentedShape", "DocumentModifiedShape")):
        if not hasattr(_bdu, _missing) and hasattr(_bdu, _src):
            setattr(_bdu, _missing, getattr(_bdu, _src))
        if not hasattr(_bdu, _missing):
            setattr(_bdu, _missing, type(_missing, (object,), {}))
except Exception:
    pass

# --- FORCE-stub diffusers even when INSTALLED (some images e.g. nemo-test ship a diffusers that CRASHES on
# import: torch custom-op double-registration in attention_dispatch/ace_step_transformer). OV2 does NOT use
# diffusers at train/convert time -> stub wins over the broken real package (front of meta_path). ---
_FORCE = ("diffusers", "modelopt")   # modelopt: real one in nemo image introspects the diffusers stub via inspect -> crash; it is a hard top-level import in checkpointing.py/eval.py so stubbing (as in the validated modelopt-absent image) is required, not optional
class _ForceStub(_Stub):
    PFX = _FORCE
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in self.PFX or any(name == p or name.startswith(p + ".") for p in self.PFX):
            return importlib.machinery.ModuleSpec(name, self)
        return None
sys.meta_path.insert(0, _ForceStub())
for _pfx in _FORCE:
    for _m in [n for n in list(sys.modules) if n == _pfx or n.startswith(_pfx + ".")]:
        del sys.modules[_m]
