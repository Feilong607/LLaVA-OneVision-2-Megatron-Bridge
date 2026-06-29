#!/usr/bin/env python3
"""Round-1 (CPU) verification of ov2_bridge.py: registration wiring, dispatch, static mapping analysis.
Run inside llava_megatron:26.05 with PYTHONPATH=<repo>/src:<repo>/3rdparty/Megatron-LM:<repo>/aiak_shim."""
import json, os, re, struct, traceback

FOURB = "/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"            # dense, has safetensors
THIRTYB = "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model"  # moe, config only


def sec(t): print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def has_ov2(names):
    return names is not None and any(("nevision" in str(s).lower()) or ("ov2" in str(s).lower()) for s in names)


def supported_names(AutoBridge):
    for meth in ("list_supported_models",):
        if hasattr(AutoBridge, meth):
            try:
                return list(getattr(AutoBridge, meth)())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------- 1. wiring
sec("1. REGISTRATION WIRING  (does the decorator run without an explicit ov2_bridge import?)")
from megatron.bridge import AutoBridge
import megatron.bridge.models.qwen_vl_ov2 as fam
print("  family __init__ exposes LlavaOnevision2MoEBridge:", hasattr(fam, "LlavaOnevision2MoEBridge"))
after_family = supported_names(AutoBridge)
print("  registered after FAMILY import only:", has_ov2(after_family))
try:
    import megatron.bridge.models.qwen_vl_ov2.ov2_bridge as ob
    print("  explicit `import ov2_bridge` OK; class:", ob.LlavaOnevision2MoEBridge.__name__)
except Exception:
    print("  explicit import FAILED:"); traceback.print_exc(); raise SystemExit(2)
after_explicit = supported_names(AutoBridge)
print("  registered after EXPLICIT import:", has_ov2(after_explicit))
if after_explicit is not None:
    print("  OV2-ish entries:", [s for s in after_explicit if has_ov2([s])])

# ---------------------------------------------------------------- 2. dispatch
sec("2. DISPATCH  (AutoBridge.supports / can_handle on the REAL configs)")
from transformers import AutoConfig
for name, path in [("4B  dense  (llava_onevision2)", FOURB), ("30B moe    (llava_onevision2_moe)", THIRTYB)]:
    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None)
        mt = getattr(cfg, "model_type", None)
        sup = AutoBridge.supports(cfg) if hasattr(AutoBridge, "supports") else "n/a"
        print(f"  {name}: architectures={archs} model_type={mt!r} -> supports={sup}")
    except Exception as e:
        print(f"  {name}: ERR {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 3. mapping static analysis
sec("3. MAPPING REGISTRY  (static structure + coverage vs real 4B HF keys)")
b = ob.LlavaOnevision2MoEBridge()
reg = b.mapping_registry()
maps = getattr(reg, "mappings", None) or getattr(reg, "_mappings", None) or list(reg)
print("  total mapping entries:", len(maps))

hf_pats, meg_pats, dup = set(), [], {}
for m in maps:
    mp = getattr(m, "megatron_param", None)
    if mp:
        meg_pats.append(mp)
    for a in ("hf_param", "q", "k", "v", "gate", "up"):
        val = getattr(m, a, None)
        if isinstance(val, str):
            hf_pats.add(val)
# duplicate megatron targets (a param mapped twice)
from collections import Counter
mc = Counter(meg_pats)
dups = {k: v for k, v in mc.items() if v > 1}
print("  duplicate megatron_param targets:", dups or "none")
# the suspicious double input_layernorm (both fused qkv LN and a standalone input_layernorm)
sus = [p for p in meg_pats if p.endswith("self_attention.linear_qkv.layer_norm_weight") or p.endswith("input_layernorm.weight")]
print("  input-layernorm-ish megatron targets:", sorted(set(sus)))


def st_keys(path):
    f = os.path.join(path, "model.safetensors")
    if os.path.isfile(f):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            hdr = json.loads(h.read(n))
        return set(k for k in hdr if k != "__metadata__")
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        return set(json.load(open(idx))["weight_map"].keys())
    return None


def rx(p):
    return re.compile("^" + re.escape(p).replace(r"\*\*", ".+").replace(r"\*", "[^.]+") + "$")


pats = [(p, rx(p)) for p in hf_pats]
keys4 = st_keys(FOURB)
print(f"\n  -- coverage vs 4B (DENSE) HF keys: note this bridge is MoE, so dense MLP keys are EXPECTED uncovered --")
if keys4:
    uncovered = sorted(k for k in keys4 if not any(r.match(k) for _, r in pats))
    print(f"  4B keys: {len(keys4)} total | {len(uncovered)} uncovered")
    # bucket the uncovered by prefix
    from collections import Counter as C2
    bc = C2(".".join(k.split(".")[:4]) for k in uncovered)
    for pref, c in bc.most_common(20):
        print(f"     uncovered[{c:4d}]  {pref}")
    dead = sorted(p for p, r in pats if not any(r.match(k) for k in keys4))
    print(f"  mapping HF-patterns matching ZERO 4B keys ({len(dead)}) — MoE/expert ones are expected dead on dense 4B:")
    for d in dead:
        print("     dead-pat:", d)
else:
    print("  (no 4B safetensors found)")
print("\nROUND-1 DONE.")
