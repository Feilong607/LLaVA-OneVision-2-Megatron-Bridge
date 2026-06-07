#!/usr/bin/env bash
# =============================================================================
# gb200_preflight.sh — RUN-READINESS gate for the OV2-30B-A3B GB200 launch.
# Complements gb200_check.sh (which does GPUs/NVLink/NCCL/perf-peaks). This one validates the things
# that actually CRASH the training launch on a no-internet box: checkpoints, data, HF dirs, disk, the
# container stack imports, and — critically — that the EXACT config you're about to launch RESOLVES
# (the `mixed_precision` recipe + all CLI overrides) BEFORE you burn a multi-node allocation.
#
# Run INSIDE the training container. It mirrors the launcher's env knobs, so set the SAME ones:
#   bash examples/models/qwen/qwen3_vl_ov2/gb200/gb200_preflight.sh                    # defaults, ACCEL=0
#   ACCEL=1 INIT_CKPT=/path SAVE=/path DATA_PATH=/path bash .../gb200_preflight.sh     # what you'll launch
# Exit 0 = GO (no blocking FAIL); exit 1 = NO-GO. WARNs are advisory.
# =============================================================================
set -uo pipefail   # NOT -e: run ALL checks, then tally.

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
RECIPE="${RECIPE:-ov2_35b_a3b_midtrain}"
ACCEL="${ACCEL:-0}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200}"
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"
export INIT_CKPT DATA_PATH SAVE ACCEL RECIPE   # the section-5 python subprocess reads these via os.environ

PASS=0; WARN=0; FAIL=0
ok(){   echo "  [PASS] $*"; PASS=$((PASS+1)); }
warn(){ echo "  [WARN] $*"; WARN=$((WARN+1)); }
bad(){  echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
sec(){  echo; echo "========== $* =========="; }
exdir(){  [[ -d "$1" ]] && ok "$2"  || bad "$2 -- MISSING dir: $1"; }
exfile(){ [[ -f "$1" ]] && ok "$2"  || bad "$2 -- MISSING file: $1"; }

sec "0. config under test (mirror your launch env)"
echo "  REPO=$REPO  RECIPE=$RECIPE  ACCEL=$ACCEL"
echo "  INIT_CKPT=$INIT_CKPT"
echo "  DATA_PATH=$DATA_PATH"
echo "  SAVE=$SAVE"

sec "1. checkpoint to load (INIT_CKPT / pretrained_checkpoint)"
if [[ -f "$INIT_CKPT/latest_checkpointed_iteration.txt" ]]; then
  it="$(cat "$INIT_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null)"
  itd="$INIT_CKPT/iter_$(printf '%07d' "$it" 2>/dev/null)"
  exfile "$itd/.metadata"        "INIT_CKPT torch_dist .metadata (iter $it)"
  exfile "$itd/run_config.yaml"  "INIT_CKPT run_config.yaml (iter $it)"
  n="$(ls "$itd"/*.distcp 2>/dev/null | wc -l)"; [[ "${n:-0}" -gt 0 ]] && ok "INIT_CKPT has $n .distcp shards" || bad "INIT_CKPT has NO .distcp shards in $itd"
elif [[ -n "$(ls "$INIT_CKPT"/release/mp_rank_00_*/model_optim_rng.pt 2>/dev/null)" ]]; then
  warn "INIT_CKPT looks like an AIAK stitch-base (release/mp_rank_*/model_optim_rng.pt) -> use convert from_base first, or point at a Bridge torch_dist ckpt"
elif [[ -d "$INIT_CKPT/release" ]]; then
  bad "INIT_CKPT has release/ but NO mp_rank_*/model_optim_rng.pt shards (corrupt/partial stitch base): $INIT_CKPT"
elif [[ "$INIT_CKPT" == "null" || -z "$INIT_CKPT" ]]; then
  warn "INIT_CKPT unset/null -> training starts from the recipe stitch base (intended only for first bootstrap)"
else
  bad "INIT_CKPT not a loadable ckpt (no latest_checkpointed_iteration.txt / release/): $INIT_CKPT"
fi

sec "2. training data (energon Metadataset + shards)"
exfile "$DATA_PATH" "DATA_PATH yaml present"
if [[ -f "$DATA_PATH" ]]; then
  dpaths=(); while IFS= read -r _d; do dpaths+=("$_d"); done < <(grep -oE 'path:[[:space:]]*\S+' "$DATA_PATH" | awk '{print $2}')   # portable (no bash-4 mapfile)
  nd=${#dpaths[@]}
  if [[ "$nd" -eq 0 ]]; then bad "no 'path:' entries parsed from $DATA_PATH"; else
    ok "$nd dataset path(s) listed in yaml"
    miss=0
    for d in "${dpaths[@]}"; do
      [[ -f "$d/.nv-meta/dataset.yaml" ]] || { miss=$((miss+1)); [[ "$miss" -le 3 ]] && bad ".nv-meta missing in $d"; }
    done
    [[ "$miss" -eq 0 ]] && ok "all $nd data dirs have .nv-meta/dataset.yaml"
    tarmiss=0; first_s=0
    for _i in "${!dpaths[@]}"; do
      _s="$(ls "${dpaths[$_i]}"/*.tar 2>/dev/null | wc -l)"; [[ "$_i" -eq 0 ]] && first_s="$_s"
      [[ "${_s:-0}" -gt 0 ]] || { tarmiss=$((tarmiss+1)); [[ "$tarmiss" -le 3 ]] && bad "no .tar shards in ${dpaths[$_i]}"; }
    done
    [[ "$tarmiss" -eq 0 ]] && ok "all $nd data dirs have .tar shards (node_0: $first_s)" || bad "$tarmiss data dir(s) have NO .tar shards"
    [[ "$miss" -gt 3 ]] && bad "...and $((miss-3)) more data dirs missing .nv-meta"
  fi
fi

sec "3. HF dirs (LLM arch + processor) — MUST be local (no internet at runtime)"
exdir "/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507"                              "LLM HF dir (AutoBridge arch/weights)"
exfile "/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507/config.json"                 "LLM HF config.json"
exdir "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model"     "HF processor dir (energon task encoder)"

sec "4. disk space for SAVE (ckpts ~60GB each, model-only; more with optimizer)"
sdir="$SAVE"; while [[ ! -d "$sdir" && "$sdir" != "/" ]]; do sdir="$(dirname "$sdir")"; done
avail_g="$(df -BG --output=avail "$sdir" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "${avail_g:-}" ]]; then
  [[ "$avail_g" -ge 200 ]] && ok "free space on $sdir: ${avail_g}G (filesystem-wide on a network mount; check your dir/tenant QUOTA separately)" || warn "only ${avail_g}G free on $sdir (a full 30B ckpt is large; ensure room for save_interval keeps)"
else warn "could not stat free space on $sdir"; fi
# real write-probe: catches read-only / quota-full that the FS-wide 'avail' figure hides
if [[ -w "$sdir" ]]; then
  _probe="$sdir/.preflight_write_test.$$"
  if (echo ok >"$_probe") 2>/dev/null; then rm -f "$_probe"; ok "SAVE parent write-probe OK: $sdir"; else bad "SAVE parent WRITE FAILED (read-only / quota-full?): $sdir"; fi
else bad "SAVE parent NOT writable: $sdir"; fi

sec "5. container stack imports + CONFIG RESOLVES (catches the mixed_precision-recipe class of crash)"
python3 - <<'PY'
import os, sys, traceback
ACCEL = os.environ.get("ACCEL", "0")
MP = "bf16_with_mxfp8_mixed" if ACCEL == "1" else "bf16_mixed"
fails = []
def ok(m): print("  [PASS] " + m)
def bad(m): print("  [FAIL] " + m); fails.append(m)
def warn(m): print("  [WARN] " + m)
try:
    import torch
    ok("torch %s  cuda=%s  n_gpu=%d" % (torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))
except Exception as e:
    bad("import torch: %s" % e); print("RESULT_FAILS=%d" % len(fails)); sys.exit(1)
for mod, label in [("transformer_engine", "TransformerEngine"), ("megatron.core", "megatron-core"),
                   ("megatron.energon", "megatron-energon")]:
    try: __import__(mod); ok("import %s" % label)
    except Exception as e: bad("import %s: %s" % (label, e))
for imp, label in [
    ("from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2, load_ov2_mcore_checkpoint", "llava_ov2 (model build + stitch)"),
    ("from megatron.bridge.models.qwen_vl_ov2.ov2_step import forward_step, get_batch", "ov2_step (packed forward)"),
    ("from megatron.bridge.recipes.ov2.ov2 import ov2_35b_a3b_midtrain", "recipe ov2_35b_a3b_midtrain"),
    ("from aiak_training_llm.data.multimodal import PackedCaptioningSample", "aiak_shim PackedCaptioningSample"),
    ("from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend", "flex dispatcher helper"),
]:
    try: exec(imp); ok("import " + label)
    except Exception as e: bad("import %s: %s" % (label, e))
# the bf16-class blocker: the mixed_precision recipe for THIS ACCEL must resolve
try:
    from megatron.bridge.training.mixed_precision import get_mixed_precision_config
    c = get_mixed_precision_config(MP)
    ok("mixed_precision %r resolves (bf16=%s fp8=%s)" % (MP, getattr(c, "bf16", None), getattr(c, "fp8", None)))
except Exception as e:
    bad("mixed_precision %r does NOT resolve: %s  <-- launch would crash at config finalize" % (MP, e))
# the recipe builds + the ACCEL override set applies (catches a bad/rejected override key)
try:
    from megatron.bridge.recipes.ov2.ov2 import ov2_35b_a3b_midtrain
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
    cfg = ov2_35b_a3b_midtrain()
    ovr = ["mixed_precision=%s" % MP, "train.micro_batch_size=1", "scheduler.lr_warmup_iters=0",
           "checkpoint.pretrained_checkpoint=%s" % os.environ["INIT_CKPT"],
           "checkpoint.save=%s" % os.environ["SAVE"], "checkpoint.load=%s" % os.environ["SAVE"],
           "dataset.path=%s" % os.environ["DATA_PATH"], "validation.eval_iters=0"]
    if ACCEL == "1":
        ovr += ["model.recompute_activations=false", "model.recompute_granularity=null"]
    cfg = process_config_with_overrides(cfg, cli_overrides=ovr)
    ok("recipe builds + ACCEL=%s overrides apply (config resolves -> launch won't crash at config)" % ACCEL)
except Exception as e:
    traceback.print_exc(); bad("recipe/override resolution: %s" % e)
# precision/dispatcher availability for the chosen ACCEL
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
if ACCEL == "1":
    if cc[0] >= 9: ok("fp8 HW present (cc=%s) -> MXFP8 usable" % (cc,))
    else: warn("ACCEL=1 (MXFP8) but cc=%s has NO fp8 (Ampere) -> launcher auto-falls back to bf16 baseline" % (cc,))
if ACCEL == "2":
    try:
        import deep_ep
        from deep_ep import HybridEPBuffer  # noqa
        from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
        assert HAVE_HYBRIDEP, "HAVE_HYBRIDEP is False (HybridEPBuffer/hybrid_ep_cpp missing)"
        ok("HybridEP fully importable (deep_ep + HybridEPBuffer + HAVE_HYBRIDEP) -> ACCEL=2 usable offline")
    except Exception as e:
        bad("ACCEL=2 (HybridEP) NOT importable offline: %s -> use ACCEL=0/1 (alltoall) on this container" % e)
# Triton x Blackwell: Triton<3.3 has no mature sm_100 codegen. The fla '>=3.3' warning is benign for
# qwen3_moe (no GDN at runtime), but TE/grouped-GEMM Triton kernels still JIT on Blackwell. The fix is
# the right CONTAINER (cu13/Blackwell ships Triton>=3.3) -- do NOT pip-upgrade Triton (ABI-pinned to torch/TE).
try:
    import triton
    _tv = tuple(int(x) for x in triton.__version__.split(".")[:2])
    if cc[0] >= 10 and _tv < (3, 3):
        bad("Triton %s on Blackwell (cc=%s): no mature sm_100 codegen -> you are likely on a cu12 container; "
            "use the cu13/Blackwell image (Triton>=3.3). Do NOT pip-upgrade Triton (ABI-pinned to torch/TE)."
            % (triton.__version__, cc))
    elif cc[0] >= 10:
        ok("Triton %s OK for Blackwell (torch cuda %s)" % (triton.__version__, torch.version.cuda))
    else:
        ok("Triton %s (cc=%s, torch cuda %s) -- fla>=3.3 warning is benign here (qwen3_moe has no GDN at runtime)"
           % (triton.__version__, cc, torch.version.cuda))
except Exception as e:
    warn("triton/Blackwell check: %s" % e)
print("RESULT_FAILS=%d" % len(fails))
sys.exit(1 if fails else 0)
PY
pyrc=$?
[[ "$pyrc" -eq 0 ]] && ok "stack + config-resolution python checks passed" || bad "stack/config-resolution checks FAILED (see [FAIL] lines above)"

sec "6. hardware profile + ACCEL availability (informational)"
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
[[ "$cc" =~ ^[0-9]+$ ]] || cc=""
if   [[ "${cc:-0}" -ge 100 ]]; then echo "  HW=gb200 (cc=$cc): NPROC=4/node, bf16 peak~2250 / fp8~4500, NVLS on, fp8 OK"
elif [[ "${cc:-0}" -ge 90  ]]; then echo "  HW=hopper (cc=$cc): NPROC=8/node, bf16~989 / fp8~1979, fp8 OK"
elif [[ -n "${cc:-}" ]];       then echo "  HW=ampere (cc=$cc): NPROC=8/node, bf16~312, NO fp8 -> ACCEL=1 auto-falls back to bf16"
else warn "could not read compute_cap (nvidia-smi); HW auto-detect will default to ampere"; fi

sec "7. multi-node / NCCL / offline hygiene"
[[ "${HF_HUB_OFFLINE:-}" == "1" && "${TRANSFORMERS_OFFLINE:-}" == "1" ]] && ok "HF offline flags set (no hub access at runtime)" || warn "HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE not both 1 -> risk of a hub fetch on a no-internet box"
echo "  reminder: EP8 needs >=2 GB200 nodes (LIST_IP=\"<ip0> <ip1>\"); GBS=128 -> use 2 or 4 nodes (world 8/16). odd node counts assert."
[[ "${OV2_MOE_PERMUTE_FUSION:-0}" == "0" ]] && ok "OV2_MOE_PERMUTE_FUSION=0 (avoids the TE Triton MoE-permute wedge)" || warn "OV2_MOE_PERMUTE_FUSION!=0 -> re-enables the fused-permute kernel that wedged 30B-A3B"

echo
echo "========== PREFLIGHT SUMMARY: $PASS pass / $WARN warn / $FAIL fail =========="
if [[ "$FAIL" -eq 0 ]]; then
  echo "GO  -- environment is ready for the ACCEL=$ACCEL launch (review any [WARN] above)."
  exit 0
else
  echo "NO-GO -- $FAIL blocking issue(s) above MUST be fixed before launching on the no-internet box."
  exit 1
fi
