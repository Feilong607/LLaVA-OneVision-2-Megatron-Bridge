#!/usr/bin/env bash
# =============================================================================
# OV2 (LLaVA-OneVision-2) checkpoint conversion — UNIFIED entry (bespoke + bridge).
# AUTO-DETECTS platform from GPU compute capability:
#   GB200 (cc >= 10.0) -> 4 GPU/node, /datasets/... paths, EP8 needs 2 nodes (LIST_IP)
#   A100/A800 (8 GPU/node) -> /ov2/... paths, EP8 on a single node
# All paths are env-overridable (CFG / CKPTA / HF_OUT / FOURB / WORK / NPROC / LIST_IP).
#
# Bridge-native path drives the REGISTERED LlavaOnevision2MoEBridge (AutoBridge dispatch).
# VERIFIED 2026-06-29 (A100 8xGPU, llava_megatron:26.05):
#   4b        dense  HF->Megatron->HF + allclose ........ 696/696,   0 mismatch
#   export    30B EP8 mcore->HF (full single HF VLM) ..... 2172 tensors -> 58G
#   roundtrip 30B EP8 HF->Megatron->HF + allclose ........ 19164/19164, 0 mismatch
#
# Modes:
#   convert.sh 30b                 export + roundtrip (the full 30B validation)        [bridge]
#   convert.sh export              30B mcore ckpt -> HF (EP8, full single HF VLM)      [bridge]
#   convert.sh roundtrip           30B exported-HF -> Megatron(EP8) -> HF + allclose   [bridge]
#   convert.sh 4b                  4B dense HF->Megatron->HF + allclose                [bridge]
#   convert.sh from_base  <args>   assembled AIAK base -> mcore torch_dist  (bootstrap)[bespoke]
#   convert.sh reshard    <args>   mcore -> mcore at a different TP/EP/ETP             [bespoke]
#   convert.sh export_hf  <args>   mcore -> partial HF (LLM-HF + vision/adapter .pt)   [bespoke]
#   convert.sh verify  A B         torch_dist GLOBAL consistency (CPU, cross-reshard)  [bespoke]
#
# Launch (run the SAME cmd on every node; node_rank auto-detected from LIST_IP):
#   A800/A100 single 8-GPU node:      bash convert.sh 30b
#   GB200 EP8 across 2x4-GPU nodes:   NPROC=4 LIST_IP="<ip0> <ip1>" bash convert.sh 30b
# =============================================================================
set -euo pipefail
MODE="${1:?usage: convert.sh 30b|4b|export|roundtrip|from_base|reshard|export_hf|verify ...}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ d="$HERE"; while [[ "$d" != "/" && ! -d "$d/src/megatron/bridge" ]]; do d="$(dirname "$d")"; done; echo "$d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found above $HERE (set REPO=)" >&2; exit 1; }

# Resolve home robustly for the gb200 scratch default (some launch contexts clear $HOME, which would collapse
# "$HOME/..." to "/..."). Prefer $HOME, else the passwd-db home, else /home/<user>. Resolves to the real home
# dir at runtime; no username literal is committed (id -un supplies it).
_HOME="${HOME:-}"
[[ -n "$_HOME" ]] || _HOME="$(getent passwd "$(id -un 2>/dev/null)" 2>/dev/null | cut -d: -f6)"
[[ -n "$_HOME" ]] || _HOME="/home/$(id -un 2>/dev/null)"

# --- legacy bespoke modes: delegate to the preserved per-dir bespoke launcher (platform-correct paths) ---
case "$MODE" in
  from_base|reshard|export_hf)
    [[ -f "$HERE/convert_bespoke.sh" ]] || { echo "FATAL: $HERE/convert_bespoke.sh (legacy launcher) not found" >&2; exit 1; }
    exec bash "$HERE/convert_bespoke.sh" "$MODE" "$@" ;;
  verify)
    [[ $# -ge 2 ]] || { echo "usage: convert.sh verify <A_ckpt> <B_ckpt>" >&2; exit 1; }
    exec env A="$1" B="$2" bash "$HERE/verify.sh" ;;
esac

# --- platform: PLAT=gb200|a800 (or HW=) override WINS; else auto-detect from compute_cap. Parse with awk
#     (first row, DIGITS ONLY: "10.0"->"100"). The old `... | head -1 | tr -d . || echo 0` broke under
#     `set -o pipefail`: head's SIGPIPE made the pipe non-zero -> `|| echo 0` appended -> _cc="100 0" ->
#     `[[ "100 0" -ge 100 ]]` syntax error -> silently fell to the a800 (A100) paths on a GB200 box. ---
PLAT="${PLAT:-${HW:-}}"
_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NR==1{gsub(/[^0-9]/,"");print}')"
[[ "$_cc" =~ ^[0-9]+$ ]] || _cc=0
[[ -n "$PLAT" ]] || { [[ "$_cc" -ge 100 ]] && PLAT=gb200 || PLAT=a800; }
if [[ "$PLAT" == "gb200" ]]; then
  PLAT=gb200; DEF_NPROC=4
  CFG_DEF="${CFG:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
  CKPTA_DEF="${CKPTA:-/datasets/llava-ov2-30b-a3b-m9lvdn}"
  FOURB_DEF="${FOURB:-/datasets/llava/11May/lmms-lab/LLaVA-OneVision-2-4B-p16m33}"
  # /datasets is the READ-ONLY dataset mount on GB200 -> scratch (cfg_dispatch + HF export) MUST be writable.
  # The gb200 training launcher writes under $HOME; mirror that for scratch. Override WORK= for a different path.
  WORK="${WORK:-$_HOME/_ov2_convert}"
else
  PLAT=a800; DEF_NPROC=8
  CFG_DEF="${CFG:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
  CKPTA_DEF="${CKPTA:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon}"
  FOURB_DEF="${FOURB:-/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33}"
  WORK="${WORK:-/ov2/feilong/gb200/_rt30b}"
fi
NPROC="${NPROC:-$DEF_NPROC}"
HF_OUT="${HF_OUT:-$WORK/hf_export}"
# Guard: keep WORK/HF_OUT OFF the repo. convert.sh does `cd "$REPO"`, so a RELATIVE WORK/HF_OUT (or one
# pointed under $REPO) would write the 30-58G HF export INTO the Bridge repo (pollutes git-sync, fills the
# repo FS -- this happened: stray hf_export*/4B dirs). Absolutize relative paths against $PWD (pre-cd),
# then reject anything under $REPO.
for _v in WORK HF_OUT; do
  _p="${!_v}"; case "$_p" in /*) ;; *) _p="$PWD/$_p";; esac; printf -v "$_v" '%s' "$_p"
done
case "$HF_OUT/" in "$REPO/"*) echo "FATAL: HF_OUT=$HF_OUT is UNDER the repo ($REPO). Weights must NOT live in the Bridge repo -- set HF_OUT to an off-repo path (e.g. \$WORK/hf_export)." >&2; exit 1;; esac
case "$WORK/"   in "$REPO/"*) echo "FATAL: WORK=$WORK is UNDER the repo ($REPO). Set WORK to an off-repo scratch dir." >&2; exit 1;; esac

# Create scratch + export dirs up-front. ensure_dispatch_cfg does `cp -r "$CFG" "$WORK/cfg_dispatch"`, which
# fails ("cannot create directory ... No such file or directory") when $WORK's parent doesn't exist yet;
# save_hf_pretrained/do_fixup likewise assume $HF_OUT exists. mkdir -p is idempotent, so this is safe to always run.
mkdir -p "$WORK" "$HF_OUT"

# env contract: offline HF; PYTHONPATH incl aiak_shim + _verify_stubs (shims optional modelopt/diffusers);
# MoE permute fusion OFF (OV2 wedge gotcha).
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TE_EXTRA_STATE_MISSING_CHECK="${TE_EXTRA_STATE_MISSING_CHECK:-1}" OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"

# torchrun rendezvous, 3 priorities (mirrors the training launcher -> LIST_IP is now OPTIONAL):
#   (1) operator-injected env (PyTorchJob/Run:AI auto-inject PET_* + MASTER_ADDR/WORLD_SIZE) -> TRUE auto, NO LIST_IP
#   (2) manual LIST_IP="<ip0> <ip1> ..." (run SAME cmd on each node)
#   (3) single-node standalone
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NN="${PET_NNODES:-$(( WORLD_SIZE / NPROC ))}"
  NR="${PET_NODE_RANK:-$(( ${RANK:-0} / NPROC ))}"
  MA="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"; MP="${MASTER_PORT:-${PET_MASTER_PORT:-26060}}"
  [[ "$NN" -gt 1 && -z "${MA:-}" ]] && { echo "ERROR: multi-node but neither MASTER_ADDR nor PET_MASTER_ADDR injected by the operator" >&2; exit 1; }
  RDZV="--nnodes=$NN --node_rank=$NR --master_addr=$MA --master_port=$MP"; WORLD=$((NN*NPROC))
  echo "==> rdzv: operator-auto (nnodes=$NN node_rank=$NR master=$MA:$MP world=$WORLD)"
elif [[ -n "${LIST_IP:-}" ]]; then
  read -ra ip <<< "$LIST_IP"; NN=${#ip[@]}
  MA="${ip[0]}"; MP="${MASTER_PORT:-26060}"; CUR="$(hostname -I | awk '{print $1}')"; NR=-1
  for i in "${!ip[@]}"; do [[ "${ip[$i]}" == "$CUR" ]] && NR=$i && break; done
  [[ "$NR" -eq -1 ]] && { echo "ERROR: $CUR not in LIST_IP (${ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NN --node_rank=$NR --master_addr=$MA --master_port=$MP"; WORLD=$((NN*NPROC))
else
  RDZV="--standalone --nnodes=1"; WORLD="$NPROC"
fi
cd "$REPO"
dist(){ python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$@"; }

# AutoBridge dispatches on config.architectures; p16m33 skeleton ships architectures:null -> WORK copy w/ it set.
ensure_dispatch_cfg(){
  local src="$1" dst="$2"
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$src/config.json')).get('architectures') else 1)"; then echo "$src"
  else rm -rf "$dst"; cp -r "$src" "$dst"
    python3 -c "import json;p='$dst/config.json';c=json.load(open(p));c['architectures']=['LlavaOnevision2ForConditionalGeneration'];json.dump(c,open(p,'w'),indent=2)"
    echo "$dst"; fi
}

do_fixup(){  # apply HF-skeleton fixes to an existing export dir (idempotent; standalone: convert.sh fixup <dir>)
  local D="${1:?usage: convert.sh fixup <hf_export_dir>}"
  [[ -f "$D/config.json" ]] || { echo "FATAL: $D has no config.json" >&2; exit 1; }
  echo "==> [$PLAT] fixup: $D"
  # (a) golden modeling: _init_weights re-inits VisionRotaryEmbedding inv_freq buffers (persistent=False ->
  #     garbage after meta-tensor from_pretrained), is_first_iteration pixel-drop (transformers 5.x removed
  #     cache_position for remote code), spatial_merge_size from config (skeleton hardcoded 2; p16m33 = 3).
  cp -f "$HERE/hf_skeleton_fixes/modeling_llava_onevision2_moe.py" "$D/modeling_llava_onevision2_moe.py"
  # (b) chat template: lmms-eval chat impl needs apply_chat_template; skeleton auto_model may lack the file
  [[ -f "$D/chat_template.jinja" ]] || cp "$HERE/hf_skeleton_fixes/chat_template.jinja" "$D/"
  # (c) config: pos_enc false (ckpt has no pos_emb weights); preprocessor jsons: sync patch/merge/temporal
  #     from vision_config (skeleton ships generic p14/m2 values -> reshape crash / silent wrong layout)
  python3 - "$D" <<'FIXJSON'
import json, os, sys
d = sys.argv[1]
cp_ = os.path.join(d, "config.json")
c = json.load(open(cp_)); hit = False
for key in ("vision_config", "visual"):
    v = c.get(key)
    if isinstance(v, dict) and v.get("use_patch_position_encoding"):
        v["use_patch_position_encoding"] = False; hit = True
if hit:
    json.dump(c, open(cp_, "w"), indent=2)
    print("==> fixup: use_patch_position_encoding true->false (ckpt has no pos_emb weights)")
vc = c.get("vision_config") or {}
patch, merge = vc.get("patch_size"), vc.get("spatial_merge_size")
tps = vc.get("temporal_patch_size") or 1
if patch and merge:
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        p2 = os.path.join(d, name)
        if not os.path.exists(p2):
            continue
        j = json.load(open(p2))
        before = (j.get("patch_size"), j.get("merge_size"), j.get("temporal_patch_size"))
        if before != (patch, merge, tps):
            j["patch_size"], j["merge_size"], j["temporal_patch_size"] = patch, merge, tps
            json.dump(j, open(p2, "w"), indent=2)
            print("==> fixup: %s (patch,merge,temporal) %s -> %s" % (name, before, (patch, merge, tps)))
FIXJSON
}

do_export(){
  (( WORLD == ${OV2_EP:-8} )) || { echo "ERROR: 30B export needs world == EP (OV2_EP=${OV2_EP:-8}) but world=$WORLD (NPROC=$NPROC nodes=${NN:-1}). Verified path = EP8 on 8 GPUs (2 nodes). Single 4-GPU node: OV2_EP=4 (EP4 load-reshards the EP8 ckpt -> UNVALIDATED, check roundtrip allclose)." >&2; exit 1; }
  local CFG_RDY; CFG_RDY="$(ensure_dispatch_cfg "$CFG_DEF" "$WORK/cfg_dispatch")"
  echo "==> [$PLAT] export: mcore $CKPTA_DEF -> HF $HF_OUT   (cfg=$CFG_RDY)"
  CFG="$CFG_RDY" CKPTA="$CKPTA_DEF" HF="$HF_OUT" dist "$HERE/ov2_30b_export_ep8.py"
  echo "==> [$PLAT] copy custom .py + tokenizer/processor aux into HF (save_hf_pretrained can't auto-copy from a local source)"
  for f in "$CFG_RDY"/*.py "$CFG_RDY"/tokenizer* "$CFG_RDY"/*token* "$CFG_RDY"/*preprocessor* "$CFG_RDY"/generation_config.json "$CFG_RDY"/vocab.json "$CFG_RDY"/merges.txt "$CFG_RDY"/chat_template.jinja "$CFG_RDY"/added_tokens.json; do
    [ -f "$f" ] && cp -n "$f" "$HF_OUT/" 2>/dev/null || true
  done
  # The p16m33 auto_model skeleton config WRONGLY ships use_patch_position_encoding:true, but the trained
  # mcore ckpt has ZERO adapter.pos_emb_* keys (never trained with -pos; no AIAK script ever passed it;
  # config-class default is False). true makes HF create + APPLY randomly-initialized pos_emb_h/w at
  # inference (the vision tower passes patch_positions into the merger unconditionally) -> silent image-
  # feature corruption + MISSING-keys load report. Force false so the exported config matches the weights.
  if [[ "${NR:-0}" == "0" ]]; then
    do_fixup "$HF_OUT"
  fi
}
do_roundtrip(){
  (( WORLD == ${OV2_EP:-8} )) || { echo "ERROR: 30B roundtrip needs world == EP (OV2_EP=${OV2_EP:-8}) but world=$WORLD." >&2; exit 1; }
  echo "==> [$PLAT] roundtrip: $HF_OUT -> Megatron(EP$WORLD) -> HF + allclose"
  dist examples/conversion/hf_megatron_roundtrip_multi_gpu.py --hf-model-id "$HF_OUT" --tp 1 --pp 1 --ep "$WORLD" --trust-remote-code --not-strict
}

case "$MODE" in
  export)    do_export ;;
  fixup)     do_fixup "${1:?usage: convert.sh fixup <hf_export_dir>}" ;;
  roundtrip) do_roundtrip ;;
  30b)       do_export; do_roundtrip ;;
  4b)
    echo "==> [$PLAT] 4B round-trip (HF->Megatron->HF + allclose), dense, 1 process"
    python -m torch.distributed.run --standalone --nproc_per_node=1 \
      examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id "$FOURB_DEF" --tp 1 --pp 1 --ep 1 --trust-remote-code --not-strict ;;
  *) echo "usage: convert.sh 30b|4b|export|roundtrip|from_base|reshard|export_hf|verify ..." >&2; exit 1 ;;
esac
echo "==> [$PLAT] convert.sh $MODE done."
