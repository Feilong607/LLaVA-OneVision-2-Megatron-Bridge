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

# --- legacy bespoke modes: delegate to the preserved per-dir bespoke launcher (platform-correct paths) ---
case "$MODE" in
  from_base|reshard|export_hf)
    [[ -f "$HERE/convert_bespoke.sh" ]] || { echo "FATAL: $HERE/convert_bespoke.sh (legacy launcher) not found" >&2; exit 1; }
    exec bash "$HERE/convert_bespoke.sh" "$MODE" "$@" ;;
  verify)
    [[ $# -ge 2 ]] || { echo "usage: convert.sh verify <A_ckpt> <B_ckpt>" >&2; exit 1; }
    exec env A="$1" B="$2" bash "$HERE/verify.sh" ;;
esac

# --- platform auto-detect (mirrors the training launcher HW gate) ---
_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo 0)"
if [[ "${_cc:-0}" -ge 100 ]]; then
  PLAT=gb200; DEF_NPROC=4
  CFG_DEF="${CFG:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
  CKPTA_DEF="${CKPTA:-/datasets/llava-ov2-30b-a3b-m9lvdn}"
  FOURB_DEF="${FOURB:-/datasets/llava/11May/lmms-lab/LLaVA-OneVision-2-4B-p16m33}"
  # /datasets is the READ-ONLY dataset mount on GB200 -> scratch (cfg_dispatch + HF export) MUST be writable.
  # The gb200 training launcher writes to /home/ftan0055/...; mirror that. Override WORK= for a different user/path.
  WORK="${WORK:-/home/ftan0055/_ov2_convert}"
else
  PLAT=a800; DEF_NPROC=8
  CFG_DEF="${CFG:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
  CKPTA_DEF="${CKPTA:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon}"
  FOURB_DEF="${FOURB:-/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33}"
  WORK="${WORK:-/ov2/feilong/gb200/_rt30b}"
fi
NPROC="${NPROC:-$DEF_NPROC}"
HF_OUT="${HF_OUT:-$WORK/hf_export}"

# env contract: offline HF; PYTHONPATH incl aiak_shim + _verify_stubs (shims optional modelopt/diffusers);
# MoE permute fusion OFF (OV2 wedge gotcha).
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TE_EXTRA_STATE_MISSING_CHECK="${TE_EXTRA_STATE_MISSING_CHECK:-1}" OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"

# torchrun rendezvous: standalone (1 node) or multi-node EP8 (GB200 2x4). node_rank auto from LIST_IP.
if [[ -n "${LIST_IP:-}" ]]; then
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

do_export(){
  (( WORLD == 8 )) || { echo "ERROR: 30B is EP8 -> world must be 8 (got $WORLD: NPROC=$NPROC nodes=${NN:-1}); on GB200 use NPROC=4 LIST_IP=<2 nodes>" >&2; exit 1; }
  local CFG_RDY; CFG_RDY="$(ensure_dispatch_cfg "$CFG_DEF" "$WORK/cfg_dispatch")"
  echo "==> [$PLAT] export: mcore $CKPTA_DEF -> HF $HF_OUT   (cfg=$CFG_RDY)"
  CFG="$CFG_RDY" CKPTA="$CKPTA_DEF" HF="$HF_OUT" dist "$HERE/ov2_30b_export_ep8.py"
  echo "==> [$PLAT] copy custom .py + tokenizer/processor aux into HF (save_hf_pretrained can't auto-copy from a local source)"
  for f in "$CFG_RDY"/*.py "$CFG_RDY"/tokenizer* "$CFG_RDY"/*token* "$CFG_RDY"/*preprocessor* "$CFG_RDY"/generation_config.json "$CFG_RDY"/vocab.json "$CFG_RDY"/merges.txt "$CFG_RDY"/chat_template.jinja "$CFG_RDY"/added_tokens.json; do
    [ -f "$f" ] && cp -n "$f" "$HF_OUT/" 2>/dev/null || true
  done
}
do_roundtrip(){
  (( WORLD == 8 )) || { echo "ERROR: 30B roundtrip is EP8 -> world must be 8 (got $WORLD)" >&2; exit 1; }
  echo "==> [$PLAT] roundtrip: $HF_OUT -> Megatron(EP$WORLD) -> HF + allclose"
  dist examples/conversion/hf_megatron_roundtrip_multi_gpu.py --hf-model-id "$HF_OUT" --tp 1 --pp 1 --ep "$WORLD" --trust-remote-code --not-strict
}

case "$MODE" in
  export)    do_export ;;
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
