#!/usr/bin/env bash
# =============================================================================
# OV2 checkpoint conversion launcher (in-container, torchrun).
# World size (= nnodes * NPROC) MUST be a whole multiple of EP, and >= EP.
# OV2 is verified at TP1/PP1/EP8. On GB200 each node has 4 GPUs, so EP8 needs >= 2 nodes.
#
#   from_base (assembled AIAK base -> Bridge torch_dist), EP8 across 2 GB200 nodes (4+4=8 GPU):
#     NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh from_base \
#         --src <base_ckpt> --out <out> --ep 8
#     # (run the SAME command on both nodes; node_rank is auto-detected from LIST_IP)
#
#   reshard TP1 -> TP2 (keep EP8), 2 GB200 nodes (8 GPU) -- run SAME cmd on both nodes:
#     NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh reshard --src <ckpt> --out <out> --tp 2 --ep 8 --etp 1
#     (--etp 1 is REQUIRED with --tp>1: else expert-TP defaults to TP and world must cover TP*EP.)
#
#   reshard EP8 -> EP4 for a single 4-GPU GB200 node (UNVALIDATED; verify afterwards):
#     NPROC=4 bash convert/convert.sh reshard --src <ep8_ckpt> --out <ep4_out> --ep 4
#
#   (PP>1 and CP>1 are rejected: OV2 is a monolithic VLM with a PP1/CP1-pinned vision tower.)
#
#   export_hf (EP8, 2 GB200 nodes):
#     NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh export_hf --src <ckpt> --out <hf_dir>
#
# On an 8-GPU node (e.g. A100), a single node suffices: NPROC=8 bash convert/convert.sh from_base ...
# Always verify after:  A=<src> B=<out> bash convert/verify.sh
# =============================================================================
set -euo pipefail
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
NPROC="${NPROC:-4}"                       # GB200 = 4 GPU/node (use 8 on an 8-GPU node)
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# --- GB200 30B-A3B reshard defaults (convert.sh was not GB200-path-aware; the launcher is). All env-overridable. ---
export OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"  # GB200 LLM HF (A100 /ov2 default does not exist here)
export OV2_CONVERT_TOKENIZER="${OV2_CONVERT_TOKENIZER:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"  # complete tokenizer (the GB200 LLM-HF copy tokenizer is incomplete)
# default the 30B backbone to p16m33 (patch16/merge3 vision tower) when the caller did not pass --backbone
case " $* " in *" --backbone "*) ;; *) set -- "$@" --backbone qwen3-30b-a3b-p16m33 ;; esac
SCRIPT="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/convert/convert_ov2_checkpoint.py"

if [[ -n "${LIST_IP:-}" ]]; then read -ra ip <<< "$LIST_IP"; else ip=(); fi
NN=${#ip[@]}; [[ "$NN" -le 1 ]] && NN=1

# --- guard: world (= NN*NPROC) must cover EP and TP (parse from args; defaults EP=8, TP=1) ---
EP=8; TP=1; ETP=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  # bounds-check i+1 so a trailing "--ep"/"--tp"/"--etp" with no value doesn't trip `set -u`
  if [[ $((i+1)) -lt ${#args[@]} ]]; then
    [[ "${args[$i]}" == "--ep" ]]  && EP="${args[$((i+1))]}"
    [[ "${args[$i]}" == "--tp" ]]  && TP="${args[$((i+1))]}"
    [[ "${args[$i]}" == "--etp" ]] && ETP="${args[$((i+1))]}"
  fi
done
WORLD=$(( NN * NPROC ))
if (( WORLD < EP || WORLD % EP != 0 )); then
  echo "ERROR: world=NN*NPROC=$NN*$NPROC=$WORLD must be >= EP=$EP and a whole multiple of it." >&2
  echo "       GB200 has 4 GPU/node -> EP8 needs >=2 nodes: NPROC=4 LIST_IP=\"<ip0> <ip1>\" ..." >&2
  exit 1
fi
if (( WORLD % TP != 0 )); then
  echo "ERROR: world=$WORLD must be divisible by TP=$TP (attention needs TP x DP == world)." >&2
  exit 1
fi
# Expert grid: mcore defaults expert-TP to TP when --etp is 0/unset, so the expert layout needs
# world % (ETP_EFF * EP) == 0. This is the #1 reshard foot-gun (--tp 2 --ep 8 with no --etp -> ETP=2 ->
# needs world%16). Catch it here with a friendly message instead of a raw mcore assert deep in _init.
ETP_EFF=$(( ETP > 0 ? ETP : TP ))
if (( WORLD % (ETP_EFF * EP) != 0 )); then
  echo "ERROR: world=$WORLD not divisible by expert grid ETP_EFF*EP=$ETP_EFF*$EP=$((ETP_EFF*EP))." >&2
  echo "       With --tp $TP and no --etp, expert-TP defaults to TP=$TP. Pass --etp 1 to keep experts" >&2
  echo "       un-TP-sharded (grid = 1*EP=$EP), or raise the node count so world covers $((ETP_EFF*EP))." >&2
  exit 1
fi

if [[ "$NN" -le 1 ]]; then
  RDZV="--standalone --nnodes=1"
else
  MA="${ip[0]}"; MP="${MASTER_PORT:-26049}"; CUR="$(hostname -I | awk '{print $1}')"; NR=-1
  for i in "${!ip[@]}"; do [[ "${ip[$i]}" == "$CUR" ]] && NR=$i && break; done
  [[ "$NR" -eq -1 ]] && { echo "ERROR: $CUR not in LIST_IP (${ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NN --node_rank=$NR --master_addr=$MA --master_port=$MP"
fi
cd "$REPO"
echo "[ov2-convert] nnodes=$NN nproc_per_node=$NPROC world=$WORLD TP=$TP EP=$EP | args: $*"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$SCRIPT" "$@"
