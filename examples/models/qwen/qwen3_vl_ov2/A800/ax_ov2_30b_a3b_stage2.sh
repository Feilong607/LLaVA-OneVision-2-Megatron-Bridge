#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE + OV2 vision + m33 adapter) · Stage-2 SFT (p16m33)
# Trains vision tower + adapter (LLM FROZEN), distributed Muon, EP8. Chains from a TRAINED stage-1.
# Bridge-native: run_recipe.py + ov2_35b_a3b_stage2. Single node (--standalone) OR multi-node (LIST_IP).
#
# INIT_CKPT must point at a trained stage-1 output dir (run ax_ov2_30b_a3b_stage1.sh first); it loads
# MODEL-ONLY via checkpoint.pretrained_checkpoint (== AIAK --load <stage1> --no-load-optim --no-load-rng).
# Code = OV2 multi-backbone in /ov2/feilong/gb200/Megatron-Bridge (synced from -refactor, identical);
# outputs go under /ov2/feilong/gb200/ckpts_video_sft.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # fresh-clone safety: apply OV2 mcore submodule patch (apply_rotary_fn hook). FAIL LOUD (no 2>/dev/null||true): a silently-missing hook -> cryptic "unexpected keyword argument 'apply_rotary_fn'" at build. Script is idempotent (no-op if already applied).
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # stage-2 = distributed Muon (needs emerging_optimizers)
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"         # stage-2 SFT data (LLaVA-Next 780k)
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage1}"  # trained stage-1 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2}"
# hf_proc = OV2 image/video processor + tokenizer dir (was a recipe default; externalized
# so GB200 can override). A800 default = /ov2.
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128
LOG_EVERY="${LOG_EVERY:-10}"; SAVE_EVERY="${SAVE_EVERY:-200}"   # log every 10; save every 200 (lose <=200 iters on a wedge)

if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-30b-stage2] ERROR: INIT_CKPT=$INIT_CKPT not found. Run ax_ov2_30b_a3b_stage1.sh first, or set INIT_CKPT=null to start from the OV2-30B base." >&2
  exit 1
fi

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26042}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"

PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"
# RECOMPUTE=1 (default) enables LLM activation recompute. The CODE now uses SELECTIVE core_attn
# (frozen experts make full-layer recompute wasted compute); set OV2_RECOMPUTE_FULL=1 to force the old
# full/uniform/1 (e.g. for midtrain with the LLM unfrozen). Stage-2 also force-DISABLES
# moe_permute_fusion in code (fixes the TE Triton-JIT permute wedge); set OV2_MOE_PERMUTE_FUSION=1 to
# re-enable for A/B. These two toggles are forwarded into the container below.
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"
# Muon: the vision tower's attention QKV (TRAINED in stage-2; the LLM is FROZEN) has a different
# head/group layout than the LLM, but Muon's split_qkv reuses the LLM's qkv_split_shapes -> it crashes
# on the vision QKV ("RuntimeError: shape '[N, M, -1]' invalid for input of size ..."). The LLM QKV is
# not trained here, so just orthogonalize QKV as a single matrix. REQUIRED for OV2 stage-2 + Muon.
MUON_NOSPLIT="optimizer.muon_split_qkv=false"

mkdir -p "$SAVE" "$SAVE/nccl"; docker rm -f ov2_30b_p16m33_s2 2>/dev/null || true
# --- Muon resume-topology guard (Pass-6): distributed Muon (LayerWise) shards momentum across the DP
# axis with the ckpt replica_id DP-coord forced to 0 ("fixed DP usage only") and NO reshard support, so
# resuming a Muon ckpt at a DIFFERENT world size (e.g. A800 16-rank -> GB200 8-rank) SILENTLY loads
# mismatched momentum (weights fine; optimizer state corrupts) with no error. This script is always
# p16m33 stage-2 -> Muon unless OV2_STAGE2_ADAMW=1. WORLD==DP here (TP/PP/CP=1). Marker lives in $SAVE.
WORLD=$(( NPROC * (NNODES > 1 ? NNODES : 1) ))
_is_muon=0; [[ "${OV2_STAGE2_ADAMW:-0}" != "1" ]] && _is_muon=1
_wf="$SAVE/.ov2_train_world"
_has_ckpt=0; { [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]] || compgen -G "$SAVE/iter_*" >/dev/null 2>&1; } && _has_ckpt=1
if [[ "$_is_muon" == "1" && "$_has_ckpt" == "1" && -f "$_wf" ]]; then
  _saved_world="$(cat "$_wf" 2>/dev/null || echo "")"
  if [[ -n "$_saved_world" && "$_saved_world" != "$WORLD" ]]; then
    if [[ "${OV2_ALLOW_DP_RESHARD:-0}" == "1" ]]; then
      echo "[ov2-30b-stage2] WARN: Muon ckpt in $SAVE saved at WORLD=$_saved_world, resuming at WORLD=$WORLD -> DP-sharded momentum MISMATCH; OV2_ALLOW_DP_RESHARD=1 set, continuing (optimizer state WILL be wrong)." >&2
    else
      echo "[ov2-30b-stage2] FATAL: distributed-Muon ckpt in $SAVE was saved at WORLD=$_saved_world but you are resuming at WORLD=$WORLD. Muon momentum is DP-sharded with NO reshard support -> resuming silently loads MISMATCHED optimizer state. Resume at WORLD=$_saved_world, OR set OV2_ALLOW_DP_RESHARD=1 to override (momentum garbage; expect a loss/grad-norm bump)." >&2
      exit 1
    fi
  fi
fi
[[ "${NODE_RANK:-0}" -eq 0 ]] && { echo "$WORLD" > "$_wf" 2>/dev/null || true; }
# NCCL flight recorder + desync debug (default ON; set NCCL_TRACE=0 to disable). On a watchdog timeout
# it dumps per-rank collective traces to $SAVE/nccl, so an intermittent expert-parallel-group hang is
# precisely diagnosable (which rank diverged on which collective). Negligible overhead otherwise.
# TORCH_NCCL_DESYNC_DEBUG omitted on purpose: it adds a per-collective synchronized hash that slowed
# the run ~2-4x. The ring-buffer trace + dump-on-timeout below are cheap and still give the per-rank
# stuck-collective dump on a hang. Set NCCL_DESYNC=1 to re-enable the (expensive) desync report.
FR_ENV=""; [[ "${NCCL_TRACE:-1}" == "1" ]] && FR_ENV="-e TORCH_NCCL_TRACE_BUFFER_SIZE=20000 -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$SAVE/nccl/trace"
[[ "${NCCL_DESYNC:-0}" == "1" ]] && FR_ENV="$FR_ENV -e TORCH_NCCL_DESYNC_DEBUG=1"
echo "[ov2-30b-stage2] nnodes=${NNODES:-1} init=$INIT_CKPT save=$SAVE"
docker run -d --name ov2_30b_p16m33_s2 --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}" -e OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}" \
  -e OV2_STAGE2_ADAMW="${OV2_STAGE2_ADAMW:-0}" -e OV2_HF_PROC_30B_P16M33="$OV2_HF_PROC_30B_P16M33" $FR_ENV \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_30b_a3b_p16m33_stage2 --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD $RECOMPUTE_FLAG $MUON_NOSPLIT \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-stage2] launched -> tail -f $SAVE/train_node*.log  (loss prints on the LAST node)"
