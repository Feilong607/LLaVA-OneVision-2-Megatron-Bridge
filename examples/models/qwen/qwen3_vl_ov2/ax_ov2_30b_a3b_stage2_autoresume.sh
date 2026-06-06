#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B Stage-2 - AUTO-RESTART WRAPPER (2 nodes: A100-22 + A100-26)
# -----------------------------------------------------------------------------
# Why: the run resumes fine from a checkpoint (model+optim+RNG+iteration all
#   restored via checkpoint.load=$SAVE) but WEDGES ~iter 700 on the EP8
#   _ALLGATHER_BASE of the 128-expert token-count vector (rank13 straggler) and
#   only checkpoints every 500 iters -> it crashes (~700) BEFORE the next save
#   (1000), so it never makes net progress. This wrapper:
#     (a) lowers save_interval to 100 so progress persists inside the crash window,
#     (b) loops: launch container -> wait -> on non-zero exit & iter<target, relaunch
#         (checkpoint.load=$SAVE auto re-reads latest_checkpointed_iteration.txt),
#     (c) is identical on BOTH nodes (node_rank auto-detected from LIST_IP),
#     (d) caps retries and logs every attempt.
#
# RUN THE SAME COMMAND ON BOTH NODES (start node22 and node26 within ~RDZV window):
#     on A100-22:  nohup bash ax_ov2_30b_a3b_stage2_autoresume.sh >> $SAVE/autoresume_node0.out 2>&1 &
#     on A100-26:  nohup bash ax_ov2_30b_a3b_stage2_autoresume.sh >> $SAVE/autoresume_node1.out 2>&1 &
#
# Resumes from checkpoint.load=$SAVE -> latest_checkpointed_iteration.txt (currently 1000).
# It only ever touches OUR container name (ov2_30b_s2); never other tenants.
# =============================================================================
set -uo pipefail   # NOT -e: we want to inspect non-zero exit codes, not abort on them

# ---- knobs (override via env) -----------------------------------------------
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge-refactor}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage1}"
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"          # final target iteration (1 epoch over 780k @ gbs128)
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"      # (a) was 500; 100 keeps a fresh ckpt inside the ~200-iter crash window
MAX_RETRIES="${MAX_RETRIES:-50}"     # (d) hard cap on relaunch attempts
RETRY_SLEEP="${RETRY_SLEEP:-30}"     # seconds between a crash and the next launch (let NCCL/IB settle)
# TIMEOUT_MIN tradeoff: the wedge is a TRUE no-show (flight recorder: one EP rank never
# posts the allgather), so a longer timeout just wastes more wall-clock before SIGABRT.
# For the auto-restart loop we want FAIL-FAST (15 min) so we relaunch sooner. Raise only
# if you suspect a slow-but-recoverable straggler (e.g. a Triton recompile finishing late).
TIMEOUT_MIN="${TIMEOUT_MIN:-15}"     # was 10 (default); 15 = small slack, still fail-fast for restart
LIST_IP="${LIST_IP:-172.16.5.22 172.16.5.26}"
CNAME="ov2_30b_s2"                   # OUR container only. NEVER touch lmms_eval_* / llava_megatron_*.

# ---- rendezvous (same logic as the existing ax_ov2_30b_a3b_stage2.sh) --------
read -ra list_ip <<< "$LIST_IP"
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"; NODE_RANK=0
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26042}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

# ---- env blocks (verbatim from the existing launcher) ------------------------
NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
FR_ENV="-e TORCH_NCCL_TRACE_BUFFER_SIZE=20000 -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$SAVE/nccl/trace -e TORCH_NCCL_DESYNC_DEBUG=1"

PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"
MUON_NOSPLIT="optimizer.muon_split_qkv=false"   # REQUIRED for OV2 stage-2 + Muon (vision QKV layout)
# RECOMPUTE: AIAK date0523 stage-2 uses --recompute-granularity full --recompute-method uniform
# --recompute-num-layers 1; the Bridge stage-2 recipe leaves it OFF (only midtrain sets it, ov2.py:345).
# model.recompute_activations=true is the provider knob (ov2_provider.py:85/169) that maps to AIAK's
# full/uniform/1 on the LLM (CLI-settable; does NOT hit the moe_permute_fusion propagation trap).
# Default ON here = aligns AIAK stage-2 + frees GPU headroom (NOT proven to fix the hang; see notes).
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"

mkdir -p "$SAVE" "$SAVE/nccl"
WRAPLOG="$SAVE/autoresume_node${NODE_RANK}.log"
echo "[autoresume] node_rank=$NODE_RANK nnodes=$NNODES save_every=$SAVE_EVERY target_iters=$ITERS timeout_min=$TIMEOUT_MIN" | tee -a "$WRAPLOG"

# ---- current iteration helper (re-read each loop; checkpoint.load uses this) --
cur_iter() { cat "$SAVE/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9' || echo 0; }

attempt=0
while :; do
  attempt=$((attempt+1))
  ITER_NOW="$(cur_iter)"; ITER_NOW="${ITER_NOW:-0}"
  if (( ITER_NOW >= ITERS )); then
    echo "[autoresume] reached target: latest=$ITER_NOW >= $ITERS. Done." | tee -a "$WRAPLOG"; exit 0
  fi
  if (( attempt > MAX_RETRIES )); then
    echo "[autoresume] hit MAX_RETRIES=$MAX_RETRIES at latest_iter=$ITER_NOW. Giving up." | tee -a "$WRAPLOG"; exit 1
  fi
  echo "[autoresume] === attempt $attempt/$MAX_RETRIES @ $(date -Is) | resume_from_iter=$ITER_NOW ===" | tee -a "$WRAPLOG"

  # Clean only OUR previous (Exited) container before each launch.
  docker rm -f "$CNAME" 2>/dev/null || true

  # Run the container in the FOREGROUND (no -d) so this loop blocks until the
  # training process exits; its exit code becomes the docker run exit code.
  docker run --rm --name "$CNAME" --network=host --privileged --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
    -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
    -e OV2_STAGE2_ADAMW="${OV2_STAGE2_ADAMW:-0}" $FR_ENV \
    -e TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_ov2s2}" \
    -e TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_ov2s2}" \
    -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
      pip install --no-input -q --timeout 10 --retries 1 py-spy 2>/dev/null || true   # best-effort; lets you 'py-spy dump' the NEXT wedge live (needs net or a pre-staged wheel)
      python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
        --recipe ov2_35b_a3b_stage2 --dataset vlm-energon --step_func ov2_step \
        dataset.path=$DATA_PATH $PRELOAD $RECOMPUTE_FLAG $MUON_NOSPLIT \
        checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
        checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
        dist.distributed_timeout_minutes=$TIMEOUT_MIN \
        validation.eval_iters=0 logger.log_interval=$LOG_EVERY" \
    >> "$SAVE/train_node${NODE_RANK}.log" 2>&1
  rc=$?

  ITER_AFTER="$(cur_iter)"; ITER_AFTER="${ITER_AFTER:-0}"
  echo "[autoresume] attempt $attempt exited rc=$rc | latest_iter now=$ITER_AFTER @ $(date -Is)" | tee -a "$WRAPLOG"

  if (( rc == 0 )); then
    echo "[autoresume] clean exit (rc=0). Training reported completion. Stopping loop." | tee -a "$WRAPLOG"; exit 0
  fi
  if (( ITER_AFTER >= ITERS )); then
    echo "[autoresume] target reached after crash (latest=$ITER_AFTER). Stopping loop." | tee -a "$WRAPLOG"; exit 0
  fi
  # Safety: if a launch made ZERO new checkpoints (e.g. crashed before iter+SAVE_EVERY),
  # we still retry, but warn so a wedged-on-launch loop is visible in the wrapper log.
  if (( ITER_AFTER <= ITER_NOW )); then
    echo "[autoresume] WARNING: no new checkpoint this attempt (still $ITER_AFTER). Retrying anyway." | tee -a "$WRAPLOG"
  fi
  echo "[autoresume] sleeping ${RETRY_SLEEP}s before relaunch..." | tee -a "$WRAPLOG"
  sleep "$RETRY_SLEEP"
done
