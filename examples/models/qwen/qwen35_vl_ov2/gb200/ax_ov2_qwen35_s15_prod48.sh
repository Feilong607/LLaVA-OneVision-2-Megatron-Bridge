#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Qwen3.5-35B-A3B stage-1.5: fresh 48-GPU midtrain, then resume its own SAVE.
#
# Workers=11: 12 pods x 4 GPUs, TP=1 / DP=48 / EP=8, GBS=240, MBS=1, seq=10192.
# GBS=256 cannot divide DP=48; 240 gives five microbatches/rank (288 gives six).
# The former 30B TP4 failure required WORLD % (ETP*EP)=WORLD%32==0; TP1 needs WORLD%8==0.
#
# First launch: initialize from staged stage-2 iter_0006000 weights, train the FULL
# 8,000,000-sample budget (ceil(8000000/240)=33334 iterations). No previous midtrain source.
# Restart: same Args + same SAVE + Workers=11. Bridge restores model, Muon, scheduler,
# iteration, RNG and Energon state from this SAVE. Never subtract consumed samples:
# train_iters remains the original endpoint. Keep GBS, budget and schedule unchanged.
# With no completed checkpoint yet, an interruption restarts from the stage-2 weights.
#
# Form: Distributed/PyTorch, qwen35-fla image, Command=bash on both sides; Args=this
# file's absolute path on both sides; OV2_K8S_NAMESPACE on both sides. Pin all 12 pods
# to one NVL72 rack. The base launcher does not enforce rack placement.
# Optional KEY=VALUE args: SAVE, INIT_CKPT (initial stage-2 weights), OV2_MIDTRAIN_GBS,
# OV2_MIDTRAIN_N_SAMPLES, SAVE_EVERY, OV2_KEEP_CKPTS. Defaults retain Muon + bf16 HybridEP,
# selective LLM recompute and vision recompute from the measured TP1 production lane.
# OV2_PREFLIGHT_ONLY=1 checks topology/checkpoint compatibility without launching GPUs;
# prod32 and the base launcher additionally check model/processor/data assets at launch.
# Iteration logs: worker-10. 48-GPU throughput still needs a GPU measurement.
# =============================================================================
set -euo pipefail

for _kv in "$@"; do
  if [[ "$_kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
    export "$_kv"
  else
    echo "FATAL: positional arg '$_kv' is not KEY=VALUE" >&2; exit 1
  fi
done

_P48_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_P48_BASE="$_P48_DIR/ax_ov2_qwen35_s15_prod32.sh"
_P48_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
mkdir -p "$HOME/train_logs"
LOG="$HOME/train_logs/prod48_qwen35_s15_${_P48_TAG}_$(hostname).log"
_say() { echo "[qwen35-s15-prod48] $*" | tee -a "$LOG"; }
_die() { echo "[qwen35-s15-prod48] FATAL: $*" | tee -a "$LOG" >&2; exit 1; }
[[ -f "$_P48_BASE" ]] || _die "prod32 wrapper missing: $_P48_BASE"

# Removed migration knobs must not silently select an unintended starting point.
for _old in RESUME_SRC RESUME_ITER SRC_GBS PRIOR_CONSUMED OV2_TOTAL_SAMPLES; do
  [[ -z "${!_old:-}" ]] || _die "$_old is not supported. Start a new SAVE or resume this 48-GPU SAVE; use OV2_MIDTRAIN_N_SAMPLES for the full budget."
done
[[ -z "${ITERS:-}" ]] || _die "set OV2_MIDTRAIN_N_SAMPLES, not ITERS; train and LR schedule derive their length from the full budget."

export TP="${TP:-1}" NPROC="${NPROC:-4}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-240}"
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
[[ "$TP" == 1 && "$NPROC" == 4 ]] || _die "this launcher requires TP=1 and NPROC=4"
[[ "$OV2_MIDTRAIN_MUON" == 1 && "${OV2_FSDP:-0}" == 0 ]] || _die "this launcher requires Muon and torch_dist (OV2_FSDP=0)"
if [[ -n "${PET_NNODES:-}" ]]; then
  [[ "$PET_NNODES" == 12 ]] || _die "48 GPUs require PET_NNODES=12 (Workers=11), got $PET_NNODES"
elif [[ "${OV2_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  _die "PET_NNODES is missing; launch as a 12-pod PyTorchJob (or use OV2_PREFLIGHT_ONLY=1)"
fi
for _name in OV2_MIDTRAIN_GBS OV2_MIDTRAIN_N_SAMPLES; do
  [[ "${!_name}" =~ ^[1-9][0-9]*$ ]] || _die "$_name must be a positive decimal integer, got '${!_name}'"
done
DP=48
(( OV2_MIDTRAIN_GBS % DP == 0 )) || _die "GBS=$OV2_MIDTRAIN_GBS must divide into DP=48 equal batches; use 240 or 288"
MB_PER_RANK=$(( OV2_MIDTRAIN_GBS / DP ))
if [[ -n "${OV2_LENGTH_SORT_WINDOW:-}" && "$OV2_LENGTH_SORT_WINDOW" != 0 && "$OV2_LENGTH_SORT_WINDOW" != "$MB_PER_RANK" ]]; then
  _die "OV2_LENGTH_SORT_WINDOW must be $MB_PER_RANK (one iteration) or 0; unset it for automatic sizing"
fi
export ACCEL="${ACCEL:-2}"
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
export OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-1}"
export OV2_LENGTH_SORT_KEY="${OV2_LENGTH_SORT_KEY:-patches}"
export OV2_CUDA_MEM_FRACTION="${OV2_CUDA_MEM_FRACTION:-0.88}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export OV2_MEM_PROBE="${OV2_MEM_PROBE:-$MB_PER_RANK}"
export OV2_PHASE_TIMER="${OV2_PHASE_TIMER:-$MB_PER_RANK}"
export SAVE="${SAVE:-$HOME/ckpts_video_sft/ov2_qwen35_s15_seed85m_muon_tp1_dp48}"
export OV2_LR="${OV2_LR:-1e-5}" OV2_MIN_LR="${OV2_MIN_LR:-1e-6}"
_ITERS=$(( (OV2_MIDTRAIN_N_SAMPLES + OV2_MIDTRAIN_GBS - 1) / OV2_MIDTRAIN_GBS ))
_WARMUP="${OV2_WARMUP_ITERS:-$(( _ITERS * 2 / 1000 ))}"
if [[ -z "${OV2_WARMUP_ITERS:-}" ]] && (( _WARMUP < 1 )); then _WARMUP=1; fi

# Match Bridge's tracker precedence: latest_train_state.pt, then the legacy text tracker.
# Never inspect the highest iter_* directory: an interrupted later save may be incomplete.
# Import checkpoint readers only, not Bridge/FLA (no CUDA context/distributed groups).
if ! _RESUME_STEP="$(python3 - "$SAVE" "$OV2_MIDTRAIN_GBS" "$_ITERS" "$_WARMUP" "$OV2_LR" "$OV2_MIN_LR" "$LOG" <<'PY'
import logging
import sys
from pathlib import Path

save = Path(sys.argv[1])
logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler(sys.argv[7])])
try:
    for marker in (".metadata", "metadata.json", "run_config.yaml", "train_state.pt"):
        if (save / marker).exists():
            raise ValueError("SAVE must be a run root, not an iter_* checkpoint directory")
    tracker = save / "latest_train_state.pt"
    legacy = save / "latest_checkpointed_iteration.txt"
    if not tracker.exists() and not legacy.exists():
        sys.stdout.write("0\n")
        sys.exit(0)

    import torch
    import yaml

    if tracker.exists():
        step = int(torch.load(tracker, map_location="cpu", weights_only=True)["step"])
    else:
        step = int(legacy.read_text().strip())
    if step <= 0:
        raise ValueError(f"invalid checkpoint iteration {step}")
    checkpoint = save / f"iter_{step:07d}"
    for name in (".metadata", "metadata.json", "train_state.pt", "run_config.yaml"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"incomplete checkpoint: {checkpoint / name} is missing")
    local_step = int(torch.load(checkpoint / "train_state.pt", map_location="cpu", weights_only=True)["step"])
    if local_step != step:
        raise ValueError(f"tracker step {step} differs from checkpoint step {local_step}")
    expected_ranks = {f"train_dataloader_dprank{rank:03d}.pt" for rank in range(48)}
    actual_ranks = {path.name for path in checkpoint.glob("train_dataloader_dprank*.pt") if path.is_file()}
    if actual_ranks != expected_ranks:
        raise ValueError(f"{checkpoint} must contain exactly DP48 dataloader states; found {len(actual_ranks)}")
    config = yaml.safe_load((checkpoint / "run_config.yaml").read_text())
    expected = {
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                  "context_parallel_size": 1, "expert_model_parallel_size": 8},
        "train": {"global_batch_size": int(sys.argv[2]), "micro_batch_size": 1, "train_iters": int(sys.argv[3])},
        "optimizer": {"optimizer": "dist_muon", "lr": float(sys.argv[5]), "min_lr": float(sys.argv[6])},
        "scheduler": {"lr_warmup_iters": int(sys.argv[4]), "lr_decay_iters": int(sys.argv[3])},
        "checkpoint": {"ckpt_format": "torch_dist", "save_optim": True, "save_rng": True},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            actual = config.get(section, {}).get(key)
            if actual != value:
                raise ValueError(f"{section}.{key}: checkpoint={actual!r}, launch={value!r}; keep the original Args")
    sys.stdout.write(f"{step}\n")
except Exception as exc:
    logging.error("48-GPU resume preflight: %s", exc)
    sys.exit(1)
PY
)"; then
  _die "SAVE resume validation failed; see the checkpoint error above"
fi

if (( _RESUME_STEP > 0 )); then
  # prod32 validates INIT_CKPT even on resume. Use our own completed checkpoint so
  # restarting never depends on the original stage-2 files still being available.
  export INIT_CKPT="$SAVE/$(printf 'iter_%07d' "$_RESUME_STEP")"
  _say "resume: $INIT_CKPT; restore optimizer/scheduler/RNG/dataloader and continue to iter $_ITERS"
else
  _P48_POOL="${OV2_STAGE4_POOL:-/datasets/feilong-stage4-datasets}"
  export INIT_CKPT="${INIT_CKPT:-$_P48_POOL/35b/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006000}"
  [[ "$(basename "$INIT_CKPT")" == iter_* && -f "$INIT_CKPT/.metadata" ]] || _die "initial stage-2 checkpoint is missing: $INIT_CKPT"
  _say "fresh midtrain: stage-2 weights from $INIT_CKPT; full sample budget=$OV2_MIDTRAIN_N_SAMPLES"
fi
# On a fresh SAVE Bridge sets finetune=True when it falls back to pretrained_checkpoint.
# On a populated SAVE require a full resume; load the saved scheduler rather than re-warm.
export EXTRA_ARGS="${EXTRA_ARGS:-} logger.log_throughput=${OV2_LOG_THROUGHPUT:-true} checkpoint.finetune=false checkpoint.load_optim=true checkpoint.load_rng=true scheduler.override_opt_param_scheduler=false scheduler.use_checkpoint_opt_param_scheduler=true"
_say "world=48 dp=48 tp=1 ep=8 gbs=$OV2_MIDTRAIN_GBS mb_per_rank=$MB_PER_RANK n_samples=$OV2_MIDTRAIN_N_SAMPLES iters=$_ITERS warmup=$_WARMUP save=$SAVE"
_say "watch: grep -E 'iteration +[0-9]+/' \$HOME/train_logs/prod_qwen35_s15_${_P48_TAG}_*worker-10.log | tail -5"
if [[ "${OV2_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  _say "preflight passed; not launching"
  exit 0
fi
exec bash "$_P48_BASE"
