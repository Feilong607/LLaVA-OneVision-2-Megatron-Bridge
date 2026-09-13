#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Qwen3.5-35B-A3B merged video stage (= the 30B line's s2+s3 with the data merged): 48-GPU production.
#
# Spec (BRINGUP §13): everything but the data matches the 30B video s2 production run
# (~/ckpts_video_sft/ov2_30b_a3b_stage2mix_v3_gbs32, run_config.yaml read 09-13):
#   objective   LM + MoE-aux 0.01, NO MTP head (OV2_MTP_LAYERS=0 -> the block is not built; runtime check in llava_ov2)
#   optimizer   Muon spectral, lr 1e-5 -> 1e-6 cosine, warmup 0.002 x iters, weight_decay 0.01 (optimizer AND
#               scheduler start/end), muon_extra_scale_factor 0.15, adam_beta2 0.95; all three siblings trainable
#   parallelism TP4 / ETP2 / EP8 / DP12 on 12 pods x 4 GPU (48 % (2*8) == 0), HybridEP (ACCEL=2; cap 18432 = 73728/4)
#   sequence    seq 73728 over 64k packs (0.6% packs dropped: shortmix/180s tokenizer+budget residue, accepted)
#   batch       GBS 48 = 4 microbatches per rank (30B s2 was GBS 32 on DP8); MBS 1; length-sorted batching auto (GBS/DP)
#   data        stage3_img38_video62_maveric.yaml (Barrett 0821 img38/video62 on MAVERIC paths; image side = 47m_v3
#               placeholder until the 56m SFT set arrives -- see the yaml header)
#   init        s1.5 final iter_0033334 (copied to ~/ckpts_keep; weights only, Muon state is rebuilt)
#   budget      sum of the blend weights = 966,252 samples = 20,131 iters @ GBS48 (OV2_MIDTRAIN_N_SAMPLES overrides;
#               960000 -> exactly 20,000 iters so the final save coincides with the last SAVE_EVERY save).
#               This is a weight-defined budget, NOT "one pass over every source": with per-part bin counts as
#               weights the video sources see ~1 epoch each, the 47m_v3 image placeholder only ~0.52 epoch.
#   saves       every 1000 iters, most_recent_k 3 (final + 2 before it); every OV2_ARCHIVE_EVERY=5000 a hard-linked
#               permanent copy under ${SAVE}_archive/ (master pod, background; mcore's rotation cannot reach it)
#
# Recompute lane (the ONE open knob, measured 09-13 on this blend, no MTP):
#   OV2_RECOMPUTE_FULL=1 (default)  full recompute, ~94 GiB predicted (115.2 measured WITH MTP) -- fits with margin,
#                                    0.726 samples/s measured with MTP (last-20 of 80 iters)
#   OV2_RECOMPUTE_FULL=0 OV2_RECOMPUTE_MOE=1 OV2_CUDA_MEM_FRACTION=0.92
#                                    selective attn+moe = the 30B s2/s3 lane; needed ~164 GiB at the 0.88 cap (OOM by 2),
#                                    the 0.92 cap (169.3) is the retry -- switch here only after the ladder passes it.
#
# First launch: fresh SAVE, weights from INIT_CKPT, full budget; a launch fingerprint (data yaml sha256, stream and
# topology settings) is written to $SAVE/ov2_launch_fingerprint.json. Restart: same Args + same SAVE (+ Workers=11);
# Bridge restores model, Muon, scheduler, iteration, RNG and Energon state. The preflight refuses a resume when
# (a) the two trackers (latest_train_state.pt / latest_checkpointed_iteration.txt) disagree or the text one is
# missing -- the model resumes from the former, Energon from the latter; (b) any stream-defining setting differs
# from the fingerprint (data yaml content, GBS, seq, workers/buffer, sort window/key, TP/ETP/EP, MTP, Muon
# constants, budget); (c) EXTRA_ARGS tries to override a protected key. Memory-only settings (recompute lane,
# OV2_CUDA_MEM_FRACTION, ACCEL) may change on resume and are logged. Never shrink the budget on restart.
# Changing TP/ETP/DP = new SAVE + INIT_CKPT=<last complete iter> (weights only).
#
# Form: Distributed/PyTorch, qwen35-fla image, gb200-nvl72-nodes, Workers=11, Command=bash (both sides),
# Args=this file's absolute path (both sides); optional KEY=VALUE args after it, e.g. SAVE=... INIT_CKPT=...
# OV2_RECOMPUTE_FULL=0 OV2_RECOMPUTE_MOE=1 OV2_CUDA_MEM_FRACTION=0.92. No env field needed.
# OV2_PREFLIGHT_ONLY=1 validates everything (topology, assets, all 127 data dirs, INIT/SAVE) without GPUs.
# Iteration lines print on the last rank -> worker-10's log; Step Time lines on rank 0 -> master's log.
# =============================================================================
set -euo pipefail

for _kv in "$@"; do
  if [[ "$_kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then export "$_kv"; else echo "FATAL: positional arg '$_kv' is not KEY=VALUE" >&2; exit 1; fi
done

_M48_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_M48_BASE="$_M48_DIR/ax_ov2_qwen35_35b_a3b_gb200.sh"
_M48_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
mkdir -p "$HOME/train_logs"
LOG="$HOME/train_logs/merged48_qwen35_${_M48_TAG}_$(hostname).log"
_say() { echo "[qwen35-merged48] $*" | tee -a "$LOG"; }
_die() { echo "[qwen35-merged48] FATAL: $*" | tee -a "$LOG" >&2; exit 1; }
[[ -f "$_M48_BASE" ]] || _die "base launcher missing: $_M48_BASE"

# ---- topology: 12 pods x 4 GPU, TP4/ETP2/EP8 -> DP12 --------------------------------------------------------
export TP=4 OV2_ETP=2 NPROC="${NPROC:-4}"
[[ "$NPROC" == 4 ]] || _die "this launcher requires NPROC=4"
if [[ -n "${PET_NNODES:-}" ]]; then
  [[ "$PET_NNODES" == 12 ]] || _die "48 GPUs require PET_NNODES=12 (Workers=11), got $PET_NNODES"
elif [[ "${OV2_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  _die "PET_NNODES is missing; launch as a 12-pod PyTorchJob (or use OV2_PREFLIGHT_ONLY=1)"
fi
DP=12
for _old in RESUME_SRC RESUME_ITER SRC_GBS PRIOR_CONSUMED OV2_TOTAL_SAMPLES ITERS; do
  [[ -z "${!_old:-}" ]] || _die "$_old is not supported; the budget is OV2_MIDTRAIN_N_SAMPLES and the schedule derives from it"
done

# ---- data: the merged blend; every dataset dir must be mounted in THIS pod --------------------------------------
export DATA_PATH="${DATA_PATH:-$_M48_DIR/stage3_img38_video62_maveric.yaml}"
[[ "$DATA_PATH" == */* ]] || DATA_PATH="$_M48_DIR/$DATA_PATH"
[[ -f "$DATA_PATH" ]] || _die "DATA_PATH not found: $DATA_PATH"
grep -q "__class__: Metadataset" "$DATA_PATH" || _die "DATA_PATH is not an energon Metadataset yaml: $DATA_PATH"
_n_ds=0; _missing=0
while read -r _d; do
  _n_ds=$((_n_ds+1)); [[ -d "$_d" ]] || { _say "MISSING dataset dir: $_d"; _missing=$((_missing+1)); }
done < <(grep -E '^\s*path:' "$DATA_PATH" | awk '{print $2}')
(( _n_ds > 0 )) || _die "no datasets in $DATA_PATH"
(( _missing == 0 )) || _die "$_missing/$_n_ds dataset dirs are not mounted (see above)"
_TOTAL_W="$(grep -E '^\s*(- )?weight:' "$DATA_PATH" | awk '{s+=$NF} END {print s+0}')"

# ---- budget / batch / schedule ------------------------------------------------------------------------------------
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-48}"
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-$_TOTAL_W}"   # weights are per-part bin counts -> 1 epoch
for _name in OV2_MIDTRAIN_GBS OV2_MIDTRAIN_N_SAMPLES; do
  [[ "${!_name}" =~ ^[1-9][0-9]*$ ]] || _die "$_name must be a positive decimal integer, got '${!_name}'"
done
(( OV2_MIDTRAIN_GBS % DP == 0 )) || _die "GBS=$OV2_MIDTRAIN_GBS must be a multiple of DP=12 (48 = 4 microbatches/rank)"
MB_PER_RANK=$(( OV2_MIDTRAIN_GBS / DP ))
if [[ -n "${OV2_LENGTH_SORT_WINDOW:-}" && "$OV2_LENGTH_SORT_WINDOW" != 0 && "$OV2_LENGTH_SORT_WINDOW" != "$MB_PER_RANK" ]]; then
  _die "OV2_LENGTH_SORT_WINDOW must be $MB_PER_RANK (one iteration) or 0; unset it for automatic sizing"
fi
_ITERS=$(( (OV2_MIDTRAIN_N_SAMPLES + OV2_MIDTRAIN_GBS - 1) / OV2_MIDTRAIN_GBS ))
_WARMUP="${OV2_WARMUP_ITERS:-$(( _ITERS * 2 / 1000 ))}"
if [[ -z "${OV2_WARMUP_ITERS:-}" ]] && (( _WARMUP < 1 )); then _WARMUP=1; fi

# ---- the §13 constants ---------------------------------------------------------------------------------------------
export OV2_SEQ_LEN=73728 ACCEL="${ACCEL:-2}"
export OV2_MTP_LAYERS=0                       # no MTP block/head (30B objective); llava_ov2 fails fast if any survives
export OV2_CE_FUSION="${OV2_CE_FUSION:-false}"
export OV2_MIDTRAIN_MUON=1 OV2_LR="${OV2_LR:-1e-5}" OV2_MIN_LR="${OV2_MIN_LR:-1e-6}" OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"
[[ "${OV2_FSDP:-0}" == 0 ]] || _die "this launcher requires torch_dist (OV2_FSDP=0)"
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
export OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"      # only read when OV2_RECOMPUTE_FULL=0
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-1}"
export OV2_CUDA_MEM_FRACTION="${OV2_CUDA_MEM_FRACTION:-0.88}"
[[ "$OV2_CUDA_MEM_FRACTION" =~ ^0\.[5-9][0-9]?$ ]] || _die "OV2_CUDA_MEM_FRACTION must be 0.50-0.99"
if [[ "$OV2_RECOMPUTE_FULL" == 0 && "$OV2_CUDA_MEM_FRACTION" == 0.88 ]]; then
  _die "selective recompute needed ~164 GiB on this blend (09-13 ladder S: OOM at the 0.88 cap); pass OV2_CUDA_MEM_FRACTION=0.92 or use OV2_RECOMPUTE_FULL=1"
fi
export OV2_MEM_PROBE="${OV2_MEM_PROBE:-$MB_PER_RANK}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export SAVE="${SAVE:-$HOME/ckpts_video_sft/ov2_qwen35_merged_img38_tp4_dp12}"

# ---- HF assets (config source + processor): the on-cluster extracts, not the pool's raw VL config ----------------
export OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$HOME/Qwen3.5-35B-A3B-text}"
export OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$HOME/qwen35_p16m33_auto_model}"
[[ -f "$OV2_LLM_HF_QWEN35/config.json" ]] || _die "OV2_LLM_HF_QWEN35 has no config.json: $OV2_LLM_HF_QWEN35"
[[ -f "$OV2_HF_PROC_QWEN35_P16M33/preprocessor_config.json" ]] || _die "OV2_HF_PROC_QWEN35_P16M33 has no preprocessor_config.json: $OV2_HF_PROC_QWEN35_P16M33"
[[ "$OV2_LLM_HF_QWEN35" == *"-text" ]] || _say "WARN: llm_hf=$OV2_LLM_HF_QWEN35 is not a '-text' extract"

# ---- EXTRA_ARGS may not override anything this wrapper pins (it would pass the preflight and change the run) -------
_USER_EXTRA="${EXTRA_ARGS:-}"
for _tok in $_USER_EXTRA; do
  case "$_tok" in
    model.tensor_model_parallel_size=*|model.expert_tensor_parallel_size=*|model.expert_model_parallel_size=*|model.pipeline_model_parallel_size=*|model.context_parallel_size=*|model.seq_length=*|model.mtp_num_layers=*|model.freeze_*|train.*|optimizer.*|scheduler.*|dataset.*|checkpoint.*)
      _die "EXTRA_ARGS override of a pinned key is refused: '$_tok' (change the wrapper or start a new SAVE deliberately)";;
  esac
done

# ---- resume vs fresh: tracker precedence + checkpoint completeness + same-topology/schedule check --------------
if ! _RESUME_STEP="$(python3 - "$SAVE" "$OV2_MIDTRAIN_GBS" "$_ITERS" "$_WARMUP" "$OV2_LR" "$OV2_MIN_LR" "$LOG" "$DP" <<'PY'
import logging
import sys
from pathlib import Path

save = Path(sys.argv[1]); dp = int(sys.argv[8])
logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler(sys.argv[7])])
try:
    for marker in (".metadata", "metadata.json", "run_config.yaml", "train_state.pt"):
        if (save / marker).exists():
            raise ValueError("SAVE must be a run root, not an iter_* checkpoint directory")
    tracker = save / "latest_train_state.pt"
    legacy = save / "latest_checkpointed_iteration.txt"
    if not tracker.exists() and not legacy.exists():
        sys.stdout.write("0\n"); sys.exit(0)
    import torch
    import yaml
    # The model/optimizer resume from latest_train_state.pt (Bridge), the Energon dataloader state from
    # latest_checkpointed_iteration.txt (base_energon_datamodule). Both must exist and agree, or model and
    # data would silently resume from different steps (or data from scratch).
    steps = {}
    if tracker.exists():
        steps["latest_train_state.pt"] = int(torch.load(tracker, map_location="cpu", weights_only=True)["step"])
    if legacy.exists():
        steps["latest_checkpointed_iteration.txt"] = int(legacy.read_text().strip())
    if "latest_checkpointed_iteration.txt" not in steps:
        raise ValueError("latest_checkpointed_iteration.txt is missing: the Energon dataloader would restart from scratch while the model resumes")
    if len(set(steps.values())) != 1:
        raise ValueError(f"trackers disagree: {steps}; model and data would resume from different steps -- repair the trackers first")
    step = steps["latest_checkpointed_iteration.txt"]
    if step <= 0:
        raise ValueError(f"invalid checkpoint iteration {step}")
    checkpoint = save / f"iter_{step:07d}"
    for name in (".metadata", "metadata.json", "train_state.pt", "run_config.yaml"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"incomplete checkpoint: {checkpoint / name} is missing")
    local_step = int(torch.load(checkpoint / "train_state.pt", map_location="cpu", weights_only=True)["step"])
    if local_step != step:
        raise ValueError(f"tracker step {step} differs from checkpoint step {local_step}")
    expected_ranks = {f"train_dataloader_dprank{rank:03d}.pt" for rank in range(dp)}
    actual_ranks = {p.name for p in checkpoint.glob("train_dataloader_dprank*.pt") if p.is_file()}
    if actual_ranks != expected_ranks:
        raise ValueError(f"{checkpoint} must contain exactly DP{dp} dataloader states; found {len(actual_ranks)}")
    config = yaml.safe_load((checkpoint / "run_config.yaml").read_text())
    expected = {
        "model": {"tensor_model_parallel_size": 4, "expert_tensor_parallel_size": 2, "pipeline_model_parallel_size": 1,
                  "context_parallel_size": 1, "expert_model_parallel_size": 8, "mtp_num_layers": None, "seq_length": 73728},
        "train": {"global_batch_size": int(sys.argv[2]), "micro_batch_size": 1, "train_iters": int(sys.argv[3])},
        "optimizer": {"optimizer": "dist_muon", "lr": float(sys.argv[5]), "min_lr": float(sys.argv[6]),
                      "weight_decay": 0.01, "adam_beta2": 0.95, "muon_extra_scale_factor": 0.15},
        "scheduler": {"lr_warmup_iters": int(sys.argv[4]), "lr_decay_iters": int(sys.argv[3])},
        "checkpoint": {"ckpt_format": "torch_dist", "save_optim": True, "save_rng": True},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            actual = config.get(section, {}).get(key)
            if actual != value:
                raise ValueError(f"{section}.{key}: checkpoint={actual!r}, launch={value!r}; keep the original Args (or start a new SAVE)")
    sys.stdout.write(f"{step}\n")
except Exception as exc:
    logging.error("merged48 resume preflight: %s", exc)
    sys.exit(1)
PY
)"; then
  _die "SAVE resume validation failed; see the checkpoint error above"
fi

# ---- launch fingerprint: stream-defining settings must match on resume; memory-only ones are logged ------------
_FP="$SAVE/ov2_launch_fingerprint.json"
_yaml_sha="$( (shasum -a 256 "$DATA_PATH" 2>/dev/null || sha256sum "$DATA_PATH") | awk '{print $1}')"
[[ "$_yaml_sha" =~ ^[0-9a-f]{64}$ ]] || _die "could not hash $DATA_PATH"
_STREAM_FP="data_sha256=$_yaml_sha data_path=$DATA_PATH gbs=$OV2_MIDTRAIN_GBS n_samples=$OV2_MIDTRAIN_N_SAMPLES iters=$_ITERS warmup=$_WARMUP seq=$OV2_SEQ_LEN tp=4 etp=2 ep=8 dp=12 mtp_layers=0 workers=${OV2_NUM_WORKERS:-2} buffer=${OV2_SHUFFLE_BUFFER:-16} sort_window=${OV2_LENGTH_SORT_WINDOW:-$MB_PER_RANK} sort_key=${OV2_LENGTH_SORT_KEY:-tokens} muon=1 lr=$OV2_LR min_lr=$OV2_MIN_LR wd=0.01 extra=0.15 beta2=0.95 aux=$OV2_MOE_AUX_LOSS_COEFF"
_MEM_FP="recompute_full=$OV2_RECOMPUTE_FULL recompute_moe=$OV2_RECOMPUTE_MOE vision_recompute=$OV2_VISION_RECOMPUTE mem_fraction=$OV2_CUDA_MEM_FRACTION accel=$ACCEL"
if (( _RESUME_STEP > 0 )); then
  [[ -f "$_FP" ]] || _die "SAVE has checkpoints but no $_FP -- cannot prove the data stream/topology are unchanged; restore the fingerprint or start a new SAVE"
  _old_stream="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stream"])' "$_FP")"
  _old_mem="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("memory",""))' "$_FP")"
  if [[ "$_old_stream" != "$_STREAM_FP" ]]; then
    _say "fingerprint (saved):  $_old_stream"; _say "fingerprint (launch): $_STREAM_FP"
    _die "stream-defining settings differ from the SAVE's fingerprint (see the two lines above); resume refused -- a changed data yaml/GBS/seq/topology needs a new SAVE"
  fi
  [[ "$_old_mem" == "$_MEM_FP" ]] || _say "memory-only settings changed on resume (allowed): saved [$_old_mem] -> launch [$_MEM_FP]"
elif [[ "${OV2_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  mkdir -p "$SAVE"
  python3 -c 'import json,sys,time; json.dump({"stream": sys.argv[2], "memory": sys.argv[3], "written": time.strftime("%Y-%m-%d %H:%M:%S")}, open(sys.argv[1], "w"), indent=1)' "$_FP" "$_STREAM_FP" "$_MEM_FP"
  _say "fingerprint written: $_FP"
fi

if (( _RESUME_STEP > 0 )); then
  export INIT_CKPT="$SAVE/$(printf 'iter_%07d' "$_RESUME_STEP")"
  _say "resume: $INIT_CKPT; restore optimizer/scheduler/RNG/dataloader and continue to iter $_ITERS"
else
  export INIT_CKPT="${INIT_CKPT:-$HOME/ckpts_keep/ov2_qwen35_s15_iter_0033334}"
  # Must be ONE checkpoint directory (an iter_* dir or a copy of one), never a run root: a root resolves through the
  # tracker, which can point at a save that no longer exists (the 6094 trap). The ~/ckpts_keep copy is not named iter_*.
  [[ -f "$INIT_CKPT/.metadata" || -f "$INIT_CKPT/metadata.json" ]] || _die "INIT_CKPT has no torch_dist metadata (point AT the checkpoint dir, not the run root): $INIT_CKPT"
  [[ ! -e "$INIT_CKPT/latest_train_state.pt" && ! -e "$INIT_CKPT/latest_checkpointed_iteration.txt" ]] || _die "INIT_CKPT looks like a run root (has a tracker): $INIT_CKPT"
  _say "fresh merged stage: weights from $INIT_CKPT (mtp.* tensors in it are ignored by the no-MTP model); budget=$OV2_MIDTRAIN_N_SAMPLES"
fi

# Muon s2 constants pinned AFTER the recipe's midtrain defaults; scheduler wd must equal optimizer wd every step.
# most_recent_k bounds the SAVE (~340 GB per save); OV2_KEEP_CKPTS raises the window (3 = final + two before it).
export EXTRA_ARGS="${EXTRA_ARGS:-} optimizer.muon_scale_mode=spectral optimizer.muon_extra_scale_factor=0.15 optimizer.adam_beta2=0.95 optimizer.weight_decay=0.01 scheduler.start_weight_decay=0.01 scheduler.end_weight_decay=0.01 model.freeze_language_model=false model.freeze_vision_model=false model.freeze_adapter=false checkpoint.most_recent_k=${OV2_KEEP_CKPTS:-3} logger.log_throughput=${OV2_LOG_THROUGHPUT:-true} checkpoint.finetune=false checkpoint.load_optim=true checkpoint.load_rng=true scheduler.override_opt_param_scheduler=false scheduler.use_checkpoint_opt_param_scheduler=true"

# ---- permanent archive every OV2_ARCHIVE_EVERY iters (master pod only; 0 disables) ------------------------------
# mcore only knows most_recent_k. This background loop watches the tracker and, once a save at a multiple of
# OV2_ARCHIVE_EVERY is complete (tracker step >= N, metadata + train_state present), hard-links it to
# ${SAVE}_archive/iter_N (cp -al: zero extra space; rotation later unlinks the original names, the data stays).
# Falls back to a real copy if hard links fail. Idempotent across restarts (skips archives that already exist).
# Kept OUTSIDE $SAVE so mcore's iter_* rotation never sees it.
export OV2_ARCHIVE_EVERY="${OV2_ARCHIVE_EVERY:-5000}"
[[ "$OV2_ARCHIVE_EVERY" =~ ^[0-9]+$ ]] || _die "OV2_ARCHIVE_EVERY must be an integer (0 disables)"
if (( OV2_ARCHIVE_EVERY > 0 )) && [[ "$(hostname)" == *-master-0 && "${OV2_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  _ARCHIVE="${OV2_ARCHIVE_DIR:-${SAVE}_archive}"
  mkdir -p "$_ARCHIVE"
  ( while sleep 60; do
      _step="$(python3 - "$SAVE" <<'PY2' 2>/dev/null
import sys, pathlib
save = pathlib.Path(sys.argv[1]); t = save / "latest_train_state.pt"; l = save / "latest_checkpointed_iteration.txt"
try:
    if t.exists():
        import torch
        print(int(torch.load(t, map_location="cpu", weights_only=True)["step"]))
    elif l.exists():
        print(int(l.read_text().strip()))
    else:
        print(0)
except Exception:
    print(0)
PY2
)"
      [[ "$_step" =~ ^[0-9]+$ ]] || continue
      for (( _n=OV2_ARCHIVE_EVERY; _n<=_step; _n+=OV2_ARCHIVE_EVERY )); do
        _src="$SAVE/$(printf 'iter_%07d' "$_n")"; _dst="$_ARCHIVE/$(printf 'iter_%07d' "$_n")"
        [[ -d "$_src" && -f "$_src/train_state.pt" && ( -f "$_src/.metadata" || -f "$_src/metadata.json" ) ]] || continue
        [[ -f "$_dst/.archived" ]] && continue
        rm -rf "$_dst.tmp"
        if cp -al "$_src" "$_dst.tmp" 2>/dev/null || cp -a "$_src" "$_dst.tmp"; then
          if [[ "$(ls "$_src" | wc -l)" == "$(ls "$_dst.tmp" | wc -l)" ]]; then
            mv "$_dst.tmp" "$_dst" && date '+%F %T' > "$_dst/.archived" && echo "[qwen35-merged48] archived $_src -> $_dst" >> "$LOG"
          else
            echo "[qwen35-merged48] WARN: archive of $_src incomplete, will retry" >> "$LOG"; rm -rf "$_dst.tmp"
          fi
        fi
      done
    done ) &
  _say "archiver: every $OV2_ARCHIVE_EVERY iters -> $_ARCHIVE (hard links; pid $!)"
fi

_lane="full"; [[ "$OV2_RECOMPUTE_FULL" == 1 ]] || _lane="selective(moe=$OV2_RECOMPUTE_MOE)"
_say "world=48 tp=4 etp=2 ep=8 dp=12 gbs=$OV2_MIDTRAIN_GBS mb_per_rank=$MB_PER_RANK seq=$OV2_SEQ_LEN accel=$ACCEL recompute=$_lane mem_fraction=$OV2_CUDA_MEM_FRACTION mtp_layers=0"
_say "blend=$DATA_PATH ($_n_ds dirs, total weight $_TOTAL_W) n_samples=$OV2_MIDTRAIN_N_SAMPLES iters=$_ITERS warmup=$_WARMUP save_every=$SAVE_EVERY keep=${OV2_KEEP_CKPTS:-3} archive_every=$OV2_ARCHIVE_EVERY"
_say "muon: lr $OV2_LR->$OV2_MIN_LR wd 0.01 extra 0.15 beta2 0.95 (= 30B video s2 run_config) save=$SAVE"
_say "watch: grep -E 'iteration +[0-9]+/' $SAVE/train_node11.log | tail -3 ; grep -h 'Step Time' $SAVE/train_node0.log | tail -3   (base launcher tees per-node logs into SAVE)"
if [[ "${OV2_PREFLIGHT_ONLY:-0}" == 1 ]]; then _say "preflight passed; not launching"; exit 0; fi
exec bash "$_M48_BASE"
