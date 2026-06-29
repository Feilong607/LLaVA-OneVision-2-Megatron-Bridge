#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B throughput benchmark harness for GB200. Wraps ax_ov2_30b_a3b_gb200.sh,
# runs a SHORT run per lever-config, parses steady-state iter/tokens-s-GPU/MFU/fwd/bwd,
# tabulates so you can A/B the throughput levers (dispatcher / recompute / fusions / graphs).
#
# Each config = the SAME launcher with a different lever set. It cold-loads the real
# pretrained weights ($INIT_CKPT) into a FRESH scratch ckpt dir (no resume, so train_iters
# =BENCH_ITERS actually runs), saves NOTHING. PRIMARY comparison metric = iter_ms (lower=faster,
# always present, unambiguous); tokens/s/GPU & MFU are secondary (need the perf line).
#
# USAGE (multi-node: run the SAME command on BOTH nodes with LIST_IP, in lockstep per config):
#   LIST_IP="<ip0> <ip1>" bash bench_ov2_gb200.sh run  recompute_off      # one config
#   LIST_IP="<ip0> <ip1>" bash bench_ov2_gb200.sh sweep                   # all configs, in order
#   bash bench_ov2_gb200.sh report                                        # print results table
#   bash bench_ov2_gb200.sh list                                          # show configs
# Knobs: BENCH_ITERS(=12) WARMUP_SKIP(=5) OV2_NUM_WORKERS(=8, must HIDE data) BENCH_SAVE(scratch)
#        BENCH_BASE_PORT(=26100, per-config offset auto-added)
# NOTE: only global-rank-0 (NODE_RANK 0) prints the perf metrics -> only that node appends the CSV;
#       other nodes just run. Logs are per-host so the shared /ov2 mount does not clobber them.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/ax_ov2_30b_a3b_gb200.sh"
[[ -f "$LAUNCHER" ]] || { echo "FATAL: launcher not found: $LAUNCHER" >&2; exit 1; }

BENCH_DIR="${BENCH_DIR:-$HERE/bench_out}"; mkdir -p "$BENCH_DIR"
RESULTS="$BENCH_DIR/results.csv"
HOST="$(hostname -s 2>/dev/null || hostname)"
BENCH_ITERS="${BENCH_ITERS:-12}"           # short run per config
WARMUP_SKIP="${WARMUP_SKIP:-5}"            # drop first N logged iters (graph capture + dataloader ramp + cuBLAS plans)
BENCH_SAVE="${BENCH_SAVE:-/home/ftan0055/_ov2_bench_ckpt}"   # FRESH scratch: cold-load pretrained, no resume, save nothing
BENCH_BASE_PORT="${BENCH_BASE_PORT:-26100}"
CSV_HEADER="config,status,iter_ms,tokens_s_gpu,mfu_pct,fwd_ms,bwd_ms,batchgen_ms,gpu_util,host"

# --- lever matrix --------------------------------------------------------------
# CFG_ENV[name]   = launcher ENV knobs (ACCEL / recompute / permute / dispatcher / DEVICE_MAX_CONN).
# CFG_XARGS[name] = extra run_recipe dotted overrides (via EXTRA_ARGS, appended last => wins).
#   Tier-2 fields (grouped_gemm/router_fusion/cuda_graph) MAY be clobbered by build_llava_ov2's HF LLM
#   rebuild -> if a config does not move vs baseline, that lever needs provider wiring, not a CLI flag.
declare -A CFG_ENV CFG_XARGS
# Tier-1 env levers all CONSUMED by the launcher / build_llava_ov2 (verified). model.* CLI is CLOBBERED by
# the HF LLM rebuild -> the real enablers are env vars wired as post-build force-sets in llava_ov2.py.
CFG_ENV[baseline]="ACCEL=0";                                                                CFG_XARGS[baseline]=""
CFG_ENV[recompute_off]="ACCEL=0 DISABLE_RECOMPUTE=1";                                       CFG_XARGS[recompute_off]=""
CFG_ENV[recompute_full]="ACCEL=0 OV2_RECOMPUTE_FULL=1";                                     CFG_XARGS[recompute_full]=""
CFG_ENV[router_fusion]="ACCEL=0 DISABLE_RECOMPUTE=1 OV2_MOE_ROUTER_FUSION=1";               CFG_XARGS[router_fusion]=""
CFG_ENV[shared_overlap]="ACCEL=0 DISABLE_RECOMPUTE=1 OV2_MOE_SHARED_EXPERT_OVERLAP=1";      CFG_XARGS[shared_overlap]=""   # alltoall-only
CFG_ENV[permute]="ACCEL=0 DISABLE_RECOMPUTE=1 OV2_MOE_PERMUTE_FUSION=1";                    CFG_XARGS[permute]=""
CFG_ENV[hybridep]="ACCEL=2 DISABLE_RECOMPUTE=1";                                            CFG_XARGS[hybridep]=""
CFG_ENV[fp8]="ACCEL=1 DISABLE_RECOMPUTE=1";                                                 CFG_XARGS[fp8]=""              # MXFP8; mid-train accuracy risk -> A/B loss
CFG_ENV[fp8_pad]="ACCEL=1 DISABLE_RECOMPUTE=1 OV2_MOE_PERMUTE_FUSION=1 OV2_MOE_ROUTER_FUSION=1 OV2_MOE_ROUTER_PAD_FP8=1"; CFG_XARGS[fp8_pad]=""
CFG_ENV[manualgc]="ACCEL=0 DISABLE_RECOMPUTE=1 CUDA_DEVICE_MAX_CONNECTIONS=32";             CFG_XARGS[manualgc]="train.manual_gc=true train.manual_gc_interval=10"
CFG_ENV[cudagraph]="ACCEL=0 DISABLE_RECOMPUTE=1";                                           CFG_XARGS[cudagraph]="model.cuda_graph_impl=transformer_engine model.cuda_graph_scope=[attn,moe_router,moe_preprocess] model.use_te_rng_tracker=true"
CFG_ENV[best_bf16]="ACCEL=0 DISABLE_RECOMPUTE=1 OV2_MOE_PERMUTE_FUSION=1 OV2_MOE_ROUTER_FUSION=1 OV2_MOE_SHARED_EXPERT_OVERLAP=1 CUDA_DEVICE_MAX_CONNECTIONS=32"; CFG_XARGS[best_bf16]="train.manual_gc=true train.manual_gc_interval=10"
CFG_ENV[best_fp8]="ACCEL=1 DISABLE_RECOMPUTE=1 OV2_MOE_PERMUTE_FUSION=1 OV2_MOE_ROUTER_FUSION=1 OV2_MOE_ROUTER_PAD_FP8=1 CUDA_DEVICE_MAX_CONNECTIONS=32"; CFG_XARGS[best_fp8]="train.manual_gc=true train.manual_gc_interval=10"
CONFIG_ORDER="baseline recompute_off recompute_full router_fusion shared_overlap permute hybridep fp8 fp8_pad manualgc cudagraph best_bf16 best_fp8"

# stable per-config port (same on both nodes -> they rendezvous; differs across configs -> no TIME_WAIT clash)
_port_for() { echo $(( BENCH_BASE_PORT + $(echo -n "$1" | cksum | cut -d" " -f1) % 800 )); }

# median of stdin numbers (one per line), skipping the first WARMUP_SKIP
_median_skip() {
  tail -n +"$((WARMUP_SKIP+1))" | sort -n | awk '
    {a[NR]=$1}
    END{ if(NR<1){print "NA"} else if(NR%2){printf "%.4g\n",a[(NR+1)/2]} else {printf "%.4g\n",(a[NR/2]+a[NR/2+1])/2} }'
}
# grab the number that FOLLOWS a label (regex matches up to the number; no trailing unit) -> series
_series() { grep -oE "$2" "$1" 2>/dev/null | grep -oE "[0-9.eE+-]+$"; }

_init_csv() { [[ -s "$RESULTS" ]] || echo "$CSV_HEADER" > "$RESULTS"; }

parse_one() {  # $1=name $2=log  -> append CSV row IF this node logged the perf metrics
  local name="$1" log="$2" nlines
  nlines=$(grep -cE "elapsed time per iteration" "$log" 2>/dev/null || echo 0)
  if [[ "$nlines" -eq 0 ]]; then
    if grep -qiE "Traceback|CUDA error|out of memory|AssertionError|RuntimeError|rendezvous" "$log" 2>/dev/null; then
      _init_csv; echo "$name,FAIL,NA,NA,NA,NA,NA,NA,NA,$HOST" >> "$RESULTS"
      echo "[bench] $name -> FAIL on $HOST (see $log)"
    else
      echo "[bench] $name -> no perf metrics on $HOST (expected on non-rank-0 node). Not appending CSV."
    fi
    return
  fi
  local iter tps mfu fwd bwd bgen util
  iter=$(_series "$log" "elapsed time per iteration \(ms\): ?[0-9.eE+-]+" | _median_skip)
  tps=$( _series "$log" "tokens/s/GPU: ?[0-9.eE+-]+"                      | _median_skip)   # NEW field uses ": " (not the legacy "=3238" line)
  mfu=$( _series "$log" "MFU: ?[0-9.eE+-]+"                             | _median_skip)
  fwd=$( _series "$log" "forward[-=: ]+compute[: ]+[0-9.eE+-]+"          | _median_skip)   # timing_log_level=2 block
  [[ "$fwd" == "NA" || -z "$fwd" ]] && fwd=$(_series "$log" "forward[=:][ ]*[0-9.eE+-]+" | _median_skip)  # fallback: custom "forward=NNNN"
  bwd=$( _series "$log" "backward[-=: ]+compute[: ]+[0-9.eE+-]+"         | _median_skip)
  [[ "$bwd" == "NA" || -z "$bwd" ]] && bwd=$(_series "$log" "backward[=:][ ]*[0-9.eE+-]+" | _median_skip)
  bgen=$(_series "$log" "batch-generator[: ]+[0-9.eE+-]+"               | _median_skip)   # DATA time: if ~= iter, the bench is DATA-bound (raise OV2_NUM_WORKERS)
  util=$(_series "$log" "GPU utilization: ?[0-9.eE+-]+"                 | _median_skip)
  for x in tps mfu fwd bwd bgen util; do [[ -z "${!x}" ]] && eval "$x=NA"; done
  _init_csv
  echo "$name,OK,${iter:-NA},$tps,$mfu,$fwd,$bwd,$bgen,$util,$HOST" >> "$RESULTS"
  echo "[bench] $name -> iter_ms=$iter tokens/s/GPU=$tps MFU%=$mfu fwd=$fwd bwd=$bwd batchgen_ms=$bgen util=$util"
  [[ "$bgen" != "NA" && "$iter" != "NA" ]] && awk -v b="$bgen" -v i="$iter" 'BEGIN{ if(b>0.25*i) printf "[bench] WARN: batch-generator %.0fms is >25%% of iter %.0fms -> DATA-BOUND, raise OV2_NUM_WORKERS (compute number is invalid)\n", b, i }'
}

run_one() {
  local name="$1"
  [[ -n "${CFG_ENV[$name]+x}" ]] || { echo "unknown config '$name' (see: bench_ov2_gb200.sh list)" >&2; return 2; }
  local log="$BENCH_DIR/bench_${name}_${HOST}.log"     # per-HOST log -> no shared-FS clobber
  local port; port="$(_port_for "$name")"
  echo "[bench] ===== config=$name on $HOST :: ${CFG_ENV[$name]} ${CFG_XARGS[$name]:+| EXTRA=${CFG_XARGS[$name]}} | port=$port"
  ( set -e
    export ITERS="$BENCH_ITERS" SAVE_EVERY=999999 LOG_EVERY=1 OV2_WARMUP_ITERS=1
    export OV2_TIMING_LOG_LEVEL=2 OV2_TIMING_PRINT_INTERVAL=1
    export OV2_PARALLEL_SHARD_ITERS=1 OV2_NUM_WORKERS="${OV2_NUM_WORKERS:-8}"   # >=8 to HIDE data (shard_iters=1 keeps FDs bounded)
    export MASTER_PORT="$port"
    eval "export ${CFG_ENV[$name]}"
    # fresh scratch ckpt: load $INIT_CKPT as pretrained (cold start), no resume, save nothing.
    # EXTRA_ARGS is appended AFTER $OVERRIDES in the launcher -> last-wins overrides checkpoint.{load,save}.
    export EXTRA_ARGS="checkpoint.load=$BENCH_SAVE checkpoint.save=$BENCH_SAVE dataset.dataloader_save=$BENCH_SAVE logger.tensorboard_dir=$BENCH_SAVE/tb ${CFG_XARGS[$name]}"
    mkdir -p "$BENCH_SAVE"
    bash "$LAUNCHER"
  ) 2>&1 | tee "$log"
  parse_one "$name" "$log"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  run)    [[ $# -ge 1 ]] || { echo "usage: bench_ov2_gb200.sh run <config>"; exit 1; }; run_one "$1" ;;
  sweep)  list="${*:-$CONFIG_ORDER}"
          for c in $list; do run_one "$c" || true; echo "[bench] settle 40s (NCCL/port teardown) before next config"; sleep 40; done
          echo; echo "===== RESULTS ($HOST) ====="; [[ -s "$RESULTS" ]] && column -s, -t < "$RESULTS" || echo "(no metrics on $HOST -- check the rank-0 node)" ;;
  report) [[ -s "$RESULTS" ]] && column -s, -t < "$RESULTS" || echo "no results yet ($RESULTS)" ;;
  list)   echo "configs (run order):"; for c in $CONFIG_ORDER; do printf "  %-16s port=%-6s %s %s\n" "$c" "$(_port_for "$c")" "${CFG_ENV[$c]}" "${CFG_XARGS[$c]:+| EXTRA=${CFG_XARGS[$c]}}"; done ;;
  *)      sed -n '2,24p' "${BASH_SOURCE[0]}" ;;
esac
