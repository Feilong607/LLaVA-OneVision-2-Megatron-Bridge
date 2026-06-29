#!/usr/bin/env bash
# Watchdog for the Qwen3.5-35B-A3B OV2 stage-1 run on this node (multi-tenant -> external kills happen).
# Runs ON THE HOST (drives docker). Auto-resumes from checkpoint on crash/kill, stops at epoch end,
# refuses to resume on NaN. Launch detached: setsid nohup bash watchdog_qwen35_stage1.sh &
set -uo pipefail
LAUNCHER=/ov2/feilong/gb200/Megatron-Bridge/examples/models/qwen/qwen35_vl_ov2/A800/ax_ov2_qwen35_35b_a3b_stage1.sh
SAVE=/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage1
LOG="$SAVE/train_node0.log"
WLOG="$SAVE/watchdog.log"
NAME=ov2_qwen35_s1
TARGET=2181
INTERVAL="${INTERVAL:-600}"     # 10 min
say(){ echo "$(date '+%F %T') $*" >> "$WLOG"; }
say "watchdog START (target iter=$TARGET, interval=${INTERVAL}s)"
miss=0
while true; do
  sleep "$INTERVAL"
  it=$(grep -oE "iteration +[0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  it=${it:-0}
  nan=$(grep -cE "nan iterations: *[1-9]" "$LOG" 2>/dev/null || echo 0)
  running=$(docker ps --filter "name=$NAME" --format '{{.Names}}' 2>/dev/null)
  # completion: reached target iters (training loop prints up to TARGET, then saves + exits)
  if [[ "$it" -ge "$TARGET" ]]; then say "EPOCH DONE at iter=$it -> stop watchdog"; break; fi
  if [[ "$nan" -ge 1 ]]; then say "NaN detected (iter=$it) -> NOT resuming, stop watchdog (investigate)"; break; fi
  if [[ -n "$running" ]]; then miss=0; say "ok iter=$it running"; continue; fi
  # container not running and not done and no NaN -> confirm it's really gone (2 strikes), then resume
  miss=$((miss+1))
  say "container DOWN (strike $miss) at iter=$it"
  if [[ "$miss" -ge 2 ]]; then
    say "RESUME from checkpoint (re-run launcher; checkpoint.load=\$SAVE auto-resumes)"
    bash "$LAUNCHER" >> "$WLOG" 2>&1
    miss=0
    sleep 120
  fi
done
say "watchdog EXIT"
