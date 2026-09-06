#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One 4-pod / 16-GPU workload: fresh custom/NCCL routing-map allgather arms.
# Both master and workers: Command=bash, Args=this script's absolute path.
# Optional Args: AB_STEPS=400 AB_DISCARD=100 AB_ORDER=custom,nccl AB_TIMEOUT_MIN=240
# See hybridep-ablation-16.md. Do not reuse the workload name after interruption.
set -euo pipefail
_AB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _kv in "$@"; do
  case "$_kv" in
    AB_STEPS=*|AB_DISCARD=*|AB_ORDER=*|AB_TIMEOUT_MIN=*|AB_ROOT=*|AB_PREFLIGHT_ONLY=*|INIT_CKPT=*|OV2_STAGE4_POOL=*|OV2_K8S_NAMESPACE=*|OV2_LLM_HF_QWEN35=*|OV2_HF_PROC_QWEN35_P16M33=*) export "$_kv" ;;
    *) echo "FATAL: unsupported ablation argument: $_kv" >&2; exit 1 ;;
  esac
done
exec python3 "$_AB_DIR/run_hybridep_ablation.py"
