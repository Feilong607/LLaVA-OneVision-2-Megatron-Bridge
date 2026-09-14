#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Same Args on master/workers. Workers=11 or 15; four GPUs/pod.
# See long-context-matrix.md. Fresh LC_ROOT/workload name for each experiment.
set -euo pipefail
_LC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _kv in "$@"; do
  case "$_kv" in
    LC_STAGE=*|LC_TPS=*|LC_STEPS=*|LC_SPLIT=*|LC_DISCARD=*|LC_TIMEOUT_MIN=*|LC_ROOT=*|LC_WORLD=*|LC_PREFLIGHT_ONLY=*|LC_SEQ_LEN=*|LC_MIN_LONG_TOKENS=*|INIT_CKPT=*|OV2_STAGE4_POOL=*|OV2_K8S_NAMESPACE=*|OV2_LLM_HF_QWEN35=*|OV2_HF_PROC_QWEN35_P16M33=*|OV2_EXTRA_PYLIBS=*) export "$_kv" ;;
    *) echo "FATAL: unsupported long-context argument: $_kv" >&2; exit 1 ;;
  esac
done
exec python3 "$_LC_DIR/run_long_context_matrix.py"
