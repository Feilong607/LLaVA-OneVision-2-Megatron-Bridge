#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build libcumemhook.so inside the training image. Finds cuda.h / cupti.h / libcupti in the usual places
# (CUDA toolkit, CUPTI extras, the pip nvidia-cuda-cupti wheel). Output: $OUT (default $HOME/cumem_hook/libcumemhook.so).
#   bash build.sh            -> builds and prints the LD_PRELOAD / LD_LIBRARY_PATH lines to put in the workload Args
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HOME/cumem_hook/libcumemhook.so}"
mkdir -p "$(dirname "$OUT")"

_first_dir() { for d in "$@"; do [[ -e "$d" ]] && { echo "$(dirname "$d")"; return 0; }; done; return 1; }
CUDA_INC="$(_first_dir /usr/local/cuda/include/cuda.h /usr/include/cuda.h)" || { echo "FATAL: cuda.h not found" >&2; exit 1; }
CUPTI_INC="$(_first_dir /usr/local/cuda/extras/CUPTI/include/cupti.h /usr/local/cuda/include/cupti.h /usr/include/cupti.h)" \
  || { echo "FATAL: cupti.h not found (CUDA toolkit 'extras/CUPTI' missing in this image)" >&2; exit 1; }
CUPTI_LIB="$(_first_dir /usr/local/cuda/extras/CUPTI/lib64/libcupti.so /usr/local/cuda/lib64/libcupti.so \
  /usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib/libcupti.so.12 /usr/lib/aarch64-linux-gnu/libcupti.so \
  /usr/lib/x86_64-linux-gnu/libcupti.so)" || { echo "FATAL: libcupti not found" >&2; exit 1; }
CUPTI_SO="$(ls "$CUPTI_LIB"/libcupti.so* | head -1)"
echo "cuda.h: $CUDA_INC  cupti.h: $CUPTI_INC  libcupti: $CUPTI_SO"

# libcupti is dlopen'ed at run time (absolute path via CUMEM_HOOK_CUPTI), so nothing CUDA-specific is linked here.
gcc -O2 -fPIC -shared -Wall -o "$OUT" "$HERE/cumem_hook.c" -I"$CUDA_INC" -I"$CUPTI_INC" -ldl -lpthread
echo "built $OUT"
ldd "$OUT" | grep -i 'not found' && { echo "FATAL: unresolved libs" >&2; exit 1; } || true
mkdir -p "$HOME/train_logs/cumem"
# smoke: load it in a throwaway python with CUMEM_HOOK=1 -> must print "armed" and create a ledger file
CUMEM_HOOK=1 CUMEM_HOOK_DIR=/tmp CUMEM_HOOK_CUPTI="$CUPTI_SO" LD_PRELOAD="$OUT" python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1").cuInit(0); print("python ok")' 2>&1 | tail -3
echo
echo "workload Args additions:"
echo "  LD_PRELOAD=$OUT CUMEM_HOOK=1 CUMEM_HOOK_DIR=$HOME/train_logs/cumem CUMEM_HOOK_CUPTI=$CUPTI_SO"
