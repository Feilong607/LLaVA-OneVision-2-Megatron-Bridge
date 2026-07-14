#!/usr/bin/env bash
# Per-rank GPU-local NUMA binding for `torchrun --no-python` (enabled via OV2_NUMA_BIND=1 in
# ax_ov2_30b_a3b_gb200.sh). Mechanism mirrors upstream Megatron-Bridge #4630
# (_kubeflow_numa_binding_script): resolve THIS rank's GPU PCI bus via nvidia-smi ->
# /sys/bus/pci/devices/<bus>/numa_node -> exec python under numactl cpu+mem binding, so each rank's
# host threads/allocations stay on its GPU's local Grace socket (NVIDIA rates CPU affinity ~+15% on
# GB200 where host overhead co-limits). SOFT-FALLBACK: any resolution failure logs and runs UNBOUND
# -- an opt-in perf lever must never kill the run (upstream exits 1; we deliberately do not).
set -euo pipefail
_lr="${LOCAL_RANK:-}"
_bind=""
# nvidia-smi -i indexes PHYSICAL GPUs, but LOCAL_RANK indexes the process's VISIBLE GPUs. If
# CUDA_VISIBLE_DEVICES masks/reorders devices (e.g. "2,3"), map local_rank -> physical index through
# it, else we'd bind to the wrong GPU's socket (a silent perf regression, the opposite of the win).
_phys="$_lr"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -n "$_lr" ]]; then
  IFS=',' read -ra _vis <<< "$CUDA_VISIBLE_DEVICES"
  # Only remap when the visible list is plain integer indices (skip UUID/MIG forms we can't -i on).
  if [[ "${_vis[$_lr]:-}" =~ ^[0-9]+$ ]]; then _phys="${_vis[$_lr]}"; else _phys=""; fi
fi
if [[ -n "$_phys" ]] && command -v nvidia-smi >/dev/null 2>&1 && command -v numactl >/dev/null 2>&1; then
  # nvidia-smi prints "00000000:1B:00.0"; sysfs wants lowercase with a 4-digit domain ("0000:1b:00.0").
  _pci=$(nvidia-smi -i "$_phys" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
      | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]' | sed -E 's/^00000000:/0000:/') || _pci=""
  _nf="/sys/bus/pci/devices/${_pci}/numa_node"
  if [[ -n "$_pci" && -r "$_nf" ]]; then
    _node=$(<"$_nf")
    # numa_node is -1 when the platform reports no affinity -> regex rejects it -> fallback.
    [[ "$_node" =~ ^[0-9]+$ ]] && _bind="$_node"
  fi
fi
if [[ -n "$_bind" ]]; then
  echo "[ov2-numa] host=$(hostname) rank=${RANK:-?} local_rank=$_lr numa=$_bind"
  exec numactl --cpunodebind="$_bind" --membind="$_bind" python "$@"
fi
echo "[ov2-numa] WARN: local_rank=${_lr:-?} could not resolve a NUMA node -> running UNBOUND" >&2
exec python "$@"
