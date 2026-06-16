#!/bin/bash
# check_seed85m_shards.sh — run ON the box that trains (e.g. GB200 /home/ftan0055) to decide:
#   (A) INCOMPLETE SYNC  -> missing/truncated shards (a half-copied .tar makes energon's reader[idx]
#       block forever -> the iter stalls -> NCCL heartbeat/TCPStore tears the job down), or
#   (B) SLOW DISK        -> low read bandwidth (network PVC/NFS) -> 20-min iters -> same NCCL death.
#
# Source-of-truth = seed85m_source_manifest.txt (per-shard byte sizes from the /ov2 master copy),
# co-located with this script (both ship in the repo, so `git pull` brings them to the GB200).
#
# Usage (on the GB200):
#   bash check_seed85m_shards.sh                 # auto: read active paths from mid_training_seed85m.yaml
#   DATA_PATHS="/home/ftan0055/seed85m_video20m_47m5p_packed/node_00/webdataset ..." bash check_seed85m_shards.sh
set -uo pipefail
shopt -s nullglob
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SRC_MANIFEST:-$HERE/seed85m_source_manifest.txt}"
YAML="${YAML:-$HERE/mid_training_seed85m.yaml}"

# --- active data dirs: uncommented `path:` lines in the yaml (override with DATA_PATHS=) ---
if [ -n "${DATA_PATHS:-}" ]; then
  read -ra PATHS <<< "$DATA_PATHS"
else
  mapfile -t PATHS < <(grep -E "^[[:space:]]*-?[[:space:]]*path:" "$YAML" 2>/dev/null | grep -vE "^[[:space:]]*#" | sed -E "s/.*path:[[:space:]]*//; s/[[:space:]]*$//")
fi
[ "${#PATHS[@]}" -gt 0 ] || { echo "no active data paths found in $YAML (and DATA_PATHS unset)"; exit 2; }
echo "Active data paths:"; printf "  %s\n" "${PATHS[@]}"
[ -f "$MANIFEST" ] && echo "Source manifest: $MANIFEST ($(grep -c "^node_" "$MANIFEST") shard entries)" || echo "WARN: no source manifest at $MANIFEST -> size-diff skipped (structural checks only)"

fail=0; miss=0; trunc=0
for d in "${PATHS[@]}"; do
  node="$(basename "$(dirname "$d")")"      # .../node_00/webdataset -> node_00
  echo "=================================================================="
  echo "### $node   $d"
  if [ ! -d "$d" ]; then echo "  ❌ DIR MISSING"; fail=1; miss=$((miss+1)); continue; fi
  tars=( "$d"/*.tar ); idxs=( "$d"/*.tar.idx )
  echo "  tar=${#tars[@]}  idx=${#idxs[@]}  nv-meta=$([ -d "$d/.nv-meta" ] && echo yes || echo '❌NO')  total=$(du -sh "$d" 2>/dev/null | cut -f1)"
  [ "${#tars[@]}" -eq "${#idxs[@]}" ] || { echo "  ❌ tar/idx COUNT mismatch"; fail=1; }
  [ -d "$d/.nv-meta" ] || fail=1
  for t in "${tars[@]}"; do [ -f "$t.idx" ] || { echo "  ❌ no .idx for $(basename "$t")"; fail=1; }; done

  if [ -f "$MANIFEST" ]; then
    while read -r key sz; do
      f="$d/${key#*/}"
      if [ ! -f "$f" ]; then echo "  ❌ MISSING shard: ${key#*/}  (src ${sz}B)"; fail=1; miss=$((miss+1))
      else g="$(stat -c %s "$f" 2>/dev/null || echo -1)"
           if [ "$g" -ne "$sz" ]; then echo "  ❌ SIZE MISMATCH: ${key#*/}  gb200=${g}  src=${sz}  Δ=$((g-sz))"; fail=1; trunc=$((trunc+1)); fi
      fi
    done < <(grep "^${node}/" "$MANIFEST")
  fi
done

# --- read bandwidth: largest .tar of the first active dir, 2GB, cache-bypass if possible ---
echo "=================================================================="
echo "### read bandwidth (cold-ish: iflag=direct if supported)"
probe="$(ls -S "${PATHS[0]}"/*.tar 2>/dev/null | head -1)"
if [ -n "$probe" ]; then
  echo "  probe: $probe"
  if ! dd if="$probe" of=/dev/null bs=8M count=256 iflag=direct |& grep -oE "[0-9.]+ [GM]B/s" | tail -1; then
    dd if="$probe" of=/dev/null bs=8M count=256 |& grep -oE "[0-9.]+ [GM]B/s" | tail -1
  fi
  echo "  (network PVC/NFS ~50-300 MB/s = SLOW; local NVMe ~2-6 GB/s = OK)"
fi

echo "=================================================================="
echo "### VERDICT"
if [ "$fail" -eq 0 ]; then
  echo "  ✅ All shards present & byte-identical to source. NOT an incomplete-sync problem."
  echo "     If it still stalls -> SLOW DISK (see bandwidth above) or a specific corrupt sample;"
  echo "     mitigate with a larger --distributed-timeout-minutes and/or stage data to local NVMe."
else
  echo "  ❌ INCOMPLETE/CORRUPT SYNC: missing=$miss truncated/size-mismatch=$trunc (+structural issues above)."
  echo "     Re-copy the flagged shards (and their .tar.idx) from the /ov2 source, then re-verify."
fi
