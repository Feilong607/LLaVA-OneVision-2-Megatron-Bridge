#!/usr/bin/env bash
# =============================================================================
# GB200 data-path / MFU probe — read-only, no GPU required except `live`.
#
# Answers two questions about the stage-3 line:
#   1. Is the container RLIMIT_NOFILE still 1024/1024, or did the platform
#      raise it? (`rlimit`)
#   2. Is the low-and-noisy MFU explained by the data path? Three independent
#      angles: what the training's own timers already recorded (`logscan`),
#      raw WekaFS shard-read throughput/latency per dataset (`io`), and — on a
#      pod that is running the training — whether GPU-util troughs coincide
#      with starved dataloader workers (`live`).
#
# Usage (run from any pod that mounts /home and /datasets; point the workload
# Args directly at this file, no wrapper):
#   bash .../probe_data_mfu.sh              # = rlimit + logscan + io
#   bash .../probe_data_mfu.sh rlimit       # one stage
#   bash .../probe_data_mfu.sh live 180     # sample a RUNNING training pod 180s
#   bash .../probe_data_mfu.sh loader       # optional: energon iteration rate
#                                           # (training image; obeys
#                                           #  OV2_PARALLEL_SHARD_ITERS/WORKERS)
# Env: DATA_PATH (blend yaml; default: stage3_mix_img10.yaml next to this
#      script), TRAIN_LOG (logscan target; default newest ~/train_logs/stage3*),
#      PROBE_MB (io read size per dataset, default 64), PROBE_SAMPLES (loader,
#      default 128), WORKERS (loader, default 2).
#
# RLIMIT trap: the limit is inherited at container start. A pod created BEFORE
# a platform-side change still shows the old value — verify in a FRESH pod.
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
LIVE_SECS="${2:-${PROBE_SECS:-120}}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-$_SCRIPT_DIR/stage3_mix_img10.yaml}"
PROBE_MB="${PROBE_MB:-64}"
PROBE_SAMPLES="${PROBE_SAMPLES:-128}"
WORKERS="${WORKERS:-2}"

# Persist from the first line: pods (and their LOGS tab) do not survive gang
# teardown; ~/train_logs on WekaFS does.
mkdir -p "$HOME/train_logs"
LOG="$HOME/train_logs/probe_data_mfu_$(hostname)_$(date +%s).log"
exec > >(tee -a "$LOG") 2>&1
echo "[probe] host=$(hostname) date=$(date +%Y-%m-%dT%H:%M:%S%z) stage=$STAGE log=$LOG"

# ── rlimit ───────────────────────────────────────────────────────────────────
probe_rlimit() {
  echo "[probe:rlimit] soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
  [[ -r /proc/self/limits ]] && grep "Max open files" /proc/self/limits | sed 's/^/[probe:rlimit] /'
  local _soft _hard
  _soft="$(ulimit -Sn)"; _hard="$(ulimit -Hn)"
  if [[ "$_hard" == "unlimited" ]] || (( _hard >= 16384 )); then
    echo "[probe:rlimit] VERDICT: RAISED (hard=$_hard) — psi=16 fits with fd-based IPC; the psi16 A/B can run without OV2_MP_SHARING_STRATEGY"
  else
    echo "[probe:rlimit] VERDICT: still capped (hard=$_hard) — the 08-25 constraint stands; psi=16 needs file_system IPC and stays margin-thin"
  fi
  echo "[probe:rlimit] NOTE: value is inherited at container start — a pod created before any platform change shows the old limit; trust only a fresh pod."
}

# ── logscan: what the training's own timers already say ─────────────────────
# timing_log_level=2 + minmax + log_throughput are on in the stage-3 launcher,
# so every iteration line carries elapsed/TFLOPs and every timer dump carries
# `batch-generator: (min, max)` across ranks. batch-generator is the time the
# step spent WAITING on the dataloader — the single most direct "is data the
# blocker" number, and its max-across-ranks is what stalls the whole DP group.
probe_logscan() {
  local log="${TRAIN_LOG:-}"
  if [[ -z "$log" ]]; then
    # Iteration/timer lines print via print_rank_last, so only the LAST rank's
    # pod log carries them — the newest file is usually master's, which has
    # none (bitten 08-27). Pick the newest log that actually has them.
    local _cand
    for _cand in $(ls -t "$HOME"/train_logs/stage3*.log 2>/dev/null); do
      if grep -q "elapsed time per iteration" "$_cand"; then log="$_cand"; break; fi
    done
  fi
  if [[ -z "$log" || ! -r "$log" ]]; then
    echo "[probe:logscan] SKIP: no training log found (set TRAIN_LOG=...)"
    return 0
  fi
  echo "[probe:logscan] parsing $log"
  python3 - "$log" <<'PYEOF'
import re
import statistics as st
import sys

text = open(sys.argv[1], errors="replace").read()
iters = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", text)]
tflops = [float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\): ([\d.]+)", text)]
# minmax format: `name ....: (min, max)`; plain format: `name ....: 123.4`
def timer(name):
    pairs = re.findall(rf"{name} \.*:? *\(([\d.]+), ([\d.]+)\)", text)
    if pairs:
        return [float(b) for _, b in pairs]  # max across ranks
    return [float(x) for x in re.findall(rf"{name} \.*:? *([\d.]+)", text)]

def dist(name, xs):
    if not xs:
        print(f"[probe:logscan] {name}: none found")
        return
    xs_s = sorted(xs)
    p = lambda q: xs_s[min(len(xs_s) - 1, int(q * len(xs_s)))]
    cv = st.pstdev(xs) / st.mean(xs) if st.mean(xs) else 0.0
    print(f"[probe:logscan] {name}: n={len(xs)} p50={p(0.5):.1f} p90={p(0.9):.1f} "
          f"max={xs_s[-1]:.1f} mean={st.mean(xs):.1f} CV={cv:.3f}")

dist("iter_ms", iters)
dist("tflops_per_gpu", tflops)
bg = timer("batch-generator")
fb = timer("forward-backward")
dist("batch_generator_maxrank_ms", bg)
dist("forward_backward_ms", fb)
n = min(len(bg), len(iters))
if n > 10:
    share = [bg[i] / iters[i] for i in range(n) if iters[i] > 0]
    hi = sum(1 for s in share if s > 0.10)
    print(f"[probe:logscan] batch-generator share of iter: mean={st.mean(share):.1%} "
          f"iters>10%: {hi}/{len(share)} ({hi/len(share):.1%})")
    # crude Pearson between iter time and bg time: data-blocked runs correlate strongly
    mi, mb = st.mean(iters[:n]), st.mean(bg[:n])
    cov = sum((iters[i] - mi) * (bg[i] - mb) for i in range(n))
    di = sum((iters[i] - mi) ** 2 for i in range(n)) ** 0.5
    db = sum((bg[i] - mb) ** 2 for i in range(n)) ** 0.5
    if di and db:
        print(f"[probe:logscan] corr(iter_ms, batch_generator_ms) = {cov/(di*db):.3f} "
              f"(near 1.0 => iteration-time noise IS dataloader noise)")
    print("[probe:logscan] READ: bg share >10% on many iters, or corr near 1 -> data path is the MFU blocker;")
    print("[probe:logscan]       bg tiny but iter_ms CV high -> look at compute-side variance (bin token spread, EP imbalance) instead.")
PYEOF
}

# ── io: raw WekaFS shard read throughput/latency, per dataset ────────────────
# Reads PROBE_MB from one tar shard of every dataset in the blend, at a random
# offset (fresh pages; posix_fadvise(DONTNEED) afterwards where available), and
# times open()+first-64KB separately. Slow or high-variance rows here are the
# storage-side signature of dataloader stalls; uniform fast rows acquit WekaFS.
probe_io() {
  if [[ ! -r "$DATA_PATH" ]]; then
    echo "[probe:io] SKIP: blend yaml not readable: $DATA_PATH"
    return 0
  fi
  echo "[probe:io] blend=$DATA_PATH read=${PROBE_MB}MB/dataset"
  python3 - "$DATA_PATH" "$PROBE_MB" <<'PYEOF'
import glob
import os
import random
import re
import sys
import time

yaml_path, probe_mb = sys.argv[1], int(sys.argv[2])
paths = re.findall(r"^\s*path:\s*(\S+)", open(yaml_path).read(), re.M)
print(f"[probe:io] datasets in blend: {len(paths)}")
rows, chunk = [], 4 * 1024 * 1024
for p in paths:
    tars = sorted(glob.glob(os.path.join(p, "*.tar")))
    name = "/".join(p.rstrip("/").split("/")[-2:])
    if not tars:
        print(f"[probe:io] {name}: NO *.tar FOUND")
        continue
    tar = random.choice(tars)
    size = os.path.getsize(tar)
    want = min(probe_mb * 1024 * 1024, size)
    off = random.randrange(0, max(1, size - want))
    t0 = time.monotonic()
    fd = os.open(tar, os.O_RDONLY)
    os.lseek(fd, off, os.SEEK_SET)
    os.read(fd, 65536)
    t_first = (time.monotonic() - t0) * 1000
    got, t1 = 65536, time.monotonic()
    while got < want:
        b = os.read(fd, min(chunk, want - got))
        if not b:
            break
        got += len(b)
    dt = time.monotonic() - t1
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    mbs = (got / 1048576) / dt if dt > 0 else float("inf")
    rows.append((mbs, t_first, name))
    print(f"[probe:io] {name}: open+64KB={t_first:.1f}ms seq={mbs:.0f}MB/s (shard={os.path.basename(tar)})")
rows.sort()
if rows:
    n = len(rows)
    print(f"[probe:io] SUMMARY seq MB/s: min={rows[0][0]:.0f} p50={rows[n//2][0]:.0f} max={rows[-1][0]:.0f}")
    print(f"[probe:io] slowest 3: " + "; ".join(f"{r[2]}={r[0]:.0f}MB/s" for r in rows[:3]))
    firsts = sorted(r[1] for r in rows)
    print(f"[probe:io] open+64KB ms: min={firsts[0]:.1f} p50={firsts[n//2]:.1f} max={firsts[-1]:.1f}")
    print("[probe:io] READ: p50 well above ~200MB/s and open p50 <50ms acquits sequential IO;")
    print("[probe:io]       a slow long-tail row here matches 'MFU dips when a rare dataset gets sampled'.")
PYEOF
}

# ── live: correlate GPU-util troughs with dataloader-worker state ────────────
# Run INSIDE a pod of the running training (TERMINAL tab). Read-only: samples
# nvidia-smi + /proc every 2s. Dataloader workers = child PIDs of the
# run_recipe.py ranks. D-state children = blocked in IO.
probe_live() {
  command -v nvidia-smi >/dev/null || { echo "[probe:live] SKIP: no nvidia-smi"; return 0; }
  pgrep -f run_recipe.py >/dev/null || { echo "[probe:live] SKIP: no run_recipe.py process on this pod"; return 0; }
  echo "[probe:live] sampling ${LIVE_SECS}s @2s: gpu_util | workers(D-state) | worker read MB/s"
  python3 - "$LIVE_SECS" <<'PYEOF'
import glob
import os
import statistics as st
import subprocess
import sys
import time

secs = int(sys.argv[1])

def rank_pids():
    out = subprocess.run(["pgrep", "-f", "run_recipe.py"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]

def children(pids):
    kids = []
    for p in pids:
        for f in glob.glob(f"/proc/{p}/task/*/children"):
            try:
                kids += [int(x) for x in open(f).read().split()]
            except OSError:
                pass
    return kids

def state(pid):
    try:
        return open(f"/proc/{pid}/stat").read().split(") ")[1].split()[0]
    except OSError:
        return "?"

def read_bytes(pids):
    tot = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/io"):
                if line.startswith("read_bytes"):
                    tot += int(line.split()[1])
        except OSError:
            pass
    return tot

def gpu_util():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.strip().isdigit()]

samples, prev_rb, prev_t = [], None, None
t_end = time.monotonic() + secs
while time.monotonic() < t_end:
    ranks = rank_pids()
    kids = children(ranks)
    dstate = sum(1 for k in kids if state(k) == "D")
    rb = read_bytes(kids)
    now = time.monotonic()
    rate = (rb - prev_rb) / (now - prev_t) / 1048576 if prev_rb is not None else 0.0
    prev_rb, prev_t = rb, now
    utils = gpu_util()
    mu = st.mean(utils) if utils else 0.0
    samples.append((mu, dstate, rate, len(kids)))
    print(f"[probe:live] gpu={mu:5.1f}% workers={len(kids)} D={dstate} read={rate:8.1f}MB/s")
    time.sleep(2)

if len(samples) > 5:
    lows = [s for s in samples[1:] if s[0] < 20.0]
    print(f"[probe:live] SUMMARY: {len(lows)}/{len(samples)-1} samples with mean GPU util <20%")
    if lows:
        d_in_low = st.mean([s[1] for s in lows])
        d_all = st.mean([s[1] for s in samples[1:]])
        print(f"[probe:live] D-state workers: {d_in_low:.1f} during troughs vs {d_all:.1f} overall")
        print("[probe:live] READ: troughs with elevated D-state => workers blocked on storage;")
        print("[probe:live]       troughs with D~0 and read~0 => workers CPU-bound (decode/pack) or too few (num_workers).")
PYEOF
}

# ── loader (optional): energon iteration rate on the real blend ──────────────
# Needs the training image (megatron.energon + aiak_shim sample types). GPU not
# required. Honors OV2_PARALLEL_SHARD_ITERS (default 1 = production) so psi=1
# vs psi=16 dataloader throughput can be A/B-ed once RLIMIT allows.
probe_loader() {
  local repo
  repo="$(cd "$_SCRIPT_DIR/../../../../.." && pwd)"
  export PYTHONPATH="$repo/src:$repo/3rdparty/Megatron-LM:$repo/aiak_shim:${PYTHONPATH:-}"
  echo "[probe:loader] blend=$DATA_PATH samples=$PROBE_SAMPLES workers=$WORKERS psi=${OV2_PARALLEL_SHARD_ITERS:-1}"
  python3 - "$DATA_PATH" "$PROBE_SAMPLES" "$WORKERS" <<'PYEOF'
import os
import sys
import time
import traceback

yaml_path, n_want, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
psi = int(os.environ.get("OV2_PARALLEL_SHARD_ITERS", "1"))
try:
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    wc = WorkerConfig(rank=0, world_size=1, num_workers=workers)
    ds = get_train_dataset(
        yaml_path, batch_size=1, shuffle_buffer_size=100, max_samples_per_sequence=None,
        worker_config=wc, parallel_shard_iters=psi, image_decode="pil",
    )
    loader = get_loader(ds)
    gaps, t_prev, t0 = [], None, time.monotonic()
    for i, _ in enumerate(loader):
        now = time.monotonic()
        if t_prev is not None:
            gaps.append((now - t_prev) * 1000)
        t_prev = now
        if i + 1 >= n_want:
            break
    dt = time.monotonic() - t0
    gaps.sort()
    p = lambda q: gaps[min(len(gaps) - 1, int(q * len(gaps)))] if gaps else 0.0
    print(f"[probe:loader] {n_want} samples in {dt:.1f}s = {n_want/dt:.2f} samples/s "
          f"(gap ms p50={p(0.5):.0f} p90={p(0.9):.0f} max={gaps[-1] if gaps else 0:.0f})")
    print("[probe:loader] READ: compare samples/s against the training's consumption need "
          "(GBS/DP per rank per iter / iter seconds); gap max >> p50 = stall long-tail.")
except Exception:
    print("[probe:loader] FAILED (this stage needs the training image; API drift is possible):")
    traceback.print_exc()
PYEOF
}

case "$STAGE" in
  rlimit)  probe_rlimit ;;
  logscan) probe_logscan ;;
  io)      probe_io ;;
  live)    probe_live ;;
  loader)  probe_loader ;;
  all)     probe_rlimit; probe_io; probe_logscan ;;
  *) echo "[probe] FATAL: unknown stage '$STAGE' (rlimit|logscan|io|live|loader|all)" >&2; exit 1 ;;
esac
echo "[probe] done — persisted at $LOG"
