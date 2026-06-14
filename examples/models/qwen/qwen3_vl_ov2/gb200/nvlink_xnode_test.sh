#!/usr/bin/env bash
# =============================================================================
# gb200/nvlink_xnode_test.sh   —   Cross-node NVLink (NVL72 MNNVL) test suite
#   AUTO topology detection (no manual IP). Run the SAME command on EACH node:
#       bash gb200/nvlink_xnode_test.sh
#
# WHAT IT DOES (in order):
#   1. Auto-detect master/worker pods (…-master-0 / …-worker-N) -> node_rank/nnodes/master_addr.
#   2. Pre-flight: NVLink link status, NVLink topology matrix, MNNVL fabric state, driver/CUDA.
#   3. Torch probe (always available, torchrun like your training): sendrecv (pure cross-node ring)
#      + all_reduce + alltoall + all_gather + reduce_scatter, reporting algbw AND busbw with the
#      SAME factors nccl-tests uses, over a message-size sweep.
#   4. nvbandwidth (NVIDIA's official BW tool) multinode pass — if the binary is present.
#   5. nccl-tests pointer (your gb200_nccl_test.sh) — if you want the canonical binaries.
#   6. Live dcgmi NVLINK TX/RX monitor + verdict vs GB200 NVLink5 theory (~900 GB/s/GPU/dir).
#
# REFERENCES (how others test):
#   NVIDIA GB200 NVL Multi-Node Tuning Guide / nccl-tests PERFORMANCE.md / NVIDIA/nvbandwidth.
#
# KNOBS: NPROC(auto) MASTER_PORT(29555) SIZES("256M 512M 1G 2G 4G") ITERS(50)
#        COLLS("sendrecv alltoall all_reduce all_gather reduce_scatter") DCGM(1) NVBW(1)
#        LIST_IP="ip0 ip1" (manual override of auto-detect)
# =============================================================================
set -uo pipefail

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29555}"
SIZES="${SIZES:-256M 512M 1G 2G 4G}"
ITERS="${ITERS:-50}"
COLLS="${COLLS:-sendrecv alltoall all_reduce all_gather reduce_scatter}"
DCGM="${DCGM:-1}"; NVBW="${NVBW:-1}"
PYFILE="${PYFILE:-/tmp/xnode_nvlink_$$.py}"

resolve_ip() {
  local h="$1" ip=""
  ip="$(getent hosts "$h" 2>/dev/null | awk '{print $1; exit}')"
  [[ -z "$ip" ]] && ip="$(python3 -c "import socket;print(socket.gethostbyname('$h'))" 2>/dev/null || true)"
  [[ -n "$ip" ]] && echo "$ip"
}
size_to_bytes() {
  local s="${1^^}" m=1
  case "$s" in *G) m=$((1024**3)); s="${s%G}";; *M) m=$((1024**2)); s="${s%M}";;
               *K) m=1024; s="${s%K}";; esac
  echo $(( ${s%.*} * m ))
}

# ---------- AUTO topology detection (no manual IP) ----------
MY_HOST="$(hostname -s 2>/dev/null || hostname)"
NODE_RANK=""; NNODES=""; MASTER_ADDR=""
if [[ -n "${LIST_IP:-}" ]]; then
  read -ra _ips <<< "$LIST_IP"; NNODES="${#_ips[@]}"; MASTER_ADDR="${_ips[0]}"; NODE_RANK=-1
  for i in "${!_ips[@]}"; do for c in $(hostname -I); do [[ "$c" == "${_ips[$i]}" ]] && NODE_RANK=$i; done; done
  [[ "$NODE_RANK" -lt 0 ]] && { echo "FATAL: my IPs ($(hostname -I)) not in LIST_IP=$LIST_IP" >&2; exit 1; }
elif [[ -n "${MASTER_ADDR:-}" && ( -n "${NNODES:-}" || -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${NNODES:-$(( WORLD_SIZE / NPROC ))}"; NODE_RANK="${NODE_RANK:-${GROUP_RANK:-0}}"
elif [[ "$MY_HOST" =~ ^(.+)-(master|worker)-([0-9]+)$ ]]; then
  _pre="${BASH_REMATCH[1]}"; _role="${BASH_REMATCH[2]}"; _idx="${BASH_REMATCH[3]}"
  MASTER_ADDR="$(resolve_ip "${_pre}-master-0")"
  [[ -z "$MASTER_ADDR" ]] && { echo "FATAL: cannot resolve ${_pre}-master-0 (no DNS/hosts). Use LIST_IP=..." >&2; exit 1; }
  if [[ "$_role" == master ]]; then NODE_RANK=0; else NODE_RANK=$((_idx+1)); fi
  _n=0; while resolve_ip "${_pre}-worker-${_n}" >/dev/null; do _n=$((_n+1)); done; NNODES=$((_n+1))
else
  echo "FATAL: cannot auto-detect topology from hostname '$MY_HOST'. Set LIST_IP=\"ip0 ip1\"." >&2; exit 1
fi
WORLD=$(( NPROC * NNODES ))
(( NNODES >= 2 )) || { echo "FATAL: cross-node test needs >=2 nodes (got NNODES=$NNODES)." >&2; exit 1; }
SIZES_BYTES=""; for s in $SIZES; do SIZES_BYTES="$SIZES_BYTES $(size_to_bytes "$s")"; done; SIZES_BYTES="${SIZES_BYTES# }"

echo "============================================================================="
echo "[xnode-nvlink] host=$MY_HOST node_rank=$NODE_RANK nnodes=$NNODES nproc/node=$NPROC world=$WORLD"
echo "[xnode-nvlink] master=$MASTER_ADDR:$MASTER_PORT sizes=[$SIZES] colls=[$COLLS] iters=$ITERS"
echo "============================================================================="

# ---------- (2) pre-flight diagnostics (this node) ----------
if command -v nvidia-smi >/dev/null; then
  echo "--- GPU / driver ---"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | head -1 | sed 's/^/    /'
  echo "--- NVLink link status (want all 'Active') ---"
  nvidia-smi nvlink --status 2>/dev/null | grep -iE "GPU 0:|Link 0|Link 1" | head -6 | sed 's/^/    /' || echo "    (unavailable)"
  echo "--- MNNVL fabric state (GB200; ClusterUUID+CliqueId present = in a multi-node NVLink domain) ---"
  nvidia-smi -q 2>/dev/null | grep -iE "Fabric|ClusterUUID|CliqueId|^[[:space:]]*State" | head -8 | sed 's/^/    /' || echo "    (no Fabric section -> not an MNNVL fabric / older driver)"
  echo "--- NVLink topology matrix (NV# = # NVLinks between GPUs) ---"
  nvidia-smi topo -m 2>/dev/null | head -12 | sed 's/^/    /' || true
fi

# ---------- the comprehensive torch probe ----------
cat > "$PYFILE" <<'PYEOF'
import os, time, torch, torch.distributed as dist
def human(n):
    n=float(n)
    for u in ("B","KB","MB","GB"):
        if n<1024: return f"{n:.0f}{u}"
        n/=1024
    return f"{n:.1f}TB"
dist.init_process_group("nccl")
rank,world=dist.get_rank(),dist.get_world_size()
nproc=int(os.environ.get("NPROC_PER_NODE",torch.cuda.device_count()))
local=int(os.environ.get("LOCAL_RANK",rank%torch.cuda.device_count()))
torch.cuda.set_device(local); dev=torch.device("cuda",local)
nnodes=world//nproc
iters=int(os.environ.get("ITERS","50")); warm=int(os.environ.get("WARMUP","5"))
sizes=[int(x) for x in os.environ.get("SIZES_BYTES","1073741824").split()]
colls=os.environ.get("COLLS","sendrecv alltoall all_reduce").split()
# cross-node ring: same local-rank on next/prev node -> every byte crosses the node boundary
send_to=(rank+nproc)%world; recv_from=(rank-nproc+world)%world
def factor(c):                                  # nccl-tests busbw factors
    if c=="all_reduce": return 2*(world-1)/world
    if c in ("alltoall","all_gather","reduce_scatter"): return (world-1)/world
    return 1.0
def timed(fn):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); dist.barrier(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters
def run(c,nbytes):
    n=max(world,nbytes//2)
    if c=="sendrecv":
        s=torch.ones(n,device=dev,dtype=torch.bfloat16); r=torch.empty_like(s)
        fn=lambda:[q.wait() for q in dist.batch_isend_irecv([dist.P2POp(dist.isend,s,send_to),dist.P2POp(dist.irecv,r,recv_from)])]
        moved=s.numel()*2
    elif c=="all_reduce":
        x=torch.ones(n,device=dev,dtype=torch.bfloat16); fn=lambda:dist.all_reduce(x); moved=x.numel()*2
    elif c=="alltoall":
        m=(n//world)*world; s=torch.ones(m,device=dev,dtype=torch.bfloat16); r=torch.empty_like(s)
        fn=lambda:dist.all_to_all_single(r,s); moved=s.numel()*2
    elif c=="all_gather":
        per=max(1,n//world); i=torch.ones(per,device=dev,dtype=torch.bfloat16); o=torch.empty(per*world,device=dev,dtype=torch.bfloat16)
        fn=lambda:dist.all_gather_into_tensor(o,i); moved=o.numel()*2
    elif c=="reduce_scatter":
        per=max(1,n//world); i=torch.ones(per*world,device=dev,dtype=torch.bfloat16); o=torch.empty(per,device=dev,dtype=torch.bfloat16)
        fn=lambda:dist.reduce_scatter_tensor(o,i); moved=i.numel()*2
    else: return None
    dt=timed(fn); alg=moved/dt; return dt,moved,alg,alg*factor(c)
if rank==0:
    print(f"# torch probe | world={world} nnodes={nnodes} nproc/node={nproc} iters={iters} bf16")
    print(f"# sendrecv = CROSS-NODE ring (pure inter-node link). others = full-world (nccl-tests style).")
    print(f"# busbw factors: all_reduce x2(n-1)/n; alltoall/all_gather/reduce_scatter x(n-1)/n; sendrecv x1.")
    print(f"# GB200 NVL72 NVLink5 ~900 GB/s/GPU/dir (1.8TB/s bidir).")
    print(f"# {'collective':<15}{'msg':>7}{'ms/it':>9}{'algbw':>9}{'busbw':>9}  (GB/s)")
for c in colls:
    for nb in sizes:
        res=run(c,nb)
        if res and rank==0:
            dt,mv,alg,bus=res
            print(f"  {c:<15}{human(nb):>7}{dt*1e3:>9.2f}{alg/1e9:>9.0f}{bus/1e9:>9.0f}")
if rank==0:
    print(f"# VERDICT: sendrecv busbw >300 GB/s/rank => inter-node NVLink5 OK; <100 => fell back to IB.")
dist.destroy_process_group()
PYEOF

# ---------- (6a) live dcgmi NVLINK monitor ----------
DMON_PID=""; DMON_LOG="/tmp/xnode_dcgm_${MY_HOST}_$$.log"
if [[ "$DCGM" == "1" ]] && command -v dcgmi >/dev/null; then
  ( dcgmi dmon -e 1011,1012 -d 1000 > "$DMON_LOG" 2>&1 ) & DMON_PID=$!
  echo "[xnode-nvlink] dcgmi dmon NVLINK TX(1011)/RX(1012) -> $DMON_LOG"
fi

# ---------- (3) run the torch probe ----------
echo "=== [3] torch collective probe (sendrecv cross-node ring + nccl-tests-style busbw) ==="
RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
NPROC_PER_NODE="$NPROC" COLLS="$COLLS" ITERS="$ITERS" SIZES_BYTES="$SIZES_BYTES" \
  torchrun $RDZV --nproc_per_node="$NPROC" "$PYFILE"; RC=$?
rm -f "$PYFILE"

# ---------- (4) nvbandwidth multinode (NVIDIA official tool, if present) ----------
NVBW_BIN="$(command -v nvbandwidth || echo "${NVBANDWIDTH:-}")"
if [[ "$NVBW" == "1" && -n "$NVBW_BIN" && -x "$NVBW_BIN" ]] && command -v mpirun >/dev/null; then
  echo "=== [4] nvbandwidth -p multinode (official direct-BW tool) ==="
  mpirun --allow-run-as-root --map-by "ppr:${NPROC}:node" --bind-to core -np "$WORLD" \
    -H "$MASTER_ADDR:$NPROC" "$NVBW_BIN" -p multinode 2>&1 | sed 's/^/    /' || \
    echo "    (nvbandwidth multinode run failed — needs the -DMULTINODE=1 build + an MPI hostfile across both pods)"
elif [[ "$NVBW" == "1" && "$NODE_RANK" -eq 0 ]]; then
  echo "=== [4] nvbandwidth NOT found — to add the official direct NVLink BW tool ==="
  echo "    git clone https://github.com/NVIDIA/nvbandwidth && cd nvbandwidth && cmake -DMULTINODE=1 . && make"
  echo "    then: mpirun --allow-run-as-root --map-by ppr:${NPROC}:node -np ${WORLD} ./nvbandwidth -p multinode"
fi

# ---------- (5) nccl-tests pointer ----------
if [[ "$NODE_RANK" -eq 0 ]]; then
  echo "=== [5] canonical nccl-tests (busbw) — for cross-check ==="
  echo "    your gb200_nccl_test.sh already runs these; the cross-node ones:"
  echo "      sendrecv_perf -b 512M -e 8G -f 2 -g ${NPROC}     # purest link BW (busbw=algbw)"
  echo "      alltoall_perf -b 512M -e 8G -f 2 -g ${NPROC}     # = your EP8 pattern"
  echo "      all_reduce_perf -b 512M -e 8G -f 2 -g ${NPROC}   # headline busbw x2(n-1)/n"
fi

# ---------- teardown ----------
[[ -n "$DMON_PID" ]] && { kill "$DMON_PID" 2>/dev/null; echo "[xnode-nvlink] dcgmi log: $DMON_LOG (TX/RX high during run = NVLink carried it)"; }
[[ "$NODE_RANK" -eq 0 ]] && echo "[xnode-nvlink] done (rc=$RC). 把 [3] 里 sendrecv/alltoall 的 busbw GB/s 贴给我，我判 NVLink5 吃满没。"
exit $RC
