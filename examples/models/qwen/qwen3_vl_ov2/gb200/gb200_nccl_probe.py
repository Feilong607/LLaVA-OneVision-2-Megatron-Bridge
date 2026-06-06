"""GB200 NCCL collective probe: all_reduce + all_to_all (EP-critical) busbw.
Launched by gb200_nccl_test.sh via torchrun. Reports per-collective bus bandwidth."""
import os, time, torch
import torch.distributed as dist


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if rank == 0:
        print(f"world={world}  torch={torch.__version__}  dev0={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}  NVLS={os.environ.get('NCCL_NVLS_ENABLE','?')}")
        print(f"{'collective':<14}{'size':>9}{'time(ms)':>11}{'busbw(GB/s)':>14}")

    for mb in (32, 128, 512, 1024):
        n = (mb * 1024 * 1024 // 4)
        x = torch.ones(n, device=dev, dtype=torch.float32)
        dt = bench(lambda: dist.all_reduce(x))
        bus = 2 * (world - 1) / world * (n * 4) / dt / 1e9
        if rank == 0:
            print(f"{'all_reduce':<14}{mb:>7}MB{dt*1e3:>11.2f}{bus:>14.1f}")

    for mb in (32, 128, 512, 1024):
        n = (mb * 1024 * 1024 // 4) // world * world
        snd = torch.ones(n, device=dev, dtype=torch.float32)
        rcv = torch.empty_like(snd)
        dt = bench(lambda: dist.all_to_all_single(rcv, snd))
        bus = (world - 1) / world * (n * 4) / dt / 1e9
        if rank == 0:
            print(f"{'all_to_all':<14}{mb:>7}MB{dt*1e3:>11.2f}{bus:>14.1f}")

    if rank == 0:
        print("done (all_to_all is the EP8 dispatch/combine pattern — watch its busbw across nodes)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
