"""GB200 achievable GEMM TFLOP/s across precisions -> MFU-peak calibration.

  python gb200_flops.py

Pick the precision you TRAIN in as MFU_PEAK_TFLOPS:
  Phase 1 (bf16)  -> MFU_PEAK_TFLOPS = <bf16 number>
  Phase 2 (MXFP8) -> MFU_PEAK_TFLOPS = <fp8_e4m3 number>   (MXFP8 ~= fp8 tensor-core peak)
GB200/B200 spec (dense, per GPU, no sparsity): bf16/fp16 ~2.25 PF, fp8 ~4.5 PF, fp4 ~9 PF.
"""
import time
import torch


def _bench(run, iters=50, warm=12):
    for _ in range(warm):
        run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def _tflops(n, dt):
    return 2.0 * n ** 3 / dt / 1e12


def main():
    assert torch.cuda.is_available(), "no CUDA"
    d = torch.device("cuda", 0)
    print(f"{torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    scale = torch.tensor(1.0, device=d)
    for n in (8192, 16384):
        print(f"\n== GEMM {n}x{n}x{n}  (achievable TFLOP/s/GPU) ==")
        a = torch.randn(n, n, device=d)
        b = torch.randn(n, n, device=d)
        rows = []

        torch.backends.cuda.matmul.allow_tf32 = False
        try: rows.append(("fp32", "%8.0f" % _tflops(n, _bench(lambda: a @ b))))
        except Exception as e: rows.append(("fp32", "n/a: " + str(e)[:48]))

        torch.backends.cuda.matmul.allow_tf32 = True
        try: rows.append(("tf32", "%8.0f" % _tflops(n, _bench(lambda: a @ b))))
        except Exception as e: rows.append(("tf32", "n/a: " + str(e)[:48]))

        for nm, dt_ in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            x, y = a.to(dt_), b.to(dt_)
            try: rows.append((nm, "%8.0f" % _tflops(n, _bench(lambda: x @ y))))
            except Exception as e: rows.append((nm, "n/a: " + str(e)[:48]))

        # fp8 e4m3 via torch._scaled_mm (mat2 must be column-major)
        try:
            af = a.to(torch.float8_e4m3fn)
            bf = torch.randn(n, n, device=d).to(torch.float8_e4m3fn).t()
            run = lambda: torch._scaled_mm(af, bf, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            run()
            rows.append(("fp8_e4m3", "%8.0f" % _tflops(n, _bench(run))))
        except Exception as e:
            rows.append(("fp8_e4m3", "n/a: " + str(e)[:60]))

        for nm, v in rows:
            print(f"  {nm:<10} {v}{' TFLOP/s' if v.strip().replace('.','',1).isdigit() else ''}")
    print("\nfp4/NVFP4 (~2x fp8) not benched here (needs TransformerEngine MXFP4 kernels).")
    print("Set MFU_PEAK_TFLOPS to the row matching your run precision (bf16=Phase1, fp8=Phase2/MXFP8).")


if __name__ == "__main__":
    main()
