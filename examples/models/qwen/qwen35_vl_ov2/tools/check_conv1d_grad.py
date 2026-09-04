#!/usr/bin/env python3
"""Does the image's fla causal_conv1d produce correct gradients on OUR GDN layout? (single GPU, ~1 min)

fla v0.5.1 fixed "causal_conv1d backward producing wrong gradients for non-contiguous dy" and the
image ships v0.5.0. mcore's GatedDeltaNet feeds the conv an x that is a transpose(0,1)+split VIEW
(non-contiguous), and consumes the output through split -> reshape -> contiguous -> l2norm, so dy
is most likely a contiguous cat — but "most likely" is not a basis for an 8-day run. Measure it:

  case A  our exact layout + our exact consumer     -> must PASS on any fla we train with
  case B  positive control: inject a NON-contiguous dy -> expected FAIL on 0.5.0, PASS on >=0.5.1
  case C  contiguous x, contiguous dy (sanity)      -> must PASS

Reference = torch F.conv1d per varlen segment in fp32 from the same bf16 inputs. Tolerance is
relative L2 5e-2 (bf16 kernel vs fp32 reference); a real bug shows O(1) discrepancies.

Run inside any pod of the training image (a busy GPU is fine, it needs <2 GB):
    python3 ~/bridge-export/examples/models/qwen/qwen35_vl_ov2/tools/check_conv1d_grad.py
"""
import sys

import torch
import torch.nn.functional as F


def ref_conv(x, w, b, cu, act="silu"):
    """Causal depthwise conv per varlen segment, fp32. x [1,T,D], w [D,W], b [D] -> [1,T,D]."""
    W = w.shape[-1]
    outs = []
    for a, e in zip(cu[:-1], cu[1:]):
        xs = x[:, a:e, :].transpose(1, 2)  # [1, D, L]
        ys = F.conv1d(xs, w.unsqueeze(1), b, padding=W - 1, groups=w.shape[0])[..., : e - a]
        outs.append(ys.transpose(1, 2))
    y = torch.cat(outs, dim=1)
    return F.silu(y) if act == "silu" else y


class _InjectGrad(torch.autograd.Function):
    """loss = sum(y) but hand back an ARBITRARY (possibly non-contiguous) dy to y's producer."""

    @staticmethod
    def forward(ctx, y, g):
        ctx.save_for_backward(g)
        return y.float().sum()

    @staticmethod
    def backward(ctx, gout):
        (g,) = ctx.saved_tensors
        return g, None  # returned as-is: strides preserved


def consumer(y, qk_dim, head_dim):
    """mcore's _prepare_qkv_for_gated_delta_rule shape: split -> reshape -> contiguous -> l2norm."""
    qk, v = torch.split(y, [qk_dim, y.shape[-1] - qk_dim], dim=-1)
    qk = F.normalize(qk.reshape(1, y.shape[1], -1, head_dim).contiguous().float(), dim=-1)
    v = v.reshape(1, y.shape[1], -1, head_dim).contiguous().float()
    return (qk * 0.7).sum() + (v * 1.3).sum()


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def main():
    import fla
    from fla.modules.convolution import causal_conv1d

    print(f"fla version={getattr(fla, '__version__', '?')} file={fla.__file__}")
    torch.manual_seed(0)
    dev = "cuda"
    T, D, EXTRA, W, HD = 10192, 8192, 2048 + 16 + 16, 4, 128   # qkv width, gate+beta+alpha tail, conv width
    QK = 4096
    cu = torch.tensor([0, 5631, T], device=dev, dtype=torch.int32)
    cu_l = cu.tolist()
    w = (torch.randn(D, W, device=dev) * 0.3).to(torch.bfloat16).requires_grad_()
    b = (torch.randn(D, device=dev) * 0.1).to(torch.bfloat16).requires_grad_()
    TOL = 5e-2
    fails = 0

    def run(tag, x_view, loss_fn, dy_inject=None):
        nonlocal fails
        # fla path (bf16, exactly the training layout)
        y = causal_conv1d(x=x_view, weight=w, bias=b, activation="silu", initial_state=None,
                          output_final_state=False, cu_seqlens=cu)
        y = y[0] if isinstance(y, tuple) else y
        loss = _InjectGrad.apply(y, dy_inject) if dy_inject is not None else loss_fn(y)
        dx_f, dw_f, db_f = torch.autograd.grad(loss, [x_view, w, b], allow_unused=False)
        # reference (fp32) from the same values
        xr = x_view.detach().float().contiguous().requires_grad_()
        wr = w.detach().float().requires_grad_()
        br = b.detach().float().requires_grad_()
        yr = ref_conv(xr, wr, br, cu_l)
        if dy_inject is not None:
            dx_r, dw_r, db_r = torch.autograd.grad(yr, [xr, wr, br], grad_outputs=dy_inject.float())
        else:
            dx_r, dw_r, db_r = torch.autograd.grad(loss_fn(yr), [xr, wr, br])
        e_y = rel(y, yr)
        e = {"y": e_y, "dx": rel(dx_f, dx_r), "dw": rel(dw_f, dw_r), "db": rel(db_f, db_r)}
        ok = all(v < TOL for v in e.values())
        fails += 0 if ok else 1
        print(f"  {tag:<44} " + " ".join(f"{k}={v:.2e}" for k, v in e.items()) + f"   x_contig={x_view.is_contiguous()}"
              + (f" dy_contig={dy_inject.is_contiguous()}" if dy_inject is not None else "") + ("   PASS" if ok else "   **FAIL**"))
        return ok

    # case A: mcore layout -- [s, b, H] -> transpose(0,1) -> split(...)[0] = non-contiguous x view; real consumer
    full = torch.randn(T, 1, D + EXTRA, device=dev, dtype=torch.bfloat16).requires_grad_()
    qkv_view = torch.split(full.transpose(0, 1), [D, EXTRA], dim=-1)[0]
    run("A: our x view + our split/l2norm consumer", qkv_view, lambda y: consumer(y, QK, HD))
    # case B: positive control -- hand the conv a NON-contiguous dy (the v0.5.1 bug's trigger)
    g_nc = torch.randn(1, D, T, device=dev, dtype=torch.bfloat16).transpose(1, 2)  # [1,T,D] view, strides (T*D, 1, T)
    run("B: our x view + NON-contiguous injected dy", qkv_view, None, dy_inject=g_nc)
    # case C: everything contiguous (sanity)
    xc = torch.randn(1, T, D, device=dev, dtype=torch.bfloat16).requires_grad_()
    run("C: contiguous x + contiguous dy", xc, None, dy_inject=torch.randn(1, T, D, device=dev, dtype=torch.bfloat16))
    print("RESULT:", "ALL PASS — this fla is safe on our layout" if fails == 0 else
          f"{fails} case(s) FAIL — if A failed the image's conv backward is WRONG for training; if only B failed, "
          "our path is safe today but rebuild with fla>=0.5.1 before anything changes the consumer")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
