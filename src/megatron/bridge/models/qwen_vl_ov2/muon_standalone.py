# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained Muon optimizer (non-distributed) — faithful copy of AIAK's
``aiak_megatron/megatron/core/optimizer/muon.py`` so we can use Muon without the
``emerging_optimizers`` package. Matrix (2D, use_muon=True) params get Newton-Schulz
orthogonalized updates scaled by sqrt(max(d_out,d_in))*matched_adamw_rms; scalar
(use_muon=False) params get internal AdamW.

For data-parallel use: average each param's .grad across DP ranks BEFORE step()
(identical synced params + identical grads => identical Muon update on every rank).
"""
import math
import torch


def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    return lr * adjusted_ratio


class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups, lr=2e-2, weight_decay=0.1, matched_adamw_rms=0.2,
                 momentum=0.95, nesterov=True, ns_steps=5, adamw_betas=(0.95, 0.95), adamw_eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        ns_inputs = {}
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert g.dim() == 2, f"muon param must be 2D, got {tuple(g.shape)}"
                state = self.state[p]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_inputs[p] = g.bfloat16()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            lr = group["lr"]; ns_steps = group["ns_steps"]; wd = group["weight_decay"]
            rms = group["matched_adamw_rms"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                ns_input = ns_inputs[p]
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)
                p.data.mul_(1 - lr * wd)
                adj = adjust_lr_wd_for_muon(lr, rms, ns_input.shape)
                p.data.add_(update.to(p.dtype), alpha=-adj)

        for group in self.param_groups:
            if group.get("use_muon", False):
                continue
            group["step"] = group.get("step", 0) + 1
            step = group["step"]; lr = group["lr"]; wd = group["weight_decay"]
            b1, b2 = group["adamw_betas"]; eps = group["adamw_eps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)
                buf1 = state["adamw_exp_avg"]; buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - b1)
                buf2.lerp_(g.square(), 1 - b2)
                gg = buf1 / (eps + buf2.sqrt())
                bc1 = 1 - b1 ** step; bc2 = 1 - b2 ** step
                scale = bc1 / bc2 ** 0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(gg, alpha=-lr / scale)
