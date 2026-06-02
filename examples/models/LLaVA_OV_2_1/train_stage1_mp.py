"""OV2.1-4B stage-1 (adapter-only) data-parallel training on the 558k WDS.

Loads the unified new-encoder mcore ckpt (LLM + p16m3 vision), random-inits the adapter,
freezes LLM+vision, trains the adapter with Muon (data-parallel grad-averaging).

Features: Megatron-style metric log, TensorBoard, resume from latest iter_NNNN,
and FULL-MODEL checkpoints (language_model.*+vision_model.*+adapter.*) in mcore layout
so stage-2 loads everything in one shot.
"""
import os, sys, time, glob, shutil, math, torch
import torch.distributed as dist
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# optimizer = torch.optim.AdamW (matches AIAK adapter-only --optimizer adam); Muon removed per audit

OVCK = os.environ.get("OVCK", "/ov2/feilong/ov2_quickstart/ov_encoder_p16m3_qwen3_mcore_tp1pp1")
PROC = "/ov2/feilong/preprocessor_p16m33"
DATA = os.environ.get("DATA", "/vlm/data/blip_laion_cc_sbu_558k_wds")
OUT = os.environ.get("OUT", "/ov2/feilong/Megatron-Bridge/ov2_1_4b/stage1_adapter_558k")
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "1"))
KEEP_LAST = int(os.environ.get("KEEP_LAST", "5"))
LR = float(os.environ.get("LR", "2e-5"))
MIN_LR = float(os.environ.get("MIN_LR", "1e-6"))             # AIAK --min-lr (cosine floor)
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.002"))  # AIAK --lr-warmup-fraction
GBS_TARGET = int(os.environ.get("GBS_TARGET", "256"))        # AIAK --global-batch-size
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", "558128"))
CLIP_GRAD = float(os.environ.get("CLIP_GRAD", "1.0"))        # AIAK --clip-grad
IMAGE_TOKEN_ID = 151655
IGNORE = -100


def _to_pil(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if torch.is_tensor(im):
        a = im.detach().cpu()
        if a.dtype.is_floating_point:
            a = (a * 255).clamp(0, 255).to(torch.uint8)
        a = a.numpy()
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        return Image.fromarray(a).convert("RGB")
    return Image.fromarray(np.asarray(im)).convert("RGB")


def build_encoder():
    from transformers import AutoProcessor
    from megatron.energon import DefaultTaskEncoder, SkipSample
    proc = AutoProcessor.from_pretrained(PROC, trust_remote_code=True)

    class Enc(DefaultTaskEncoder):
        def encode_sample(self, s):
            try:
                msgs = s.messages or []
                user = next(m["content"] for m in msgs if m.get("role") == "user")
                ans = next(m["content"] for m in msgs if m.get("role") == "assistant")
                imgs = [_to_pil(x) for x in (s.image or [])]
                vis = "<|vision_start|><|image_pad|><|vision_end|>"
                user = user.replace("<image>", vis) if "<image>" in user else (vis + "\n" + user if imgs else user)
                sys_ = f"<|im_start|>system\n{s.system}<|im_end|>\n" if s.system else ""
                prompt = f"{sys_}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                ft = proc(text=[prompt + ans + "<|im_end|>"], images=imgs or None, return_tensors="pt")  # supervise answer+eos only (no trailing \n), matches AIAK
                pt = proc(text=[prompt], images=imgs or None, return_tensors="pt")
                ids = ft["input_ids"][0]
                if ids.shape[0] > 32000:
                    raise SkipSample()
                labels = ids.clone(); labels[: pt["input_ids"].shape[1]] = IGNORE
                return {"input_ids": ids, "labels": labels,
                        "pixel_values": ft.get("pixel_values"), "image_grid_thw": ft.get("image_grid_thw")}
            except SkipSample:
                raise
            except Exception:
                raise SkipSample()

        def batch(self, samples):
            return samples[0]

    return Enc()


@torch.no_grad()
def init_adapter(adapter, std=0.02):
    for n, p in adapter.named_parameters():
        if p.dim() >= 2: p.normal_(0.0, std)
        elif "layernorm" in n and "weight" in n: p.fill_(1.0)
        else: p.zero_()


def lr_at(step, total, warmup):
    # AIAK: linear warmup over warmup steps, then cosine decay LR -> MIN_LR over the rest
    if step < warmup:
        return LR * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def latest_iter(out):
    f = os.path.join(out, "latest_checkpointed_iteration.txt")
    if os.path.exists(f):
        try:
            return int(open(f).read().strip())
        except Exception:
            return 0
    return 0


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(1, 1)
    ACCUM = int(os.environ.get("ACCUM", str(max(1, round(GBS_TARGET / world)))))  # micro-steps/opt-step to reach ~GBS_TARGET
    gbs = world * ACCUM  # MBS=1 per rank, ACCUM grad-accum micro-steps
    STEPS = int(os.environ.get("STEPS", str((TOTAL_SAMPLES + gbs - 1) // gbs)))   # 1 epoch over TOTAL_SAMPLES
    warmup_steps = max(1, int(WARMUP_FRAC * STEPS))

    from megatron.bridge.models.qwen_vl_ov2.llava_ov2_4b import build_llava_ov2_4b, load_ov2_4b_mcore_checkpoint
    model = build_llava_ov2_4b(perform_init=False, use_cpu_init=True, grad_accum_fusion=False)

    resume_iter = latest_iter(OUT)
    if resume_iter > 0:
        ck = os.path.join(OUT, f"iter_{resume_iter:07d}", "mp_rank_00", "model_optim_rng.pt")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=False)
        if rank == 0: print(f"[resume] from iter {resume_iter} ({ck})", flush=True)
    else:
        load_ov2_4b_mcore_checkpoint(model, OVCK, load_adapter=False)  # LLM + new vision; adapter random
        init_adapter(model.adapter)
    model.freeze(freeze_language_model=True, freeze_vision_model=True, freeze_adapter=False)
    model = model.to("cuda", dtype=torch.bfloat16); model.train()
    for p in model.adapter.parameters():
        dist.broadcast(p.data, src=0)

    all_p = [p for p in model.adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(all_p, lr=LR, betas=(0.9, 0.99), eps=1e-5, weight_decay=0.0)  # AIAK adapter-only: adam 0.9/0.99 eps1e-5 wd0
    start = resume_iter
    if resume_iter > 0:
        ostate = os.path.join(OUT, f"iter_{resume_iter:07d}", "optim.pt")
        if os.path.exists(ostate):
            opt.load_state_dict(torch.load(ostate, map_location="cuda", weights_only=False))

    writer = None
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(OUT, "tensorboard"))
        except Exception as e:
            print(f"[tb] disabled: {e}", flush=True)
        print(f"[init] world={world} ACCUM={ACCUM} gbs={gbs} trainable={sum(p.numel() for p in all_p)/1e6:.1f}M "
              f"opt=AdamW(0.9,0.99,eps1e-5,wd0) lr={LR}->min{MIN_LR} cosine warmup={warmup_steps} clip={CLIP_GRAD} "
              f"steps={STEPS} start={start} ckpt={OVCK}", flush=True)

    from megatron.energon import get_train_dataset, get_loader, WorkerConfig
    wc = WorkerConfig(rank=rank, world_size=world, num_workers=2)
    ds = get_train_dataset(DATA, batch_size=1, task_encoder=build_encoder(), worker_config=wc,
                           shuffle_buffer_size=1000, max_samples_per_sequence=None)
    loader = get_loader(ds, worker_config=wc)

    def save_ckpt(step, consumed):
        d = os.path.join(OUT, f"iter_{step:07d}")
        os.makedirs(os.path.join(d, "mp_rank_00"), exist_ok=True)
        sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        torch.save({"model": sd, "iteration": step, "consumed_samples": consumed},
                   os.path.join(d, "mp_rank_00", "model_optim_rng.pt"))
        torch.save(opt.state_dict(), os.path.join(d, "optim.pt"))
        with open(os.path.join(OUT, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(step))
        for o in sorted(glob.glob(os.path.join(OUT, "iter_*")))[:-KEEP_LAST]:
            shutil.rmtree(o, ignore_errors=True)
        print(f"[save] full model -> {d} (consumed={consumed})", flush=True)

    it = iter(loader); t0 = time.time(); nan_iters = 0; run_loss = 0.0; run_tok = 0.0; seen = 0
    for step in range(start, STEPS):
        lr = lr_at(step, STEPS, warmup_steps)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        rank_sum = 0.0; rank_tok = 0
        for _ in range(ACCUM):                       # grad accumulation to reach gbs ~= GBS_TARGET
            b = next(it)
            ids = b["input_ids"].unsqueeze(0).cuda(); labels = b["labels"].unsqueeze(0).cuda()
            # next-token alignment: mcore CE does NOT shift internally, so shift labels left by one
            # (score logits[t] vs input_ids[t+1]); mask the wrapped last pos. AIAK pretrain:150-155.
            labels = torch.roll(labels, shifts=-1, dims=1); labels[:, -1] = IGNORE
            pv = b["pixel_values"].cuda().to(torch.bfloat16) if b["pixel_values"] is not None else None
            grid = b["image_grid_thw"].cuda() if b["image_grid_thw"] is not None else None
            out = model(images=pv, image_grid_thw=grid, input_ids=ids, position_ids=None, attention_mask=None, labels=labels)
            sel = labels != IGNORE
            n = int(sel.sum().item())
            if out.dim() == 2 and n > 0:
                sl = out[sel].float().sum()          # RAW summed loss (token-weighted after /global_tok)
                sl.backward(); rank_sum += float(sl.item()); rank_tok += n
        # token-weighted DP reduction (AIAK): sum grads + sum(loss,tokens) across ranks, divide grads by global token count
        gt = torch.tensor([rank_sum, float(rank_tok)], device="cuda", dtype=torch.float64); dist.all_reduce(gt)
        global_sum = float(gt[0].item()); global_tok = max(1.0, float(gt[1].item()))
        for p in all_p:
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= global_tok
        gnorm = float(torch.nn.utils.clip_grad_norm_(all_p, CLIP_GRAD))  # AIAK --clip-grad 1.0
        step_loss = global_sum / global_tok
        if not math.isfinite(step_loss): nan_iters += 1
        opt.step()
        run_loss += global_sum; run_tok += global_tok; seen += 1
        if rank == 0 and (step % LOG_EVERY == 0 or step == STEPS - 1):
            dt = time.time() - t0; ms = dt / max(seen, 1) * 1000
            consumed = (step + 1) * gbs
            avg = run_loss / max(1.0, run_tok); tput = run_tok / max(dt, 1e-6)
            print(f"iteration {step+1}/{STEPS} | consumed samples: {consumed} | "
                  f"elapsed time per iteration (ms): {ms:.1f} | throughput per GPU (tokens/sec): {tput/world:.1f} | "
                  f"learning rate: {lr:.6E} | global batch size: {gbs} | lm loss: {avg:.6E} | "
                  f"loss scale: 1.0 | grad norm: {gnorm:.3f} | num zeros: 0 | "
                  f"number of skipped iterations: 0 | number of nan iterations: {nan_iters} |", flush=True)
            if writer:
                writer.add_scalar("lm_loss", avg, step + 1); writer.add_scalar("learning_rate", lr, step + 1)
                writer.add_scalar("grad_norm", gnorm, step + 1)
            t0 = time.time(); run_loss = 0.0; run_tok = 0.0; seen = 0
        if rank == 0 and step > start and step % SAVE_EVERY == 0:
            save_ckpt(step, (step + 1) * gbs)
        dist.barrier()
    if rank == 0:
        save_ckpt(STEPS, STEPS * gbs)
        if writer: writer.close()
        print("=== STAGE1 DP RUN DONE ===", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
