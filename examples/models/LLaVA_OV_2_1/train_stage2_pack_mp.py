"""OV2.1-4B stage-2 with BRIDGE-NATIVE ENERGON SEQUENCE PACKING.

Same as train_stage2_mp.py (freeze LLM, train vision+adapter, Muon, multi-turn masking) but
greedy-knapsack-packs several conversations into one <=SEQ sequence and runs block-diagonal
varlen attention via PackedSeqParams (cu_seqlens). Fixes the DP load imbalance -> higher util.
ACCUM here counts PACKS per optimizer step (each pack already holds many samples).
"""
import os, sys, time, glob, shutil, math, itertools, torch
import torch.distributed as dist
import numpy as np
from datetime import timedelta
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muon_standalone import Muon


def greedy_knapsack(sizes, items, cap):
    """Pack items into bins of capacity `cap`. Items LARGER than cap get their OWN single-item bin
    (they hold only 1 image -> fit like the non-packed long samples; multi-item bins stay <=cap so
    they hold few images and fit memory). Returns list of bins."""
    import bisect
    big = [items[i] for i in range(len(sizes)) if sizes[i] > cap]          # oversized -> singleton bin
    sm = sorted((i for i in range(len(sizes)) if sizes[i] <= cap), key=lambda i: sizes[i])
    sz = [sizes[i] for i in sm]; it = [items[i] for i in sm]
    bins = [[b] for b in big]
    while sz:
        cur = []; rem = cap
        while True:
            j = bisect.bisect_right(sz, rem) - 1
            if j < 0:
                break
            rem -= sz[j]; cur.append(it.pop(j)); sz.pop(j)
        if not cur:
            break
        bins.append(cur)
    return bins

STAGE1_CKPT = os.environ.get("STAGE1_CKPT",
    # date0523 lineage: colleague's Muon-trained vit+adapter stage-1 (loads 588/588 into our Bridge model)
    "/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/date0513-corrected-muon-stage1-vit-adapter/date0511_ax_stage_1_alignment_p16m3_packed_new16_muon/release/mp_rank_00/model_optim_rng.pt")
PROC = "/ov2/feilong/preprocessor_p16m33"
DATA = os.environ.get("DATA", "/vlm/data/llava_next_full_mega")
OUT = os.environ.get("OUT", "/ov2/feilong/Megatron-Bridge/ov2_1_4b/stage2_encoder_adapter_780k")
SEQ = int(os.environ.get("SEQ", "32000"))         # skip threshold (keep samples up to SEQ)
PACK_CAP = int(os.environ.get("PACK_CAP", "8000"))  # multi-sample bin cap (memory-bound); samples > PACK_CAP get own bin
ACCUM = int(os.environ.get("ACCUM", "18"))            # GBS = world*MBS*ACCUM (7*1*18=126 ~ ref 128)
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "1000"))  # in optimizer steps
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))
KEEP_LAST = int(os.environ.get("KEEP_LAST", "5"))
LR = float(os.environ.get("LR", "2e-5"))
WARMUP = int(os.environ.get("WARMUP", "0"))   # date0523: warmup 0 (constant lr)
CLIP_GRAD = float(os.environ.get("CLIP_GRAD", "1.0"))
MUON_RMS = float(os.environ.get("MUON_RMS", "0.2"))  # date0523 --muon-matched-adamw-rms 0.2
TOTAL_SAMPLES = int(os.environ.get("TOTAL_SAMPLES", "779111"))
# 1-epoch target measured in RAW samples (a pack holds many). The loop stops when the
# all-reduced raw-sample count reaches TARGET_SAMPLES; total_steps is only a loose upper bound.
TARGET_SAMPLES = int(os.environ.get("TARGET_SAMPLES", "779111"))
START_CONSUMED = int(os.environ.get("START_CONSUMED", "0"))  # raw samples already done (e.g. when continuing a non-packed run)
IMAGE_TOKEN_ID = 151655
IM_END = 151645
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
    asst_hdr = proc.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    class Enc(DefaultTaskEncoder):
        def encode_sample(self, s):
            try:
                msgs = s.messages or []
                if not any(m.get("role") == "assistant" for m in msgs):
                    raise SkipSample()
                imgs = [_to_pil(x) for x in (s.image or [])]
                if not imgs:
                    raise SkipSample()  # stage-2 trains vision+adapter -> text-only samples give no grad
                vis = "<|vision_start|><|image_pad|><|vision_end|>"
                text = f"<|im_start|>system\n{s.system}<|im_end|>\n" if s.system else ""
                for m in msgs:
                    c = m.get("content") or ""
                    role = m.get("role")
                    if role == "user" and "<image>" in c:
                        c = c.replace("<image>", vis)
                    if role in ("user", "assistant"):
                        text += f"<|im_start|>{role}\n{c}<|im_end|>\n"
                ft = proc(text=[text], images=imgs or None, return_tensors="pt")
                ids = ft["input_ids"][0]
                if ids.shape[0] > SEQ:
                    raise SkipSample()
                labels = torch.full_like(ids, IGNORE)
                il = ids.tolist(); L = len(il); H = len(asst_hdr); i = 0
                while i <= L - H:
                    if il[i:i + H] == asst_hdr:
                        j = i + H
                        while j < L and il[j] != IM_END:
                            j += 1
                        end = min(j + 1, L)  # include the closing <|im_end|> in the loss
                        labels[i + H:end] = ids[i + H:end]
                        i = end
                    else:
                        i += 1
                if (labels != IGNORE).sum() == 0:
                    raise SkipSample()
                return {"input_ids": ids, "labels": labels,
                        "pixel_values": ft.get("pixel_values"), "image_grid_thw": ft.get("image_grid_thw")}
            except SkipSample:
                raise
            except Exception:
                raise SkipSample()

        # ---- energon packing protocol ----
        def select_samples_to_pack(self, samples):
            sizes = [s["input_ids"].shape[0] for s in samples]
            return greedy_knapsack(sizes, samples, PACK_CAP)  # multi-sample bins <=PACK_CAP; longer samples -> own bin

        def pack_selected_samples(self, samples):
            ids = torch.cat([s["input_ids"] for s in samples], dim=0)
            labels = torch.cat([s["labels"] for s in samples], dim=0)
            pvs = [s["pixel_values"] for s in samples if s["pixel_values"] is not None]
            grids = [s["image_grid_thw"] for s in samples if s["image_grid_thw"] is not None]
            lens = [s["input_ids"].shape[0] for s in samples]
            cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32)
            return {"input_ids": ids, "labels": labels,
                    "pixel_values": torch.cat(pvs, dim=0) if pvs else None,
                    "image_grid_thw": torch.cat(grids, dim=0) if grids else None,
                    "cu_seqlens": cu}

        def batch(self, samples):
            return samples[0]

    return Enc()


def lr_at(step):
    if WARMUP > 0 and step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR  # constant (date0523: min-lr == lr, warmup 0)


def latest_iter(out):
    f = os.path.join(out, "latest_checkpointed_iteration.txt")
    if os.path.exists(f):
        try:
            return int(open(f).read().strip())
        except Exception:
            return 0
    return 0


def main():
    dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT_S", "7200"))))
    rank = dist.get_rank(); world = dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(1, 1)
    gbs = world * ACCUM
    total_steps = (TOTAL_SAMPLES + gbs - 1) // gbs

    from megatron.bridge.models.qwen_vl_ov2.llava_ov2_4b import build_llava_ov2_4b, load_ov2_4b_mcore_checkpoint
    RECOMPUTE = os.environ.get("RECOMPUTE", "1") == "1"
    model = build_llava_ov2_4b(perform_init=False, use_cpu_init=True, grad_accum_fusion=False, recompute=RECOMPUTE)

    resume_iter = latest_iter(OUT)
    if resume_iter > 0:
        ck = os.path.join(OUT, f"iter_{resume_iter:07d}", "mp_rank_00", "model_optim_rng.pt")
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"], strict=False)
        if rank == 0: print(f"[resume] stage-2 from iter {resume_iter}", flush=True)
    else:
        s = load_ov2_4b_mcore_checkpoint(model, STAGE1_CKPT, load_adapter=True, load_vision=True)  # full stage-1 model
        if rank == 0: print(f"[init] loaded stage-1 full ckpt: loaded={s['loaded']} missing={len(s['missing'])} unexpected={len(s['unexpected'])}", flush=True)
    # stage-2: freeze LLM; train vision encoder + adapter
    model.freeze(freeze_language_model=True, freeze_vision_model=False, freeze_adapter=False)
    model = model.to("cuda", dtype=torch.bfloat16); model.train()
    for p in model.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_p = [p for p in trainable if p.dim() == 2]
    other_p = [p for p in trainable if p.dim() != 2]
    opt = Muon([{"params": muon_p, "use_muon": True}, {"params": other_p, "use_muon": False}],
               lr=LR, weight_decay=0.0, matched_adamw_rms=MUON_RMS, momentum=0.95, ns_steps=5,
               nesterov=True, adamw_betas=(0.9, 0.99), adamw_eps=1e-5)
    start = resume_iter
    if resume_iter > 0:
        op = os.path.join(OUT, f"iter_{resume_iter:07d}", "optim.pt")
        if os.path.exists(op): opt.load_state_dict(torch.load(op, map_location="cuda", weights_only=False))

    writer = None
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(OUT, "tensorboard"))
        except Exception as e:
            print(f"[tb] disabled: {e}", flush=True)
        print(f"[init] world={world} ACCUM={ACCUM} gbs={gbs} seq={SEQ} total_steps={total_steps} "
              f"trainable={sum(p.numel() for p in trainable)/1e9:.3f}B (vision+adapter) lr={LR} start={start}", flush=True)

    from megatron.energon import get_train_dataset, get_loader, WorkerConfig
    wc = WorkerConfig(rank=rank, world_size=world, num_workers=2)
    ds = get_train_dataset(DATA, batch_size=1, task_encoder=build_encoder(), worker_config=wc,
                           shuffle_buffer_size=500, max_samples_per_sequence=None,
                           packing_buffer_size=int(os.environ.get("PACK_BUF", "64")))
    loader = get_loader(ds, worker_config=wc); it = iter(loader)

    def save_ckpt(step, consumed):
        d = os.path.join(OUT, f"iter_{step:07d}"); os.makedirs(os.path.join(d, "mp_rank_00"), exist_ok=True)
        sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        torch.save({"model": sd, "iteration": step, "consumed_samples": consumed},
                   os.path.join(d, "mp_rank_00", "model_optim_rng.pt"))
        torch.save(opt.state_dict(), os.path.join(d, "optim.pt"))
        open(os.path.join(OUT, "latest_checkpointed_iteration.txt"), "w").write(str(step))
        for o in sorted(glob.glob(os.path.join(OUT, "iter_*")))[:-KEEP_LAST]:
            shutil.rmtree(o, ignore_errors=True)
        print(f"[save] full model -> {d} (consumed={consumed})", flush=True)

    t0 = time.time(); nan_iters = 0; run_loss = 0.0; run_tok = 0.0; seen = 0; toks = 0
    consumed_raw = 0; global_consumed = START_CONSUMED  # consumed_raw = raw samples this rank pulled this run
    from megatron.core.packed_seq_params import PackedSeqParams
    for step in range(start, total_steps):
        lr = lr_at(step)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True); rank_sum = 0.0; rank_tok = 0
        for _ in range(ACCUM):
            b = next(it)
            ids = b["input_ids"].unsqueeze(0).cuda(); labels = b["labels"].unsqueeze(0).cuda()
            pv = b["pixel_values"].cuda().to(torch.bfloat16) if b["pixel_values"] is not None else None
            grid = b["image_grid_thw"].cuda() if b["image_grid_thw"] is not None else None
            cu = b["cu_seqlens"].to(torch.int32).cuda()
            consumed_raw += int(cu.shape[0] - 1)  # samples in this pack (cu has n_samples+1 offsets)
            # next-token alignment for PACKED seq: roll left by one, then mask EACH sub-sample's last
            # position (cu[1:]-1) so no label leaks across pack boundaries. block-diagonal attention
            # (cu at SAMPLE boundaries) keeps within-conversation causal context but blocks cross-sample.
            labels = torch.roll(labels, shifts=-1, dims=1); labels[:, cu[1:].long() - 1] = IGNORE
            maxlen = int((cu[1:] - cu[:-1]).max())
            psp = PackedSeqParams(cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=maxlen,
                                  max_seqlen_kv=maxlen, qkv_format="thd")
            out = model(images=pv, image_grid_thw=grid, input_ids=ids, position_ids=None,
                        attention_mask=None, labels=labels, packed_seq_params=psp)
            sel = labels != IGNORE
            n = int(sel.sum().item())
            if out.dim() == 2 and n > 0:
                sl = out[sel].float().sum()           # RAW summed loss (token-weighted after /global_tok)
                sl.backward(); rank_sum += float(sl.item()); rank_tok += n; toks += int(ids.shape[1])
        # token-weighted DP reduction + coalesced grad all-reduce (one flat NCCL op)
        gt = torch.tensor([rank_sum, float(rank_tok)], device="cuda", dtype=torch.float64); dist.all_reduce(gt)
        global_sum = float(gt[0].item()); global_tok = max(1.0, float(gt[1].item()))
        grads = [p.grad for p in trainable if p.grad is not None]
        if grads:
            flat = torch._utils._flatten_dense_tensors(grads); dist.all_reduce(flat); flat /= global_tok
            for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)): g.copy_(s)
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_GRAD))  # date0523 --clip-grad 1.0
        step_loss = global_sum / global_tok
        if not math.isfinite(step_loss): nan_iters += 1
        opt.step()
        run_loss += global_sum; run_tok += global_tok; seen += 1
        # global raw-sample count across all DP ranks (each rank pulled a disjoint data shard)
        cr_t = torch.tensor([consumed_raw], device="cuda", dtype=torch.long); dist.all_reduce(cr_t)
        global_consumed = START_CONSUMED + int(cr_t.item())
        if rank == 0 and (step % LOG_EVERY == 0 or global_consumed >= TARGET_SAMPLES):
            dt = time.time() - t0; ms = dt / max(seen, 1) * 1000; consumed = global_consumed
            tput = toks / max(dt, 1e-6); avg = run_loss / max(1.0, run_tok)  # toks is rank-0 (per-GPU) -> no /world
            est_total = max(step + 1, int((step + 1) * TARGET_SAMPLES / max(global_consumed, 1)))  # data-driven 1-epoch step estimate
            print(f"iteration {step+1}/{est_total} | consumed samples: {consumed} | "
                  f"elapsed time per iteration (ms): {ms:.1f} | throughput per GPU (tokens/sec): {tput:.1f} | "
                  f"learning rate: {lr:.6E} | global batch size: {gbs} | lm loss: {avg:.6E} | "
                  f"loss scale: 1.0 | grad norm: {gnorm:.3f} | num zeros: 0 | "
                  f"number of skipped iterations: 0 | number of nan iterations: {nan_iters} |", flush=True)
            if writer:
                writer.add_scalar("lm_loss", avg, step + 1); writer.add_scalar("learning_rate", lr, step + 1)
                writer.add_scalar("grad_norm", gnorm, step + 1)
            t0 = time.time(); run_loss = 0.0; run_tok = 0.0; seen = 0; toks = 0
        if rank == 0 and step > start and step % SAVE_EVERY == 0:
            save_ckpt(step, global_consumed)
        # no per-step barrier: the grad + consumed all_reduce above already sync all ranks
        if global_consumed >= TARGET_SAMPLES:  # 1 epoch (raw samples) reached -> stop on all ranks (same global_consumed on all ranks)
            if rank == 0: print(f"[done] reached {global_consumed}/{TARGET_SAMPLES} raw samples at step {step+1}", flush=True)
            break
    if rank == 0:
        save_ckpt(step + 1, global_consumed)
        if writer: writer.close()
        print("=== STAGE2 DP RUN DONE ===", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
