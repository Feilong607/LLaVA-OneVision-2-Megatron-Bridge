# LLaVA-OneVision-2.1 — 4B (p16m33) on Megatron-Bridge

Two-stage build of the **genuine 4B** OV2.1 model under the NVIDIA Megatron-Bridge
framework, bootstrapped from the already-assembled mcore checkpoint.

> Status: scaffold + launch scripts. The two code pieces marked **[BUILD]** below
> must be added/validated in the training env before stage-2 can run end-to-end.
> Stage-1 (packed 558k) runs on the **existing** OV2 packed pipeline.

---

## Model (resolved from `lmms-lab/LLaVA-OneVision-2-4B-p16m33/config.json`)

| Part | Spec |
|------|------|
| LLM backbone | **Qwen3-4B-Instruct-2507** — dense Qwen3, hidden **2560**, **36** layers, heads 32 / kv 8, head_dim 128, ffn 9728, rope_θ 5e6, rms_eps 1e-6, vocab 151936 |
| Vision encoder | onevision_encoder, **patch_size 16**, hidden 1024, 24 layers, 16 heads, ffn 4096, **spatial_merge_size 3**, image 448, out_hidden 2560 |
| Adapter (m33) | input 1024 → `hidden = 1024 * 3² = 9216` → output **2560** (layernorm@1024, linear_fc1 9216→9216, linear_fc2 9216→2560) |
| image_token_id | 151655 |

NB vs the 8B graft: 8B used patch **14** / merge **2** / adapter→4096 on a **Qwen3-VL-8B** LLM.
This is a different model — patch **16**, merge **3**, adapter→**2560**, plain **Qwen3-4B** LLM.

## Assets (all verified present on this host)

| Asset | Path |
|-------|------|
| Assembled mcore ckpt (LLM + p16m33 encoder + adapter) | `/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_p16m33_mcore_tp1_pp1` (`release/mp_rank_00/model_optim_rng.pt`) |
| HF reference model | `/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33` |
| LLM HF | `/ov2/pretrain_models/Qwen3-4B-Instruct-2507` |
| Standalone encoder (HF) | `/ov2/pretrain_models/onevision-encoder-large` |
| Preprocessor (p16/m3) | `/ov2/feilong/preprocessor_p16m33` |
| Stage-1 data — 558k **non-packed** (MultiMixQA) | `/vlm/data/blip_laion_cc_sbu_558k_wds` |
| Stage-2 data — 780k **non-packed** (MultiMixQA) | `/vlm/data/llava_next_full_mega` |
| Bridge runtime | docker image `mbridge:qwen35` (has megatron.core 0.18.0; run `--gpus all`, `PYTHONPATH=<repo>/src:<repo>/3rdparty/Megatron-LM`) |

## Design (per request)

- **Init**: load LLM + vision encoder from the assembled mcore ckpt; **adapter random-init**.
- **Stage 1** — train **adapter only** (freeze LLM + encoder), 558k **non-packed** data, **1 epoch**, **AIAK adapter-only recipe**: Adam (0.9,0.99,eps1e-5,wd0), lr 2e-5→cosine→1e-6 (warmup 0.002), clip-grad 1.0, gbs 256 via grad-accum, GELU adapter, next-token label shift, token-weighted loss. (The label shift is mandatory — mcore CE does not shift internally; without it loss starts >random ~16.)
- **Stage 2** — train **encoder + adapter** (freeze LLM), 780k **non-packed** data, **1 epoch**, optimizer **Muon** (`dist_muon`, momentum 0.95, matched_adamw_rms 0.15, scalar-Adam β 0.9/0.99 eps 1e-5 — mirrors the date0528 reference).
- Both stages use **non-packed MultiMixQA** data → one task encoder ([BUILD-2]) covers both. No packed pipeline used.
- **No FSDP** (use the distributed optimizer; `data_parallel_sharding_strategy` only).

## Assembled mcore ckpt — verified layout (the "stitch" target)

`llava_onevision2_4b_p16m33_mcore_tp1_pp1` has **3 sibling namespaces** (standard mcore):
- `language_model.*` (435) — Qwen3-4B GPTModel (embedding/decoder/output_layer, TE, qk-layernorm)
- `vision_model.*` (388) — onevision encoder directly (qkv has bias; 3072 = 3×1024)
- `adapter.*` — layernorm(1024) + linear_fc1(9216×9216) + linear_fc2(2560×9216)

Bridge's OV2 `onevision_encoder_model` + `adapter` are **verbatim AIAK ports**, so a Bridge model
with the **same 3-sibling layout** (port of AIAK `LlavaOnevision2`) loads this ckpt with
near-identity naming (drop `_extra_state`; adapter re-init for stage-1). That model class is [BUILD-1].

> Muon caveat: Bridge `dist_muon` needs `emerging_optimizers` (git v0.2.0). It is **missing** from
> `mbridge:qwen35` — pip-install into the image, or use an image that has it, before stage-2.

## Two engineering gaps (why this isn't pure config)

**[BUILD-1] Multimodal model class for 4B.** The repo ported only the OV2 *vision tower*
(`models/qwen_vl_ov2/{onevision_encoder_model,adapter,...}` — verbatim from AIAK, so their
param names match the mcore ckpt's `vision_model.*` / `adapter.*`). The 8B recipe reused
`Qwen3VLModel`'s merge/forward by grafting the tower into a **Qwen3-VL-8B**. There is **no
Qwen3-VL-4B**, and the 4B LLM is plain Qwen3-4B. So we need a multimodal wrapper holding
`language_model` (Qwen3-4B GPTModel) + `vision_model` (OV2 tower) that loads the AIAK mcore
ckpt cleanly. Recommended: **port AIAK's `LlavaOnevision2Model`** (forward = encode→adapter→
scatter at image_token→LLM) since the mcore ckpt was saved by that exact class → loads 1:1.

**[BUILD-2] Non-packed MultiMixQA task encoder (stage-2 only).** `llava_next_full_mega` is an
Energon WDS whose `dataset.yaml` sample_type is `aiak_training_llm.data.multimodal.MultiMixQASample`
(multi-turn conversation, image/video). The current `aiak_shim` only re-exports
`PackedCaptioningSample`, and `OV2PackingTaskEncoder` raises `NotImplementedError` on anything
but `PackedCaptioningSample`. So stage-2 needs (a) a `MultiMixQASample` export in the shim and
(b) a non-packed conversation task encoder (qwen2-vl chat template, mask non-assistant tokens).
**Stage-1 (558k packed) needs neither — it uses the existing packed pipeline.**

## Run

```bash
cd /ov2/feilong/Megatron-Bridge
bash examples/models/LLaVA_OV_2_1/ax_stage_1_alignment_p16m3_adapter_only.sh   # adapter align (558k, AIAK recipe)
bash examples/models/LLaVA_OV_2_1/stage2_encoder_adapter_780k.sh              # encoder+adapter (780k)
```

Both scripts expose every hyper-parameter as an overridable env var at the top.
