# How Qwen3-VL Runs in NVIDIA Megatron-Bridge

**Target:** Qwen3-VL-4B-Instruct (dense) on 8× A800-80GB, docker `mbridge:qwen35` (megatron.core 0.18.0; `PYTHONPATH=<repo>/src:<repo>/3rdparty/Megatron-LM`), first test on `/vlm/data/blip_laion_cc_sbu_558k_wds`.

> **Critical up-front fact:** There is **no `qwen3_vl_4b_*` recipe** in the repo. `qwen3_vl.py` ships only **8B dense**, **30B-A3B MoE**, **235B-A22B MoE**. The `qwen35_vl_4b_*` recipes are a *different model family* (Qwen3.5-VL, hybrid GDN+attn). Qwen3-VL-4B-Instruct is dense Qwen3-VL and routes through `Qwen3VLBridge` / `Qwen3VLModelProvider`, so the actionable plan is to **clone the 8B dense recipe and repoint `hf_path` at the 4B model**.

---

## 1. Big picture

Four-layer pipeline, each layer doing one job:

- **HF model** — `transformers` `Qwen3VLForConditionalGeneration` (text_config + vision_config). Source of truth for every dimension; nothing hardcoded by size.
- **AutoBridge** — `AutoBridge.from_hf_pretrained(path)` loads HF config + lazy safetensors, dispatches by architecture string to `Qwen3VLBridge` (`qwen3_vl_bridge.py:41`, `model_type="qwen3_vl"`). `provider_bridge()` translates HF config → Megatron provider; `mapping_registry()` declares HF↔mcore weight names. `to_megatron_provider(load_weights=...)` returns the provider and (if True) registers a pre-wrap hook streaming weights in.
- **mcore provider** — `Qwen3VLModelProvider` (`GPTModelProvider` subclass). `.provide()` builds `Qwen3VLModel` (one `MegatronModule` wrapping ViT + LLM `GPTModel`), wiring TP/PP/CP/EP, freeze flags, kernels, mrope, deepstack.
- **Trainer** — `scripts/training/run_recipe.py` resolves a recipe (builder → `ConfigContainer`), applies CLI overrides, picks `qwen3_vl_step`, calls `pretrain()`/`finetune()` → `setup()` → `train()`.

Mental model: **HF config defines the model; AutoBridge translates it; the provider builds & parallelizes it; recipe + run_recipe.py assemble and run it.**

---

## 2. Model architecture

**Top-level** `Qwen3VLModel` (`model.py:55`) = `Qwen3VLVisionModel` (ViT, first PP stage only) + `Qwen3VLGPTModel` (LLM decoder, subclass of MCore `GPTModel`).

- **Vision tower:** `nn.Conv3d` patch embed (temporal×patch×patch); interpolated learned abs pos-embed + 2D rotary; MCore TE ViT layers with `self_attention` overridden to `Qwen3VLSelfAttention` (**non-causal/full**, THD packed, each frame = one sequence); `Qwen3VLVisionPatchMerger` merges `spatial_merge_size²`(=4) patches and projects `hidden*4 → out_hidden (== LLM hidden)`.
- **DeepStack:** PatchMergers at `deepstack_visual_indexes` (4B = `[5,11,17]`); tapped ViT-layer features are **additively injected into the first few LLM decoder layers**.
- **Vision-token injection** (`model.py:333-645`): `reorganize_inputs` builds `vision_mask[b,s]`; run ViT → `vision_embeds`; text embeddings; **scatter-write** `combined_embeddings[vision_mask] = vision_embeds`; deepstack added at early layers.
- **MRoPE:** `position_ids` forced `None` (model computes them); `get_rope_index` builds `[3,b,s]` t/h/w indices; `apply_interleaved_mrope` per `mrope_section [24,20,20]`; applied **per-token absolute** to q/k, bypassing MCore's fused dispatcher.
- **Config derivation:** language dims from HF `text_config`; vision config via `get_vision_model_config`.

**Parallelism:** LLM decoder supports TP/PP/CP/EP(MoE)/VPP/SP. Vision tower: **TP inherited from LLM; PP/CP/EP forced to 1.** Constraints: `apply_rope_fusion` must stay **False** (asserted); `deepstack_visual_indexes` all `<` num LLM layers on first PP stage; CP requires `calculate_per_token_loss=True`.

---

## 3. Weight bridge & checkpoints

`Qwen3VLBridge` (dense) + `Qwen3VLMoEBridge` registered against HF arch classes. Each implements `provider_bridge()` (HF config → provider) and `mapping_registry()` (name map).

- **Language:** `embedding.word_embeddings ↔ model.language_model.embed_tokens`; `output_layer ↔ lm_head`; `final_layernorm ↔ ...norm`. Per-layer LN fused into `linear_qkv.layer_norm_weight` / `mlp.linear_fc1.layer_norm_weight`; `q_norm/k_norm` → `q_layernorm/k_layernorm` (Qwen3 QK-LayerNorm). QKV: separate HF q/k/v → interleaved GQA `linear_qkv` (`QKVMapping`); `gate+up → linear_fc1` (`GatedMLPMapping`).
- **Vision:** mcore `vision_model.decoder.layers.*` ↔ HF `model.visual.blocks.*`. Vision attention uses a **single packed HF qkv** (`ConcatenatedQKVMapping`) — language uses separate q/k/v (don't confuse). patch_embed/pos_embed use `ReplicatedMapping` (not TP-shardable).

**`load_weights` semantics:** `load_weights=True` sets `perform_initialization=False` + pre-wrap hook streaming weights in; `load_weights=False` → random init, weights instead come from a converted `torch_dist` checkpoint via `checkpoint.pretrained_checkpoint` (what the recipes do).

**Two checkpoint notions:** HF safetensors vs native Megatron `torch_dist` sharded. `import_ckpt` = HF→mcore→torch_dist; `export_ckpt` = inverse. **torch_dist is reshardable** — import on 1 GPU, train on 8 with any TP/PP/EP topology. Tool: `examples/conversion/convert_checkpoints.py {import,export}`.

**4B-specific bridge notes:** all dims read from `text_config` (size-agnostic). `tie_word_embeddings` read from **top-level** config (4B = True → `output_layer` dropped, expected). Dense bridge reads `text_config.head_dim` with **no fallback** (4B = 128 ✓). `mrope_section` default `[24,20,20]` (4B ✓).

---

## 4. Data pipeline

Two modes dispatched by `apply_dataset_override` (`dataset_utils.py:152`); both emit the same batch contract (`input_ids/labels/loss_mask/position_ids` + a `Qwen2_5_VLVisualInputs` under `batch['visual_inputs']`) consumed by `qwen3_vl_step`:

- **vlm-hf:** HF dataset via a named maker (`make_cord_v2_dataset`), tokenized + vision-processed **at collate time** by `Qwen3VLProcessor` (`qwen2_5_collate_fn`). **Labels left-shifted by 1** in the collator. Needs `pip install qwen-vl-utils`.
- **vlm-energon:** Megatron-Energon WebDataset (a `.nv-meta/` dir) of ChatML tar shards, processed per-sample in `QwenVLTaskEncoder` (`task_encoder.py:197`), which expands the `<image>` placeholder to `prod(t,h,w)//merge²` tokens and left-shifts labels. Visual-token budget knobs: `min/max_pixels`, `max_num_images/frames`, `max_visual_tokens`.
  - **`--dataset vlm-energon` only keeps a usable provider if the recipe already supplies a `QwenVLEnergonProvider`** (with the task encoder); otherwise it builds a *bare* `EnergonProvider` with no encoder. → need a 4B *energon* recipe variant.
  - `QwenVLTaskEncoder` is `DefaultTaskEncoder[ChatMLSample, ...]` → energon must yield **`ChatMLSample`** (`conversation`: JSON string, `imgs`: list of tensors).

**Label shift:** pre-shifted by the **data layer** in both modes. The model/step do **not** shift (see §5). Feeding unshifted labels silently trains an identity objective (same class of bug as the OV2.1-4B `torch.roll` fix).

**Verdict on `/vlm/data/blip_laion_cc_sbu_558k_wds`:** It is LLaVA-Pretrain 558k. It *is* energon-prepared, **but** its `.nv-meta/dataset.yaml` declares `sample_type: aiak_training_llm.data.multimodal.MultiMixQASample` — an **AIAK class absent from `mbridge:qwen35`**, and the wrong schema for Bridge. Its shards hold `<id>.img_000.jpg` + `<id>.json` (`{messages, image_keys}`). **Fix:** build a **sibling** energon dataset (symlink the shards; never mutate the shared original) with a Bridge-compatible `.nv-meta` whose `sample_type` is `megatron.bridge.data.energon.task_encoder_utils.ChatMLSample` and a small `sample_loader.py` that emits `conversation = json.dumps(messages)` and `imgs = [decoded image]`. Then run via a 4B energon recipe with `dataset.path=<sibling dir>`.

---

## 5. Training / forward step (`qwen3_vl_step.py`)

`forward_step` (`:210`), registered as `qwen3_vl_step`.

- **Batch unpack:** moves tensors + each field of `visual_inputs` to CUDA; `labels/loss_mask` only on last PP stage; `visual_inputs.normalized_for_model()` flattens `[B,N,...]→[B*N,...]`. Video supported.
- **Padding/packing in the step (not the dataset):** pad/pack to `divisible_by = tp*cp*2 (cp>1) else tp`, LCM 16 for FP8; `force_to_pad_to_seq_len = seq_length` when PP>1 or EP>1. (Differs from `gpt_step` because the VLM needs original un-CP-split `input_ids` to splice vision tokens + compute MRoPE.)
- **CP:** slices across CP group, then **restores** un-sliced `input_ids`, forces `position_ids=None`; loss uses CP-sliced `loss_mask`.
- **PackedSeqParams:** THD always built, only attached when `pack_sequences_in_batch=True`.
- **Label shift — none** anywhere in step/model/`compute_language_model_loss`. Labels must arrive pre-shifted.
- **Loss:** `masked_next_token_loss` — `sum(per_token_losses * loss_mask)`, `num_tokens = loss_mask.sum()`, reports `{'lm loss'}`.
- **MoE aux loss:** not handled in step (folded into backward inside MCore). Irrelevant for dense 4B.

---

## 6. Recipe & launcher flow (`run_recipe.py`)

1. **Launch:** `python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py ...`; `world_size = TP*PP*EP*CP*DP`.
2. **Recipe resolution:** `getattr(megatron.bridge.recipes, name)` — recipes `__init__` does `from ...qwen_vl import *`, so all builders are top-level. Unknown name → `AttributeError`. (So a new 4B recipe must be exported in `qwen_vl/__init__.py.__all__`.)
3. **Arg injection:** only forwards `peft_scheme/packed_sequence/seq_length/hf_path` kwargs the builder actually accepts (the parameterless SFT/PEFT configs ignore `--hf_path`; the `**kwargs` mock config accepts it but `peft=None` injection then breaks it — so **add explicit 4B builders** rather than rely on `--hf_path`).
4. **Dataset selection:** `--dataset` → `apply_dataset_override` replaces `cfg.dataset`; omitting it keeps the recipe's built-in dataset. `infer_mode_from_dataset`: `llm-pretrain*` → pretrain, else finetune.
5. **Forward step:** `--step_func qwen3_vl_step`.
6. **CLI overrides:** Hydra-style dotted paths in struct mode (typos raise).
7. **seq_length sync:** forces `model.seq_length = dataset.seq_length`.
8. **Train func:** `finetune()` for SFT/PEFT — **asserts `checkpoint.pretrained_checkpoint` OR `checkpoint.load` is set**, then → `pretrain()` → `setup()` → `train()`.
9. **Checkpoint:** `pretrained_checkpoint` = weights-only finetune; `checkpoint.load` = full resume. Saves torch_dist `iter_XXXXXXX` every `save_interval`.

Env: `WORKSPACE`, `WANDB_*`, `HF_HOME`/`HF_TOKEN`.

---

## 7. RUN PLAN — Qwen3-VL-4B-Instruct on 8× A800-80GB

HF dir already at `/ov2/pretrain_models/Qwen3-VL-4B-Instruct` — everything local (set `HF_HUB_OFFLINE=1`).

**Pre-flight (verified):** arch `Qwen3VLForConditionalGeneration`; `head_dim=128` present; heads 32 / KV 8 (TP=2 valid); 36 layers; mrope `[24,20,20]`; deepstack `[5,11,17]` (all <36); `tie_word_embeddings=True` (output_layer dropped — fine).

**Parallelism:** **TP=2, PP=1, CP=1, EP=1, DP=4** (world=8). 4B bf16 (~8GB) + fp32 optimizer state (sharded across DP) fits easily; TP=2 buys activation headroom at seq 4096.

**(a)** Add `qwen3_vl_4b_{pretrain_mock,sft,peft,sft_energon}_config` by cloning the 8B dense builders, repointing `hf_path → /ov2/pretrain_models/Qwen3-VL-4B-Instruct`; export in `qwen_vl/__init__.py`.

**(b)** Convert weights once: `convert_checkpoints.py import --hf-model /ov2/pretrain_models/Qwen3-VL-4B-Instruct --megatron-path <mcore dir>`; pass bare path as `checkpoint.pretrained_checkpoint` (torch_dist is reshardable).

**(c) FASTEST smoke (no data/weights):** `--recipe qwen3_vl_4b_pretrain_mock_config --step_func qwen3_vl_step` — `MockVLMConversationProvider` generates synthetic batches; validates 8-GPU model build + forward/backward.

**(d) Real SFT:** add `checkpoint.pretrained_checkpoint`, run on cord_v2 (vlm-hf) or the 558k energon (sibling ChatML dataset). `mbs=1` start, push up after stable.

**(f) A800 gotchas:** no FP8/MXFP8 (Blackwell-only); `apply_rope_fusion` stays False; `finetune()` requires a checkpoint; verify `megatron.core` resolves to the 3rdparty submodule; `qwen_vl_utils` needed for the cord_v2 vlm-hf path (not for mock/energon).
