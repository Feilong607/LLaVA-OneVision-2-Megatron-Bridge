# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Local / A100 variants of the official Qwen3-30B-A3B MoE recipes.

These MIRROR recipes/qwen/qwen3_moe.py::qwen3_30b_a3b_pretrain_config exactly (same MoE
acceleration + parallelism), with only the two changes needed to run on our offline A100 cluster:

  1. LOCAL HF PATH (offline): the official recipe hardcodes the Hub id "Qwen/Qwen3-30B-A3B", which
     fails under HF_HUB_OFFLINE=1. Here `hf_path` defaults to the local mirror and is overridable via
     run_recipe.py --hf_path. We use AutoBridge.from_hf_config (config-only, no 60GB weight load) so a
     random-init pipeline smoke builds fast; flip USE_HF_WEIGHTS=True to load real weights.
  2. NO DeepEP (A100): the official recipe calls apply_flex_dispatcher_backend("deepep"), which on
     Ampere flips moe_token_dispatcher_type -> "flex" + backend "deepep"; deep_ep is NOT installed in
     the A100 image, so the flex dispatcher fails at runtime. We keep the portable "alltoall"
     dispatcher (grouped-GEMM experts) — correct, just less optimized than DeepEP. The GB200 script
     re-enables DeepEP/HybridEP.

Everything else (TP4/PP2/EP4, SP, grouped_gemm, permute_fusion, recompute full, distributed
optimizer + overlap, TE, manual GC, bf16) is identical to the official recipe.
"""
import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer

_DEFAULT_LOCAL_HF = "/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507"


def _build_qwen3_moe_provider(hf_path: str, load_weights: bool):
    """Build the Megatron provider from a LOCAL HF dir (offline-safe).

    Uses from_hf_config (config only) by default — no multi-GB weight load — so a random-init
    pipeline smoke is fast. Set OV2_LOAD_HF_WEIGHTS=1 to load real weights instead.
    """
    if load_weights:
        return AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(
            load_weights=True
        )
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    return AutoBridge.from_hf_config(hf_cfg).to_megatron_provider(load_weights=False)


def qwen3_30b_a3b_pretrain_a100_config(hf_path: str = _DEFAULT_LOCAL_HF) -> ConfigContainer:
    """Qwen3-30B-A3B MoE pre-training on offline A100 (mirrors the official recipe; alltoall, no DeepEP).

    Parallelism: TP=4, PP=2, EP=4, SP=True. On 1 node (8 GPU) DP=1; on 2 nodes (16 GPU) DP=2.
    Run (mock data + random init smoke):
      torchrun ... run_recipe.py --recipe qwen3_30b_a3b_pretrain_a100_config \
        --dataset llm-pretrain-mock train.train_iters=20 logger.log_interval=1
    """
    load_weights = os.environ.get("OV2_LOAD_HF_WEIGHTS", "0") == "1"
    cfg = _pretrain_common()

    cfg.model = _build_qwen3_moe_provider(hf_path, load_weights)
    cfg.tokenizer.tokenizer_model = hf_path

    # Dataset - mock by default (pipeline smoke); set dataset.blend for real data.
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # --- Parallelism (identical to official qwen3_30b_a3b_pretrain_config) ---
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # --- MoE dispatcher: A100 portable path (NO DeepEP) ---
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    # NB: deliberately NOT calling apply_flex_dispatcher_backend() (it would flip to flex/deepep on Ampere).

    # --- Training ---
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # --- TE + kernels (identical to official) ---
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    # fp32 routing for >=32 experts (numerical stability; official emits a warning when off).
    cfg.model.moe_router_dtype = "fp32"

    # --- Memory saving (full recompute, as official 30B pretrain) ---
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.moe_router_padding_for_fp8 = False

    # --- Optimizer precision (identical) ---
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # --- DDP (identical) ---
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    cfg.model.moe_router_force_load_balancing = False
    return cfg


def qwen3_30b_a3b_pretrain_gb200_config(hf_path: str = _DEFAULT_LOCAL_HF) -> ConfigContainer:
    """Qwen3-30B-A3B MoE pre-training tuned for GB200 (Grace-Blackwell, NVL72).

    Starts from the official A100/H100 config and flips on the Blackwell-class accelerators:

      * HybridEP dispatcher (NOT DeepEP): apply_flex_dispatcher_backend gates "deepep" on
        device major in {8,9} or name "NVIDIA B200/B300"; GB200 reports major=10 / name "NVIDIA GB200"
        so DeepEP is SKIPPED there, but "hybridep" is gated on major in {8,9,10} and is the intended
        NVL72 path -> we request "hybridep" and let apply_flex_dispatcher_backend set token_dispatcher
        ="flex". HybridEP keeps all-to-all expert traffic inside the NVLink domain.
      * MXFP8 (Blackwell-native FP8): mixed_precision = bf16_with_mxfp8_mixed(); moe_router_padding_for_fp8
        =True so the MoE router shapes are FP8-friendly.
      * Comm overlap: overlap EP all-to-all with compute + delay wgrad (big win on GB200's fast NVLink).

    Parallelism: BASELINE = official TP4/PP2/EP4 (portable). On a full GB200 NVL72 domain you typically
    DROP TP/PP and grow EP across the NVLink domain (e.g. TP1/PP1/EP8..32) because HybridEP makes
    NVLink-domain expert parallel cheap — tune EP to your rack / token-routing balance. seq_length and
    global_batch_size scale with the domain; values below are a safe starting point. NOT validated on
    GB200 hardware here (no GB200 access) — treat parallelism/EP as the first knobs to tune.
    """
    from megatron.bridge.training.comm_overlap import CommOverlapConfig
    from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
    from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed

    load_weights = os.environ.get("OV2_LOAD_HF_WEIGHTS", "0") == "1"
    cfg = _pretrain_common()

    cfg.model = _build_qwen3_moe_provider(hf_path, load_weights)
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # --- Parallelism: portable baseline (tune EP up on a full NVL72 domain) ---
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # --- MoE dispatcher: HybridEP (Blackwell / NVL72) ---
    cfg.model.moe_token_dispatcher_type = "alltoall"          # base value; flex set by apply_* below
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16

    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # --- TE + kernels (as official) ---
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.moe_router_dtype = "fp32"
    # CUDA graphs help on GB200; enable once stable (start "none" to de-risk first bring-up).
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # --- Memory: 30B fits without recompute on GB200's larger HBM; keep off for speed (re-enable if OOM). ---
    cfg.model.recompute_granularity = None

    # --- Blackwell FP8 (MXFP8) ---
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.model.moe_router_padding_for_fp8 = True

    # --- Optimizer precision ---
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # --- Communication overlap (GB200 fast NVLink) ---
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    # --- DDP ---
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.model.moe_router_force_load_balancing = False

    # Activate HybridEP (sets moe_token_dispatcher_type="flex" when GPU is Blackwell-class).
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)
    return cfg
