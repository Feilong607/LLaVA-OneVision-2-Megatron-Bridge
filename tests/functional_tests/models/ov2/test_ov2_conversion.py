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

"""Functional HF<->Megatron round-trip for the LLaVA-OneVision-2 (OV2) composite bridge.

Toy round-trip on a shrunk Qwen3-30B-A3B-p16m33 composite (model_type=llava_onevision2_moe):
build a random toy HF model -> AutoBridge HF->mcore (load_hf_weights) -> mcore->HF (export_hf_weights)
-> assert every persisted HF param round-trips byte-for-value (atol/rtol 2e-2 in bf16).

GPU test (single rank, TP1/PP1/EP1). use_patch_position_encoding is set False to match the OV2 mcore
adapter (built -pos-less); the expert tensors are PER-EXPERT on disk (qwen3_moe storage)."""

import glob
import json
import os
import shutil
import tempfile

import pytest
import torch

SRC = "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OV2 bridge import (fla/triton) + build need a GPU")
@pytest.mark.skipif(not os.path.isdir(SRC), reason="OV2 composite HF skeleton not present")
def test_ov2_hf_megatron_roundtrip():
    import torch.distributed as dist
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    import megatron.bridge.models.qwen_vl_ov2.ov2_bridge  # noqa: F401 (registers the OV2 bridge)
    from megatron.bridge import AutoBridge

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("OV2_SKIP_BASE_STITCH", "1")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    cfg = json.load(open(SRC + "/config.json"))
    tc = cfg["text_config"]
    tc.update(dict(num_hidden_layers=2, num_experts=4, hidden_size=256, moe_intermediate_size=128,
                   intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=2048))
    vc = cfg["vision_config"]
    vc.update(dict(num_hidden_layers=2, out_hidden_size=256, use_patch_position_encoding=False))
    toy = tempfile.mkdtemp(prefix="ov2_rt_")
    for f in os.listdir(SRC):
        if f.endswith(".py") or f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json"):
            shutil.copy(os.path.join(SRC, f), toy)
    json.dump(cfg, open(toy + "/config.json", "w"))

    hfcfg = AutoConfig.from_pretrained(toy, trust_remote_code=True)
    hfm = AutoModelForCausalLM.from_config(hfcfg, trust_remote_code=True).to(torch.bfloat16)
    hfm.save_pretrained(toy, safe_serialization=True)
    del hfm
    torch.cuda.empty_cache()

    ref = {}
    for sf in glob.glob(toy + "/*.safetensors"):
        ref.update({k: v.float().cpu() for k, v in load_file(sf).items()})
    assert len(ref) > 0

    bridge = AutoBridge.from_hf_pretrained(toy, trust_remote_code=True)
    assert type(bridge._model_bridge).__name__ == "LlavaOnevision2MoEBridge"
    prov = bridge.to_megatron_provider(load_weights=False)
    prov.tensor_model_parallel_size = 1
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = 1
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    exported = {n: t.detach().float().cpu() for n, t in bridge.export_hf_weights(model, cpu=True)}

    missing = [k for k in ref if k not in exported]
    extra = [k for k in exported if k not in ref]
    mismatched = [
        k for k in ref
        if k in exported and (tuple(exported[k].shape) != tuple(ref[k].shape)
                              or not torch.allclose(ref[k], exported[k], atol=2e-2, rtol=2e-2))
    ]
    assert not missing, f"HF params not produced by export: {missing[:8]}"
    assert not extra, f"export produced params absent from HF reference: {extra[:8]}"
    assert not mismatched, f"round-trip value mismatch: {mismatched[:8]}"
