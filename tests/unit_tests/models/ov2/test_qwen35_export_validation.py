"""CPU tests of actual HF meta-model shape validation and safetensors shards."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[4]
CONVERT = ROOT / "examples/models/qwen/qwen35_vl_ov2/convert"
SPEC = importlib.util.spec_from_file_location("validate_export", CONVERT / "validate_hf_export.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture
def skeleton(tmp_path):
    golden = ROOT / "examples/models/qwen/qwen3_vl_ov2/gb200/convert/hf_skeleton_fixes"
    for name in ("modeling_llava_onevision2_moe.py", "configuration_llava_onevision2_moe.py"):
        shutil.copy(golden / name, tmp_path / name)
    config = {
        "model_type": "llava_onevision2_moe",
        "auto_map": {"AutoConfig": "configuration_llava_onevision2_moe.LlavaOnevision2MoeConfig"},
        "image_token_id": 248056,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 0, 0],
            },
        },
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "spatial_merge_size": 3,
            "out_hidden_size": 32,
            "use_head": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_real_meta_model_all_inference_tensors(skeleton):
    shapes = validator.expected_shapes(skeleton)
    assert any("linear_attn" in key for key in shapes)
    assert any("self_attn" in key for key in shapes)
    assert any("shared_expert" in key for key in shapes)
    assert any("visual" in key for key in shapes)
    save_file(
        {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in shapes.items()},
        skeleton / "model.safetensors",
    )
    inventory = validator.tensor_inventory(skeleton)
    result = validator.check_shapes(inventory, shapes, require_mtp=False)
    assert result["inference_tensors_checked"] == len(shapes)
    assert result["numerical_parity_verified"] is False


@pytest.mark.parametrize("damage", ["missing", "shape", "legacy_expert", "mtp"])
def test_bad_exports_are_rejected(skeleton, damage):
    shapes = validator.expected_shapes(skeleton)
    actual = shapes.copy()
    key = next(k for k in shapes if "self_attn" in k)
    if damage == "missing":
        del actual[key]
    elif damage == "shape":
        actual[key] = (1,)
    elif damage == "legacy_expert":
        actual["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"] = (2, 2)
    with pytest.raises(ValueError):
        validator.check_shapes(actual, shapes, require_mtp=damage == "mtp")


@pytest.mark.parametrize("damage", ["missing_shard", "phantom_key", "wrong_file", "duplicate"])
def test_actual_shards_must_match_index(tmp_path, damage):
    save_file({"a": torch.zeros(2)}, tmp_path / "part1.safetensors")
    save_file({"a" if damage == "duplicate" else "b": torch.zeros(3)}, tmp_path / "part2.safetensors")
    index = {"a": "part1.safetensors", "b": "part2.safetensors"}
    if damage == "missing_shard":
        (tmp_path / "part2.safetensors").unlink()
    elif damage == "phantom_key":
        index["missing"] = "part1.safetensors"
    elif damage == "wrong_file":
        index = {"a": "part2.safetensors", "b": "part1.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    with pytest.raises((ValueError, OSError)):
        validator.tensor_inventory(tmp_path)
