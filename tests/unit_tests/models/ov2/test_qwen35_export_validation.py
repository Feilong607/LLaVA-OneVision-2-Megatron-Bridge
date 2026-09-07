"""CPU tests of actual HF meta-model shape validation and safetensors shards."""

import argparse
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


@pytest.mark.unit
@pytest.mark.parametrize("legacy_architectures", [None, ["LlavaOnevision2ForConditionalGeneration"]])
def test_builder_replaces_lossy_base_config_and_preserves_dispatch(skeleton, legacy_architectures):
    """A valid JSON architecture must survive AutoConfig and config synthesis, not only text dispatch."""
    from transformers import AutoConfig

    spec = importlib.util.spec_from_file_location("q35_skeleton_builder", CONVERT / "build_qwen35_hf_skeleton.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    config_path = skeleton / "config.json"
    config = json.loads(config_path.read_text())
    text = skeleton / "text"
    text.mkdir()
    (text / "config.json").write_text(json.dumps(config["text_config"]))
    # Tokenizer contents are outside this config-dispatch regression.
    (text / "tokenizer.json").write_text("{}")
    config["architectures"] = legacy_architectures
    config_path.write_text(json.dumps(config))
    # Representative legacy constructor: correct Qwen3.5 text type, but loses HF dispatch metadata.
    config_file = skeleton / "configuration_llava_onevision2_moe.py"
    original = config_file.read_text()
    config_file.write_text(
        original.replace(
            "super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)",
            "kwargs.pop('architectures', None)\n        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)",
        )
    )
    before = AutoConfig.from_pretrained(skeleton, trust_remote_code=True, local_files_only=True)
    assert before.text_config.model_type == "qwen3_5_moe_text"
    assert before.architectures is None  # The old text-only selftest would pass this broken config.
    with pytest.raises(SystemExit):
        builder.selftest(str(skeleton), 248056, 3, False)
    proc = skeleton / "processor"
    proc.mkdir()
    (proc / "preprocessor_config.json").write_text("{}")
    out = skeleton / "built"
    args = argparse.Namespace(
        text_dir=str(text),
        base_skeleton=str(skeleton),
        proc_dir=str(proc),
        tokenizer_dir=None,
        out=str(out),
        patch_size=16,
        merge_size=3,
    )
    builder.build(args)
    loaded = AutoConfig.from_pretrained(out, trust_remote_code=True, local_files_only=True)
    assert loaded.architectures == ["LlavaOnevision2ForConditionalGeneration"]
    assert loaded.text_config.to_dict() == before.text_config.to_dict()
    assert loaded.image_token_id == 248056
    assert loaded.tie_word_embeddings is False
    assert loaded.vision_config.use_post_layernorm is True
    assert loaded.vision_config.post_layernorm_eps == 1e-5
    assert loaded.vision_config.pre_layernorm_eps == 1e-4
    assert loaded.vision_config.layer_norm_eps == 1e-5
    assert loaded.vision_config.merger_layernorm_eps == loaded.text_config.rms_norm_eps
    assert loaded.vision_config.zero_centered_gamma is True
    assert loaded.vision_config.merger_zero_centered_gamma is True
    synthesized = type(loaded)(**loaded.to_dict())
    assert synthesized.architectures == loaded.architectures
    assert synthesized.text_config.model_type == "qwen3_5_moe_text"
    builder.selftest(str(out), 248056, 3, False)


@pytest.mark.unit
@pytest.mark.parametrize("zero_centered", [False, True])
def test_trained_vision_final_norm_is_exported_and_applied_before_merger(skeleton, zero_centered):
    """Exercise actual HF vision forward around the final norm using fixed encoder features."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_outputs import BaseModelOutput

    config_path = skeleton / "config.json"
    config = json.loads(config_path.read_text())
    baseline_shapes = validator.expected_shapes(skeleton)
    assert not any("layernorm_post" in key for key in baseline_shapes)
    config["vision_config"].update(
        use_post_layernorm=True,
        post_layernorm_eps=1e-5,
        pre_layernorm_eps=1e-4,
        layer_norm_eps=1e-5,
        merger_layernorm_eps=1e-6,
        zero_centered_gamma=zero_centered,
        merger_zero_centered_gamma=zero_centered,
    )
    config_path.write_text(json.dumps(config))
    expected = validator.expected_shapes(skeleton)
    assert set(expected) - set(baseline_shapes) == {
        "model.visual.layernorm_post.weight",
        "model.visual.layernorm_post.bias",
    }
    cfg = AutoConfig.from_pretrained(skeleton, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(
        "modeling_llava_onevision2_moe.LlavaOnevision2VisionPretrainedModel", str(skeleton), local_files_only=True
    )
    model = cls(cfg.vision_config).eval()
    assert model.head is None
    assert model.layernorm_post.eps == 1e-5
    assert model.layernorm_pre.eps == 1e-4
    assert model.merger.ln_q.eps == 1e-6
    features = torch.randn(1, 9, 32) * 0.01 + 0.2

    class FixedEncoder(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return BaseModelOutput(last_hidden_state=features, hidden_states=(features, features), attentions=None)

    encoder_norms = [model.encoder.layers[0].layer_norm1, model.encoder.layers[0].layer_norm2]
    model.encoder = FixedEncoder()
    with torch.no_grad():
        model.layernorm_post.weight.fill_(1.7)
        model.layernorm_post.bias.fill_(0.3)
    captured = []
    handle = model.merger.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    positions = torch.tensor([[0, h, w] for h in range(3) for w in range(3)])
    with torch.no_grad():
        model(torch.randn(9, 3, 16, 16), grid_thw=torch.tensor([[1, 3, 3]]), patch_positions=positions)
    handle.remove()
    reference = torch.nn.functional.layer_norm(
        features, (32,), model.layernorm_post.weight + int(zero_centered), model.layernorm_post.bias, 1e-5
    )
    torch.testing.assert_close(captured[0], reference, rtol=0, atol=0)
    assert not torch.equal(captured[0], features)
    for norm in (model.layernorm_pre, *encoder_norms, model.merger.ln_q):
        with torch.no_grad():
            norm.weight.fill_(0.25)
            norm.bias.fill_(0.1)
        expected_norm = torch.nn.functional.layer_norm(
            features, (32,), norm.weight + int(zero_centered), norm.bias, norm.eps
        )
        torch.testing.assert_close(norm(features), expected_norm, rtol=0, atol=0)
    # The strict output validator must now reject an export missing either trained norm tensor.
    inventory = {key: shape for key, shape in expected.items() if "layernorm_post" not in key}
    with pytest.raises(ValueError):
        validator.check_shapes(inventory, expected, require_mtp=False)


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
