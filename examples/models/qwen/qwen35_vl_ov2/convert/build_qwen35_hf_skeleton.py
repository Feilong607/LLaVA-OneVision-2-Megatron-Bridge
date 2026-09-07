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
"""Build the Qwen3.5-35B-A3B OV2 composite HF *skeleton* (CFG dir) that the HF export dispatches on.

Why this exists: ``run_export_hf.sh`` / ``convert.sh export`` synthesise the exported ``config.json`` from the
CFG skeleton (AutoBridge.from_auto_config conforms the mcore-derived config to the skeleton's dict), and copy
the skeleton's tokenizer / processor / remote-code files next to the weights. Every default in that chain
points at the 30B skeleton (``/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model``): exporting a Qwen3.5 SAVE with
it yields a config whose text_config is qwen3_moe (48L / 128 experts / vocab 151936) with image_token_id
151655 -- HF would instantiate Qwen3MoeModel over Qwen3.5 weights and load 30/40 layers' mixers as
"missing" (random), warning only. No Qwen3.5 composite skeleton exists anywhere on the cluster: the
hand-assembled ``~/qwen35_p16m33_auto_model`` is "30B image half + Qwen3.5 tokenizer half" (processor use
only; its config.json is still the 30B composite).

What it produces (CPU only, seconds, no weights):
  <out>/config.json                 composite: model_type llava_onevision2_moe, architectures set, auto_map from
                                    the base skeleton, text_config = the Qwen3.5-35B-A3B TEXT config verbatim
                                    (qwen3_5_moe_text, 40L, 256 experts, layer_types, linear_* GDN dims,
                                    rope_parameters{mrope_section [11,11,10], mrope_interleaved, partial 0.25},
                                    vocab 248320, mtp_num_hidden_layers), vision_config = the base p16m33 tower
                                    (patch 16 / merge 3 / out_hidden_size == text hidden_size asserted,
                                    use_patch_position_encoding false), Qwen3.5 multimodal token ids read from
                                    the tokenizer (image_pad 248056 / video_pad 248057 / vision_start 248053 /
                                    vision_end 248054), tie_word_embeddings false.
  <out>/configuration_*.py          the golden composite configuration class: preserves top-level architectures
                                    and deserializes text_config as a real Qwen3_5MoeTextConfig. The base
                                    skeleton supplies vision geometry, not the Qwen3.5 dispatch implementation.
  <out>/modeling_*.py               the golden hf_skeleton_fixes modeling (M-RoPE aware).
  <out>/chat_template.jinja         the golden OV2 template (Qwen2/3-VL style, 'You are a helpful assistant.',
                                    NO <think>): the Qwen3.5-native template would prepend '<think>\\n' to every
                                    assistant turn and drop the system line the OV2 encoder trained with.
  <out>/tokenizer*, vocab/merges    Qwen3.5 tokenizer (from --tokenizer-dir, default --text-dir), with any
                                    embedded chat_template stripped from tokenizer_config.json.
  <out>/*preprocessor_config.json   from --proc-dir (patch 16 / merge 3 / temporal 1 enforced).
  <out>/generation_config.json      Qwen3.5 eos (<|im_end|>, <|endoftext|>) / pad (<|endoftext|>), greedy.

Then ``convert.sh fixup`` re-applies the same contract on the exported dir (idempotent), so the skeleton and
the export cannot drift apart.

Usage (cluster, code-sync or export workspace, no GPU):
  python3 build_qwen35_hf_skeleton.py \\
      --text-dir /home/ftan0055/Qwen3.5-35B-A3B-text \\
      --base-skeleton /datasets/llava-ov2-30b-a3b-m9lvdn/auto_model \\
      --proc-dir /home/ftan0055/qwen35_p16m33_auto_model \\
      --out /home/ftan0055/qwen35_hf_skeleton
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
# Golden fixes live with the 30B export tooling (one copy for every OV2 line).
GOLDEN_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "qwen3_vl_ov2", "gb200", "convert", "hf_skeleton_fixes"))
MODELING = "modeling_llava_onevision2_moe.py"
CONFIGURATION = "configuration_llava_onevision2_moe.py"
CHAT_TEMPLATE = "chat_template.jinja"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
PREPROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json")
SPECIAL_TOKENS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}
# Qwen3.5 fallbacks (used only when tokenizer.json is missing; a warning is printed).
QWEN35_IDS = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "<|im_end|>": 248046,
    "<|endoftext|>": 248044,
}
REQUIRED_VISION_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "patch_size",
    "spatial_merge_size",
    "out_hidden_size",
)


def _die(msg: str) -> None:
    print(f"[q35-skeleton] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _log(msg: str) -> None:
    print(f"[q35-skeleton] {msg}", flush=True)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _token_ids_from_tokenizer_json(tok_dir: str) -> dict[str, int]:
    """content -> id for the added (special) tokens in tokenizer.json ({} when the file is absent)."""
    p = os.path.join(tok_dir, "tokenizer.json")
    if not os.path.isfile(p):
        return {}
    tj = _load_json(p)
    return {t["content"]: int(t["id"]) for t in tj.get("added_tokens", []) if "content" in t and "id" in t}


def _copy(src: str, dst_dir: str, force: bool = True) -> bool:
    if not os.path.isfile(src):
        return False
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst) and not force:
        return False
    shutil.copy2(src, dst)
    return True


def build(args: argparse.Namespace) -> str:
    """Write the qwen35 composite skeleton into args.out and return that path."""
    text_dir, base, proc_dir, out = args.text_dir, args.base_skeleton, args.proc_dir, args.out
    tok_dir = args.tokenizer_dir or text_dir
    for d, what in (
        (text_dir, "--text-dir"),
        (base, "--base-skeleton"),
        (proc_dir, "--proc-dir"),
        (tok_dir, "--tokenizer-dir"),
    ):
        os.path.isdir(d) or _die(f"{what} is not a directory: {d}")
    for f in (MODELING, CONFIGURATION, CHAT_TEMPLATE):
        os.path.isfile(os.path.join(GOLDEN_DIR, f)) or _die(f"golden file missing: {GOLDEN_DIR}/{f}")
    os.makedirs(out, exist_ok=True)

    base_cfg = _load_json(os.path.join(base, "config.json"))
    text_cfg = _load_json(os.path.join(text_dir, "config.json"))

    # ---- text_config: the Qwen3.5 text model config, verbatim (that dir is what the recipe trains from) ----
    tmt = str(text_cfg.get("model_type", ""))
    tmt.startswith("qwen3_5") or _die(
        f"{text_dir}/config.json model_type={tmt!r}; expected qwen3_5_moe_text (run tools/extract_qwen35_text.py first)"
    )
    text_config = {
        k: v
        for k, v in text_cfg.items()
        if k not in ("architectures", "_name_or_path", "auto_map", "transformers_version")
    }
    text_hidden = int(text_config["hidden_size"])
    rp = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
    rp.get("mrope_section") or _die(
        "text_config has no rope_parameters.mrope_section; the Qwen3.5 line trains with interleaved mrope"
    )

    # ---- vision_config: the base p16m33 tower; geometry asserted, pos-enc forced off (ckpt has no pos_emb) ----
    vision_config = dict(base_cfg.get("vision_config") or {})
    missing = [k for k in REQUIRED_VISION_KEYS if k not in vision_config]
    not missing or _die(f"base skeleton vision_config lacks {missing}: {base}/config.json")
    if (
        int(vision_config["patch_size"]) != args.patch_size
        or int(vision_config["spatial_merge_size"]) != args.merge_size
    ):
        _log(
            f"WARN: base vision_config patch/merge = {vision_config['patch_size']}/{vision_config['spatial_merge_size']}; "
            f"forcing {args.patch_size}/{args.merge_size} (the qwen35 line is p{args.patch_size}m{args.merge_size}{args.merge_size})"
        )
        vision_config["patch_size"], vision_config["spatial_merge_size"] = args.patch_size, args.merge_size
    int(vision_config["out_hidden_size"]) == text_hidden or _die(
        f"vision_config.out_hidden_size={vision_config['out_hidden_size']} != text hidden_size={text_hidden}: the adapter "
        f"projects 1024*merge^2 -> text hidden, so this skeleton cannot host the Qwen3.5 weights"
    )
    vision_config["use_patch_position_encoding"] = False
    # Existing Qwen3.5 OV2 checkpoints inherit non-None mtp_num_layers into the
    # vision TransformerBlock, which applies its final LN despite post_process=False.
    # Preserve that trained operation independently of the unused vision pooling head.
    vision_config["use_post_layernorm"] = True
    vision_config["post_layernorm_eps"] = 1e-5  # get_vision_config().layernorm_epsilon
    vision_config["layer_norm_type"] = "layer_norm"
    vision_config["layer_norm_eps"] = 1e-5
    vision_config["pre_layernorm_eps"] = 1e-4  # OneVisionEncoderModel.pre_layernorm
    vision_config["merger_layernorm_eps"] = text_config.get("rms_norm_eps", 1e-6)
    # Both vision and adapter deepcopy the Qwen3.5 LLM's zero-centered gamma flag.
    # Keep raw checkpoint weights; the HF norm applies (1 + gamma) at runtime.
    vision_config["zero_centered_gamma"] = True
    vision_config["merger_zero_centered_gamma"] = True
    vision_config.setdefault("temporal_patch_size", 1)

    # ---- multimodal token ids from the Qwen3.5 tokenizer ----
    ids = _token_ids_from_tokenizer_json(tok_dir)
    tok_ids = {}
    for key, tok in SPECIAL_TOKENS.items():
        if tok in ids:
            tok_ids[key] = ids[tok]
        else:
            _log(
                f"WARN: {tok} not in {tok_dir}/tokenizer.json added_tokens; using the Qwen3.5 constant {QWEN35_IDS[key]}"
            )
            tok_ids[key] = QWEN35_IDS[key]
    tok_ids["image_token_id"] == 248056 or _log(
        f"WARN: image_token_id resolved to {tok_ids['image_token_id']} (Qwen3.5 <|image_pad|> is 248056)"
    )

    # ---- composite config ----
    cfg = {
        k: v
        for k, v in base_cfg.items()
        if k not in ("text_config", "vision_config", "_name_or_path", "transformers_version")
    }
    cfg.get("auto_map") or _die(
        f"base skeleton {base}/config.json has no auto_map (not a dispatch-ready remote-code skeleton)"
    )
    cfg["model_type"] = "llava_onevision2_moe"
    cfg["architectures"] = ["LlavaOnevision2ForConditionalGeneration"]
    cfg["auto_map"] = dict(cfg["auto_map"])
    cfg["auto_map"]["AutoConfig"] = "configuration_llava_onevision2_moe.LlavaOnevision2MoeConfig"
    cfg["auto_map"]["AutoModelForCausalLM"] = "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration"
    cfg["text_config"] = text_config
    cfg["vision_config"] = vision_config
    cfg.update(tok_ids)
    cfg["tie_word_embeddings"] = False
    cfg["torch_dtype"] = text_config.get("dtype", text_config.get("torch_dtype", "bfloat16"))
    cfg.setdefault("router_aux_loss_coef", text_config.get("router_aux_loss_coef", 0.001))
    _dump_json(cfg, os.path.join(out, "config.json"))
    _log(
        f"config.json: text={tmt} {text_config.get('num_hidden_layers')}L/{text_config.get('num_experts')}E hidden={text_hidden} "
        f"vocab={text_config.get('vocab_size')} mtp={text_config.get('mtp_num_hidden_layers')} mrope={rp.get('mrope_section')} | "
        f"vision p{vision_config['patch_size']}m{vision_config['spatial_merge_size']} out={vision_config['out_hidden_size']} | "
        f"ids={tok_ids} tie=false"
    )

    # ---- Use the known composite class; legacy base classes can discard architectures even
    # when their text_config dispatch works. JSON-only checks cannot detect that failure. ----
    for py in glob.glob(os.path.join(base, "*.py")):
        _copy(py, out)
    _copy(os.path.join(GOLDEN_DIR, CONFIGURATION), out)
    _copy(os.path.join(GOLDEN_DIR, MODELING), out)
    _copy(os.path.join(GOLDEN_DIR, CHAT_TEMPLATE), out)
    for stale in ("chat_template.json",):
        p = os.path.join(out, stale)
        os.path.exists(p) and os.remove(p)

    # ---- tokenizer (Qwen3.5) ----
    copied = [f for f in TOKENIZER_FILES if _copy(os.path.join(tok_dir, f), out)]
    "tokenizer.json" in copied or _die(
        f"{tok_dir}/tokenizer.json missing; the export dir must ship the Qwen3.5 tokenizer"
    )
    tc_path = os.path.join(out, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        tc = _load_json(tc_path)
        if "chat_template" in tc:
            tc.pop("chat_template")
            _dump_json(tc, tc_path)
            _log(
                "tokenizer_config.json: stripped the embedded Qwen3.5 chat_template (golden chat_template.jinja is authoritative)"
            )

    # ---- processor jsons (p16m33) ----
    got = [f for f in PREPROCESSOR_FILES if _copy(os.path.join(proc_dir, f), out)]
    "preprocessor_config.json" in got or _die(f"{proc_dir}/preprocessor_config.json missing")
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        p = os.path.join(out, name)
        if not os.path.isfile(p):
            continue
        j = _load_json(p)
        before = (j.get("patch_size"), j.get("merge_size"), j.get("temporal_patch_size"))
        want = (
            int(vision_config["patch_size"]),
            int(vision_config["spatial_merge_size"]),
            int(vision_config["temporal_patch_size"]),
        )
        if before != want:
            j["patch_size"], j["merge_size"], j["temporal_patch_size"] = want
            _dump_json(j, p)
            _log(f"{name}: (patch,merge,temporal) {before} -> {want}")
    "video_preprocessor_config.json" in got or _log(
        "WARN: no video_preprocessor_config.json in --proc-dir; transformers 5.x Qwen2_5_VLProcessor wants one (copy the 30B skeleton's)"
    )

    # ---- generation_config.json: Qwen3.5 eos/pad ----
    im_end = ids.get("<|im_end|>", QWEN35_IDS["<|im_end|>"])
    eot = ids.get("<|endoftext|>", QWEN35_IDS["<|endoftext|>"])
    _dump_json(
        {
            "bos_token_id": None,
            "eos_token_id": [im_end, eot],
            "pad_token_id": eot,
            "do_sample": False,
            "transformers_version": "5.7.0",
        },
        os.path.join(out, "generation_config.json"),
    )
    return out


def selftest(out: str, expect_image_token: int, merge_size: int, install_fallback: bool) -> None:
    """Prove the skeleton deserializes as a Qwen3.5 composite; install the vendored configuration class if not."""
    try:
        import transformers  # noqa: F401
        from transformers import AutoConfig
    except Exception as e:  # pragma: no cover - environment without transformers
        _log(
            f"WARN: transformers not importable ({e}); skipping the dispatch self-test. Run convert.sh fixup <dir> in the export image."
        )
        return

    def _check() -> tuple[bool, str]:
        cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
        tc = cfg.text_config
        name = type(tc).__name__
        mt = getattr(tc, "model_type", None)
        layer_types = getattr(tc, "layer_types", None)
        architectures = getattr(cfg, "architectures", None)
        ok = (
            str(mt).startswith("qwen3_5")
            and name.startswith("Qwen3_5")
            and bool(layer_types)
            and architectures == ["LlavaOnevision2ForConditionalGeneration"]
        )
        return (
            ok,
            f"text_config -> {name} model_type={mt} layer_types={'ok' if layer_types else 'MISSING'}; architectures={architectures}; image_token_id={getattr(cfg, 'image_token_id', None)} tie={getattr(cfg, 'tie_word_embeddings', None)}",
        )

    ok, desc = _check()
    _log(f"dispatch self-test (skeleton's own configuration class): {'PASS' if ok else 'FAIL'} -- {desc}")
    if not ok:
        install_fallback or _die(
            "the configuration class loses Qwen3.5 text dispatch or composite architectures; rerun without --no-fallback"
        )
        shutil.copy2(os.path.join(GOLDEN_DIR, CONFIGURATION), os.path.join(out, CONFIGURATION))
        # transformers caches remote modules under HF_HOME/modules; a fresh import needs a fresh cache key,
        # so bust it by making the installed file differ (it does) and re-running the check in-process may
        # still see the old module -> re-check in a subprocess.
        import subprocess

        code = (
            "from transformers import AutoConfig;import sys;c=AutoConfig.from_pretrained(sys.argv[1],trust_remote_code=True);"
            "assert c.architectures==['LlavaOnevision2ForConditionalGeneration'], c.architectures;"
            "t=c.text_config;print(type(t).__name__, t.model_type, bool(getattr(t,'layer_types',None)))"
        )
        res = subprocess.run([sys.executable, "-c", code, out], capture_output=True, text=True)
        _log(f"installed vendored {CONFIGURATION}; re-check: {res.stdout.strip() or res.stderr.strip()[-400:]}")
        (res.returncode == 0 and "Qwen3_5" in res.stdout and "qwen3_5" in res.stdout and "True" in res.stdout) or _die(
            "vendored configuration class still fails to deserialize text_config as Qwen3_5MoeTextConfig"
        )

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
        got = tok.convert_tokens_to_ids("<|image_pad|>")
        got == expect_image_token or _die(
            f"tokenizer <|image_pad|>={got} != config image_token_id={expect_image_token}"
        )
        _log(f"tokenizer: <|image_pad|>={got} eos={tok.eos_token!r} pad={tok.pad_token!r} vocab={len(tok)} -- OK")
    except SystemExit:
        raise
    except Exception as e:
        _log(f"WARN: tokenizer self-test skipped/failed: {e}")
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(out, trust_remote_code=True)
        ms = getattr(getattr(proc, "image_processor", None), "merge_size", None)
        _log(
            f"processor: {type(proc).__name__} image_processor.merge_size={ms} -- {'OK' if ms == merge_size else 'WARN (expected %d)' % merge_size}"
        )
    except Exception as e:
        _log(f"WARN: processor self-test skipped/failed: {e}")


def main() -> None:
    """CLI entry point: build the skeleton, then (unless --no-selftest) prove it dispatches as Qwen3.5."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--text-dir",
        required=True,
        help="Qwen3.5-35B-A3B TEXT dir (extract_qwen35_text.py output): config.json + tokenizer",
    )
    ap.add_argument(
        "--base-skeleton",
        required=True,
        help="30B composite auto_model (dispatch-ready: config.json with auto_map + remote-code .py)",
    )
    ap.add_argument(
        "--proc-dir",
        required=True,
        help="p16m33 processor dir (preprocessor_config.json [+ video_preprocessor_config.json])",
    )
    ap.add_argument("--out", required=True, help="output skeleton dir (pass as CFG= to run_export_hf.sh)")
    ap.add_argument("--tokenizer-dir", default=None, help="Qwen3.5 tokenizer dir (default: --text-dir)")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--merge-size", type=int, default=3)
    ap.add_argument(
        "--no-selftest", action="store_true", help="skip the transformers dispatch/tokenizer/processor checks"
    )
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        help="do not install the vendored configuration class when the base one fails",
    )
    args = ap.parse_args()

    out = build(args)
    cfg = _load_json(os.path.join(out, "config.json"))
    if not args.no_selftest:
        selftest(out, int(cfg["image_token_id"]), args.merge_size, install_fallback=not args.no_fallback)
    _log(f"DONE: {out}  (use: CFG={out} ... bash run_export_hf.sh <SAVE>/iter_XXXXXXX)")


if __name__ == "__main__":
    main()
