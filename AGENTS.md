# AGENTS.md — Megatron Bridge

> **Project:** PyTorch-native bridge between Hugging Face and Megatron-Core.
> Bidirectional checkpoint conversion, pretraining, SFT, and LoRA recipes
> with optimized NVIDIA GPU performance. Package: `megatron.bridge` (Python 3.12).

## Skills

The `skills/` directory contains structured guides for common tasks (adding
models, running experiments, debugging multi-node jobs, performance tuning,
etc.). **Always read the relevant `SKILL.md` before starting any task it
covers — skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Boundaries

**NEVER:**
- Modify files inside `3rdparty/Megatron-LM/` by hand — the submodule tracks NVIDIA upstream and
  is un-pushable. The OV2 mcore edits ship as `3rdparty/*.patch`; change those, not the tree
- Run the full test suite — run only the specific tests relevant to your change
- Add required (non-optional) dependencies — use optional extras; submit dependency changes as a separate PR
- Commit secrets, tokens, `.env` files, or environment-specific paths / account names (e.g. `/home/yuya/…`, usernames, cluster hostnames)
- Use bare `print()` — use `logging.getLogger(__name__)` or `print_rank_0()`

**ASK FIRST:**
- Before adding any new dependency to `pyproject.toml`
- Before modifying CI workflows (`.github/workflows/`)
- Before changing public API signatures in `models/conversion/`

**ALWAYS:**
- Run `uv run pre-commit run --all-files` before committing
- Add NVIDIA copyright headers to new Python files (except under `tests/`)
- Sign off commits: `git commit -s -m "message"`
- Use `uv run python -m pytest` and `uv run python -m torch.distributed.run`, not bare `pytest` / `torchrun`
- Use the current year (2026) in generated content — do not default to 2025 or any past year

## Toolchain

| Action | Command | Config |
|--------|---------|--------|
| Install deps | `uv sync` | `pyproject.toml`, `uv.lock` |
| Install dev tools | `uv sync --group dev` | |
| Lint + format | `uv run pre-commit run --all-files` | `ruff.toml`, `.pre-commit-config.yaml` |
| Unit tests | `uv run python -m pytest tests/unit_tests/` | `pyproject.toml [tool.pytest]` |
| Distributed run | `uv run python -m torch.distributed.run --nproc_per_node=2 script.py` | |
| Type check | `uv run mypy --strict path/to/file.py` | |
| Regen lock file | `uv lock` | |
| Init submodule | `git submodule update --init` | `.gitmodules` |
| Apply OV2 mcore patches | `bash 3rdparty/apply_megatron_patch.sh` | `3rdparty/*.patch` |

## Code Style

Lint and format are enforced by pre-commit hooks (ruff). See @ruff.toml for
the authoritative rules. For judgment calls not covered by tooling, see
@skills/linting-and-formatting/SKILL.md. Key points the linter cannot catch:

- Type hints required on all public API functions (`X | None`, not `Optional[X]`)
- Google-style docstrings on public classes and functions
- Use `*` separator for functions with multiple same-type parameters
- No arbitrary defaults for config values — be explicit

## Testing

- **No foreign `setattr` on config dataclasses in tests.** When a test applies overrides to a recipe config via `setattr(config_obj, key, value)`, always guard with `if not hasattr(config_obj, key): raise ValueError(...)` first. Setting an attribute that does not exist on the dataclass silently creates a phantom field — the test passes but the recipe would fail for a real user who never sets that key. This applies to all override patterns (`model_overrides`, `checkpoint_overrides`, `config_overrides`, etc.) in `tests/functional_tests/`.

## Contributing

See @CONTRIBUTING.md for the full contributor guide, including:

- Commit and PR title format ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <description>`)
- PR labeling taxonomy (type, area, state, risk labels)
- Testing conventions (unit preferred, functional max 2 GPUs, L0/L1/L2 tiers)
- Dependency management policy
- DCO sign-off requirements
- CI triggering (`/ok to test <commit-SHA>`)

## Architecture

### Bridge Pattern

Each supported model family lives under `src/megatron/bridge/models/<family>/` with:

| File | Role |
|------|------|
| `bridge.py` | HF ↔ Megatron conversion logic |
| `config_mapping.py` | Maps HF config → Megatron config |
| `param_mapping.py` | Maps parameter names between formats |
| `hf_pretrained/` | HF model definition provider |

Recipes live in `src/megatron/bridge/recipes/<family>/`.

`AutoBridge` (`models/conversion/auto_bridge.py`) auto-selects the correct
bridge from a HF model name or path.

### Megatron-Core Submodule

`3rdparty/Megatron-LM/` is a git submodule pinned to a specific commit,
installed as an editable package via `[tool.uv.sources]` in `pyproject.toml`.
Use `scripts/switch_mcore.sh` to switch versions.

## OV2 Fork

This repo is a fork that grafts the LLaVA-OneVision-2 vision tower onto Qwen3-family LLMs.
~99% is stock upstream Megatron-Bridge; the OV2 delta is five places:

| Path | What it adds |
|------|--------------|
| `src/megatron/bridge/models/qwen_vl_ov2/` | The composite VLM: OV2.1 vision tower, m33 adapter, 3-sibling assembly + stitch-load, HF bridge, forward step |
| `src/megatron/bridge/recipes/ov2/` | Recipes, backbone path factory, energon task encoder |
| `examples/models/qwen/qwen3_vl_ov2/`, `.../qwen35_vl_ov2/` | Per-hardware launchers (A800 / B200 / GB200), checkpoint conversion & export, smoke/probe tools |
| `3rdparty/*.patch` + `apply_megatron_patch.sh` | Three mandatory mcore patches |
| `aiak_shim/`, `_verify_stubs/` | PYTHONPATH shims: AIAK `MultiMixQASample` types; stubs for config-only verification |

**After a fresh clone, run `bash 3rdparty/apply_megatron_patch.sh`** (idempotent, each patch
independently guarded). Without it the OV2 build dies with
`unexpected keyword argument 'apply_rotary_fn'`; with `FLEX_BACKEND=hybridep` and no THD-pad
patch, the first MoE dispatch dies with an async `cudaErrorIllegalAddress`.

### Model shape

`LlavaOnevision2` (`models/qwen_vl_ov2/llava_ov2.py`) is one mcore module holding three
siblings — `language_model` (Bridge-built GPTModel), `vision_model` (`OneVisionEncoderModel`,
ported from AIAK, *not* Qwen's native ViT), and `adapter` (m33 patch merger). The tower,
adapter, and step are LLM-agnostic (the adapter auto-sizes to `llm_cfg.hidden_size`), so a new
backbone changes only paths plus a dense/MoE flag. Weights load through the provider's
`pre_wrap_hook` stitch, **bypassing Bridge's torch_dist loader** — that is the fork's core trick.

### Running training

Never hand-write a `torchrun` line — use the launcher for the target hardware. It resolves
`REPO`, applies the mcore patches, exports the `OV2_*` environment, picks the precision /
EP-dispatcher lane (`ACCEL`), then execs:

```
python -m torch.distributed.run … scripts/training/run_recipe.py \
  --recipe <name> --dataset vlm-energon --step_func ov2_step
```

`load_recipe()` resolves the name via `getattr(megatron.bridge.recipes, name)` and `recipes/ov2`
is star-imported, so **a new recipe must be listed in `recipes/ov2/__init__.py`'s `__all__`** or
the launcher fails with `AttributeError`. `_ov2_common(backbone, stage)` is the single assembly
function every recipe wraps; it derives `train_iters` and the LR schedule from `OV2_*`.

Recipe names deliberately do not match their backbones — `ov2_35b_a3b_*` is Qwen3-**30B**-A3B,
not Qwen3.5. Check `_OV2_BACKBONES` in `recipes/ov2/ov2.py` before assuming.

### OV2 invariants

- **CP = 1, PP = 1.** `LlavaOnevision2.forward` asserts CP=1; tower and adapter are built on
  every rank, so PP must be 1. Scale with TP / EP.
- **Labels are pre-shifted** (roll -1) by the energon task encoder. Neither the step nor mcore's
  `compute_language_model_loss` shifts — missing it trains an identity objective *silently*.
- **`apply_rope_fusion` stays `False`.**
- **Midtrain on MoE uses AdamW** (auto-switched). Stage-2 Muon + EP8 is fine because stage-2
  freezes the experts. Muon forces `use_distributed_optimizer=False` and cannot resume from an
  AdamW checkpoint — switching optimizers needs a fresh `SAVE`.
- **30B / 35B mcore checkpoints are EP8-sharded**; build with EP=8 and prove every conversion
  lossless with `convert/verify.sh`.
- **HF processor patch/merge must match the tower** — p16m33 processor output into a patch14
  tower crashes at the `patch_embed` reshape.
- **PYTHONPATH order:** `$REPO/src` and `$REPO/3rdparty/Megatron-LM` must precede the
  container's bundled `megatron.core`, or you import unpatched mcore and fail much deeper.
- **`ckpt_format=fsdp_dtensor` (`OV2_FSDP=1`) is a one-way door** — torch_dist recipes and the
  convert tools cannot read it back.
- **`OV2_*` edits are two-sided** — launcher and recipe both read them; changing one side yields
  an inconsistent config.

### OV2 env-var convention

~90 `OV2_*` names exist across `src/` and `examples/`. New ones must follow the house rule:
**the default degrades the new logic to a no-op on the already-validated path.** `OV2_FP8_PAD_MULT`
gives `lcm(tp, 1) == tp` under bf16; `OV2_ADAPTER_INIT_SCALE` defaults to 1.0; MTP/aux zeroing
applies only to the Qwen3.5 line; unset `OV2_FLEX_BACKEND` leaves the dispatcher untouched. A 30B
run takes days — this is what lets the code evolve without invalidating verified configs.

### OV2 tests

```bash
uv run python -m pytest tests/unit_tests/models/ov2/ -v
uv run python -m pytest tests/functional_tests/models/ov2/test_ov2_conversion.py -v
```

Both need a **visible GPU**, including the config-only unit tests: importing `ov2_bridge` pulls
`llava_ov2` → `fla`/`triton`, which initializes a Triton driver at import time. The functional
test additionally skips without a local composite HF skeleton.

### Git history

Work on `main`. The repo was **re-rooted as an orphan commit on 2026-06-14**, so `main` and the
archived `gb200` branch share no merge-base: `git log` on `main` cannot reach past that date, and
cherry-picking between the two is impossible. `gb200` is the only branch holding full NVIDIA
upstream history — switch to it read-only for `git log -S` archaeology.

## Tool Compatibility

This file is the single source of agent instructions. For Claude Code
compatibility, create a symlink: `ln -s AGENTS.md CLAUDE.md`
