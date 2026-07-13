#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Idempotent patcher: enable FP8/MXFP8 (ACCEL=1) packed-sequence padding in ov2_step.py.

WHAT IT DOES
    Pads the packed token dim to ``lcm(TP, 32)`` -- but ONLY when the run uses fp8/fp4
    (ACCEL=1, i.e. ``mixed_precision.fp8`` is set, e.g. bf16_with_mxfp8_mixed). bf16 runs
    (ACCEL=0 baseline / ACCEL=2 HybridEP) are BYTE-IDENTICAL to before: the helper returns
    1, so ``lcm(TP, 1) == TP`` and the existing TP-only pad behavior is unchanged.

WHY 32 (not 16)
    TE's crash ``assert_dim_for_fp8_exec`` only needs the token dim %8 (last dim %16), which
    16 would satisfy. But MXFP8 (the ACCEL=1 recipe) uses 1x32 block scaling, and in the
    weight-gradient GEMM the token dim is the contraction dim -> ``MXFP8Quantizer.is_quantizable``
    requires the token dim %32, or the GEMM SILENTLY falls back to bf16 for that step. 32 stops
    the crash AND keeps MXFP8 engaged; it is a strict superset of the %8/%16 crash-fix and of the
    TP/SP multiple (for TP<=32).

    The pad tail is masked (labels=-100, loss_mask=0) and folded into the existing THD
    ``cu_last < seq_len`` branch, so the loss is identical to the unpadded sequence.

RUNTIME OVERRIDES (no code change needed)
    OV2_FP8_PAD_MULT=<int>   change the fp8 alignment multiple (default 32)
    OV2_FP8_SEQ_PAD=1|0      force the fp8 pad on/off regardless of the resolved precision

USAGE
    python apply_ov2_fp8_seqpad.py [/path/to/repo_root_or_ov2_step.py]
    # no arg -> auto-detect the repo root from this script's location, else from CWD.

    Safe to run repeatedly (idempotent). Verifies the expected baseline before editing and
    fails loud if ov2_step.py has diverged (so it never blind-patches a changed file).
"""

import os
import sys


REL_PATH = "src/megatron/bridge/models/qwen_vl_ov2/ov2_step.py"

# --- anchors (must match the current ov2_step.py exactly) --------------------------------
IMPORT_ANCHOR = "import logging\nimport os\n"
IMPORT_REPLACEMENT = "import logging\nimport math\nimport os\n"

HELPER_MARKER = "_ov2_fp8_pad_mult"  # idempotency sentinel
FORWARD_ANCHOR = "\n\ndef forward_step(\n"

OLD_BLOCK = (
    "    if tokens is not None:\n"
    "        from megatron.core import parallel_state\n"
    "        _tp = parallel_state.get_tensor_model_parallel_world_size()\n"
    "        if _tp > 1 and tokens.shape[1] % _tp:\n"
    "            _pad = _tp - tokens.shape[1] % _tp\n"
    "            tokens = torch.nn.functional.pad(tokens, (0, _pad), value=0)\n"
    "            if labels is not None:    labels = torch.nn.functional.pad(labels, (0, _pad), value=-100)\n"
    "            if loss_mask is not None: loss_mask = torch.nn.functional.pad(loss_mask, (0, _pad), value=0)\n"
)

NEW_BLOCK = (
    "    if tokens is not None:\n"
    "        from megatron.core import parallel_state\n"
    "        _tp = parallel_state.get_tensor_model_parallel_world_size()\n"
    "        # Base alignment: SP scatter needs a TP multiple. FP8/MXFP8 (ACCEL=1) ALSO needs the packed\n"
    "        # token dim aligned or TE's fp8 GEMMs fail assert_dim_for_fp8_exec (crash); and for MXFP8's\n"
    "        # 1x32 block scaling the wgrad GEMM (token dim = contraction) silently drops to bf16 unless\n"
    "        # the token dim is a multiple of 32. _ov2_fp8_pad_mult(state) returns 1 for bf16 (ACCEL=0/2)\n"
    "        # -> lcm(_tp, 1) == _tp -> byte-identical to the old TP-only pad; 32 for fp8/fp4 (ACCEL=1).\n"
    "        _pad_mult = math.lcm(_tp, _ov2_fp8_pad_mult(state))\n"
    "        if _pad_mult > 1 and tokens.shape[1] % _pad_mult:\n"
    "            _pad = _pad_mult - tokens.shape[1] % _pad_mult\n"
    "            tokens = torch.nn.functional.pad(tokens, (0, _pad), value=0)\n"
    "            if labels is not None:    labels = torch.nn.functional.pad(labels, (0, _pad), value=-100)\n"
    "            if loss_mask is not None: loss_mask = torch.nn.functional.pad(loss_mask, (0, _pad), value=0)\n"
)

HELPER_DEF = (
    "def _ov2_fp8_pad_mult(state) -> int:\n"
    '    """Token-dim alignment the packed sequence needs so TE fp8/MXFP8 GEMMs stay engaged.\n'
    "\n"
    "    Returns 1 for bf16 runs (ACCEL=0/2) so the pad block is byte-identical to the TP-only pad.\n"
    "    Returns 32 (override: OV2_FP8_PAD_MULT) for fp8/fp4 runs (ACCEL=1, mixed_precision.fp8 set):\n"
    "    the TE crash assert_dim_for_fp8_exec only needs the token dim %8 (last dim %16), but MXFP8 uses\n"
    "    1x32 block scaling and in the weight-gradient GEMM the token dim is the contraction dim, so\n"
    "    MXFP8Quantizer.is_quantizable requires the token dim %32 or the GEMM silently falls back to bf16.\n"
    "    OV2_FP8_SEQ_PAD=1/0 force-enables/disables regardless of the resolved precision.\n"
    '    """\n'
    '    _ovr = os.environ.get("OV2_FP8_SEQ_PAD")\n'
    "    if _ovr is not None:\n"
    '        _on = _ovr == "1"\n'
    "    else:\n"
    '        mp = getattr(getattr(state, "cfg", None), "mixed_precision", None)\n'
    '        _lp = getattr(mp, "fp8", None) or getattr(mp, "fp4", None)  # resolved MixedPrecisionConfig\n'
    "        if _lp is not None:\n"
    "            _on = bool(_lp)\n"
    "        elif isinstance(mp, str):  # defensive: pre-resolution registry key (e.g. MIMO path)\n"
    '            _on = ("fp8" in mp) or ("fp4" in mp)\n'
    "        else:\n"
    "            _on = False\n"
    "    if not _on:\n"
    "        return 1\n"
    "    try:\n"
    '        return max(1, int(os.environ.get("OV2_FP8_PAD_MULT", "32")))\n'
    "    except ValueError:\n"
    "        return 32\n"
    "\n"
    "\n"
)


def _resolve_target(argv):
    if len(argv) > 1:
        p = os.path.abspath(argv[1])
        if os.path.isfile(p):
            return p
        cand = os.path.join(p, REL_PATH)
        if os.path.isfile(cand):
            return cand
        sys.exit(f"FATAL: no ov2_step.py at '{argv[1]}' (tried it as a file and as repo/{REL_PATH}).")
    # auto-detect: walk up from this script, then from CWD, looking for src/megatron/bridge.
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        d = start
        while d != os.path.dirname(d):
            cand = os.path.join(d, REL_PATH)
            if os.path.isfile(cand):
                return cand
            d = os.path.dirname(d)
    sys.exit(
        f"FATAL: could not auto-detect the repo root (no {REL_PATH} above this script or CWD). "
        f"Pass the repo root or the ov2_step.py path explicitly."
    )


def main():
    """Locate ov2_step.py, verify the expected baseline, and apply the FP8 seq-pad patch idempotently."""
    target = _resolve_target(sys.argv)
    with open(target, "r", encoding="utf-8") as f:
        src = f.read()

    if HELPER_MARKER in src:
        print(
            f"[apply_ov2_fp8_seqpad] already patched (found '{HELPER_MARKER}'): {target}\n"
            f"  no changes made -- this script is idempotent."
        )
        return

    # Pre-flight: the exact baseline block must be present, or we refuse to touch the file.
    problems = []
    if OLD_BLOCK not in src:
        problems.append("the expected baseline TP-pad block was not found")
    if IMPORT_ANCHOR not in src:
        problems.append("the 'import logging / import os' header was not found")
    if FORWARD_ANCHOR not in src:
        problems.append("the 'def forward_step(' anchor was not found")
    if problems:
        sys.exit(
            "FATAL: ov2_step.py has diverged from the expected baseline; refusing to patch.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n  Re-generate this patcher against the current file, or apply the change by hand."
        )

    patched = src
    # 1) add `import math` (only if missing)
    if "\nimport math\n" not in patched:
        patched = patched.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    # 2) insert the helper immediately before forward_step
    patched = patched.replace(FORWARD_ANCHOR, "\n\n" + HELPER_DEF + "def forward_step(\n", 1)
    # 3) swap the pad block
    patched = patched.replace(OLD_BLOCK, NEW_BLOCK, 1)

    if patched == src:
        sys.exit("FATAL: no substitutions applied (unexpected). File left untouched.")

    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)

    print(
        f"[apply_ov2_fp8_seqpad] PATCHED {target}\n"
        f"  + import math\n"
        f"  + _ov2_fp8_pad_mult(state) helper (returns 1 for bf16, 32 for fp8/fp4)\n"
        f"  ~ pad block now uses lcm(TP, _ov2_fp8_pad_mult(state))\n"
        f"  ACCEL=0/2 (bf16) unchanged; ACCEL=1 (MXFP8) now pads the token dim to a multiple of 32.\n"
        f"  overrides: OV2_FP8_PAD_MULT=<int> (default 32), OV2_FP8_SEQ_PAD=1|0 (force on/off)."
    )


if __name__ == "__main__":
    main()
