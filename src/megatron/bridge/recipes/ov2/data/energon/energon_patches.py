# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for the installed megatron-energon (a pip package in the image, not a submodule).

drop_yielded -- stop every blend component from pinning its last decoded sample
================================================================================
energon builds one generator chain per blend component (RepeatDataset -> MapDataset(decode) -> shard
loader) and BlendDataset keeps all of them suspended between draws. ``RepeatDataset.__iter__`` does
``for sample in self.dataset: ... yield sample`` and ``MapDataset.__iter__`` does
``mapped_sample = self.map_fn(sample); yield add_sample_restore_key(mapped_sample, ...)``: after the
yield, both suspended frames keep ``sample`` / ``mapped_sample`` bound until that component is drawn
again. With a 127-way blend of 64k video packs (a decoded pack is 200-400 MB of PIL frames) every
rarely-drawn component pins one pack for good, so a dataloader worker's host memory climbs to
~components x pack size (x glibc fragmentation) and then plateaus -- 900 GB per pod on the merged48
production runs (2026-09-14/15), where the 49-component Qwen3 stage-3 blend stayed under 1 TB.

The patch re-compiles the two ``__iter__`` methods from the INSTALLED source so that the yielded sample is
boxed in a one-element list and yielded via ``box.pop()``: while the generator sits suspended at the yield,
its frame holds only the empty box. (A ``del`` after the yield would only run on the component's next
draw, i.e. never for a rare one.) The raw undecoded sample dict (JPEG bytes, ~15 MB per pack) stays
bound in MapDataset's frame and the failure-handler context; that residue is ~2 GB per worker for 127
components and is accepted. Compiled in the original module namespace. Anchors must match exactly once,
otherwise nothing is changed and a warning is logged -- zero drift from the image's energon logic.
Applied in the main process before the datasets are built; dataloader workers are forked afterwards
and inherit the patched classes. Opt-in via ``OV2_ENERGON_DROP_YIELDED=1`` (default off: the
validated path is untouched); the merged48 production wrapper turns it on.
"""
import importlib
import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

# (module, class, method, anchor in the dedented method source, replacement) -- energon 7.4.1 anchors
_DROP_YIELDED_PATCHES = (
    (
        "megatron.energon.wrappers.repeat_dataset",
        "RepeatDataset",
        "__iter__",
        "        for sample in self.dataset:\n            self._index += 1\n            yield sample\n",
        # Box the sample and yield box.pop(): while the generator is suspended at the yield, the frame's only
        # local is the (now empty) box. A `del` AFTER the yield would run only on the next draw -- useless.
        "        for sample in self.dataset:\n            self._index += 1\n            sample = [sample]\n            yield sample.pop()\n",
    ),
    (
        "megatron.energon.wrappers.map_dataset",
        "MapDataset",
        "__iter__",
        "                yield add_sample_restore_key(\n                    mapped_sample,\n                    sample_idx,\n                    src=self,\n                )\n",
        "                mapped_sample = [add_sample_restore_key(\n                    mapped_sample,\n                    sample_idx,\n                    src=self,\n                )]\n                yield mapped_sample.pop()\n",
    ),
)


def _energon_version() -> str:
    try:
        return importlib.metadata.version("megatron-energon")
    except Exception:  # not installed via pip metadata
        return "?"


def rebind_with_patch(cls, method_name: str, old: str, new: str, module) -> tuple[bool, str]:
    """Re-compile ``cls.<method_name>`` from its installed source with ``old`` replaced by ``new`` (exactly once).

    The new function is compiled with ``module.__dict__`` as globals, so every name the original resolved
    (imports, type vars) still resolves. Returns (patched, reason).
    """
    fn = getattr(cls, method_name, None)
    if fn is None:
        return False, f"{cls.__name__}.{method_name} missing"
    if getattr(fn, "__ov2_patched__", False):
        return True, "already patched"
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:  # no source in the image (pyc-only) -> cannot patch safely
        return False, f"no source for {cls.__name__}.{method_name}: {e}"
    n = src.count(old)
    if n != 1:
        return False, f"anchor matched {n}x in {cls.__name__}.{method_name} (energon {_energon_version()})"
    patched_src = src.replace(old, new)
    ns: dict = {}
    code = compile(patched_src, f"<ov2-drop-yielded {module.__name__}.{cls.__name__}.{method_name}>", "exec")
    exec(code, module.__dict__, ns)  # noqa: S102 -- re-compiling the installed source with the box/pop edit
    new_fn = ns[method_name]
    new_fn.__ov2_patched__ = True
    new_fn.__wrapped_source__ = patched_src
    setattr(cls, method_name, new_fn)
    return True, "patched"


def apply_drop_yielded_patch() -> bool:
    """Apply the drop_yielded patch to the installed energon. Returns True only if EVERY anchor was patched.

    Partial application is rolled back to keep a single, predictable state: either both frames drop their
    reference or the validated (unpatched) behaviour stays.
    """
    results = []
    originals = []
    for mod_name, cls_name, meth, old, new in _DROP_YIELDED_PATCHES:
        try:
            module = importlib.import_module(mod_name)
            cls = getattr(module, cls_name)
        except Exception as e:
            results.append((False, f"{mod_name}.{cls_name}: import failed: {type(e).__name__}: {e}"))
            continue
        originals.append((cls, meth, getattr(cls, meth, None)))
        results.append(rebind_with_patch(cls, meth, old, new, module))
    ok = bool(results) and all(r[0] for r in results)
    if not ok:
        for cls, meth, orig in originals:  # roll back any partial success
            if orig is not None:
                setattr(cls, meth, orig)
        logger.warning(
            "[ov2 energon_patches] drop_yielded NOT applied (energon %s): %s -- running with the unpatched "
            "energon; expect per-component sample pinning in dataloader workers",
            _energon_version(), "; ".join(r[1] for r in results),
        )
        return False
    logger.info("[ov2 energon_patches] drop_yielded applied to energon %s: %s", _energon_version(),
                "; ".join(f"{p[1]}.{p[2]} {r[1]}" for p, r in zip(_DROP_YIELDED_PATCHES, results)))
    return True
