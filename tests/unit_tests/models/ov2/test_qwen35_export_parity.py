"""CPU tests for the Qwen3.5 export parity check (raw safetensors bytes; no torch/transformers needed)."""

import importlib.util
import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONVERT = ROOT / "examples/models/qwen/qwen35_vl_ov2/convert"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CONVERT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load("ov2_parity_under_test", "verify_export_parity.py")
validator = _load("ov2_validator_under_test", "validate_hf_export.py")


def bf16(values) -> bytes:
    """Pack float values as bf16 (truncate the f32 mantissa), the dtype the export actually ships."""
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16) for v in values)


def write_shard(path: Path, tensors: dict) -> None:
    """Write one safetensors file: {name: (dtype, shape, payload_bytes)}."""
    header, blob, offset = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(blob))


def write_export(
    root: Path,
    *,
    experts=(1.0, 2.0, 3.0, 4.0),
    sharded: bool = False,
    ep: int = 4,
    source: str = "/ckpts/iter_0004500",
    iteration: int | None = 4500,
    stamped: bool = True,
) -> Path:
    """A miniature export: one packed expert tensor (leading dim = expert), one norm, one MTP tensor."""
    packed = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": (
            "BF16",
            (4, 2),
            bf16([experts[0], experts[0], experts[1], experts[1], experts[2], experts[2], experts[3], experts[3]]),
        )
    }
    rest = {
        "model.language_model.norm.weight": ("BF16", (4,), bf16([0.5, 0.25, 0.125, 1.0])),
        "mtp.layers.0.input_layernorm.weight": ("BF16", (2,), bf16([1.0, 2.0])),
    }
    root.mkdir(parents=True, exist_ok=True)
    if sharded:
        write_shard(root / "model-00001-of-00002.safetensors", packed)
        write_shard(root / "model-00002-of-00002.safetensors", rest)
        weight_map = dict.fromkeys(packed, "model-00001-of-00002.safetensors")
        weight_map.update(dict.fromkeys(rest, "model-00002-of-00002.safetensors"))
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        write_shard(root / "model.safetensors", {**packed, **rest})
    if stamped:
        # What ov2_30b_export_ep8.py stamps after a completed save.
        (root / "export_provenance.json").write_text(
            json.dumps({"expert_parallel_size": ep, "source_checkpoint": source, "iteration": iteration})
        )
    return root


class ExportParityTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_same_weights_pass_across_different_shard_layouts(self) -> None:
        """Bit-identical values must pass even when the two runs split the shards differently."""
        ref = write_export(self.base / "ep2", sharded=True, ep=2)
        cand = write_export(self.base / "ep4", sharded=False)
        report = parity.compare(ref, cand)
        self.assertEqual(report["verdict"], "PASS", report)
        self.assertEqual(report["tensors_compared"], 3)
        self.assertEqual(report["mtp_tensors_compared"], 1)
        self.assertEqual(report["candidate_manifest_digest"], parity.manifest_digest(parity.manifest(cand)))
        # The digest must ignore how the run happened to shard, and track only dtype/shape/values.
        self.assertEqual(report["candidate_manifest_digest"], parity.manifest_digest(parity.manifest(ref)))

    def test_permuted_experts_are_caught_and_localised(self) -> None:
        """An EP remap bug keeps every value present but moves it: the global stats match, bytes do not."""
        ref = write_export(self.base / "ep2", experts=(1.0, 2.0, 3.0, 4.0), ep=2)
        cand = write_export(self.base / "ep4", experts=(1.0, 2.0, 4.0, 3.0))
        report = parity.compare(ref, cand)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["differing_tensors"], ["model.language_model.layers.0.mlp.experts.gate_up_proj"])
        detail = report["diagnosis"][0]
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy unavailable: sha256 verdict still correct, diagnosis degraded")
        self.assertEqual(detail["max_abs_diff"], 1.0)
        self.assertEqual(detail["differing_leading_slices"], [2, 3])

    def test_missing_and_extra_tensors_fail(self) -> None:
        ref = write_export(self.base / "ep2", ep=2)
        cand = write_export(self.base / "ep4")
        write_shard(
            cand / "model.safetensors",
            {"model.language_model.norm.weight": ("BF16", (4,), bf16([0.5, 0.25, 0.125, 1.0]))},
        )
        report = parity.compare(ref, cand)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertIn("model.language_model.layers.0.mlp.experts.gate_up_proj", report["missing_in_candidate"])

    def test_truncated_shard_is_rejected_not_silently_hashed(self) -> None:
        cand = write_export(self.base / "ep4")
        path = cand / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises(parity.ParityError):
            parity.manifest(cand)

    def test_both_weight_layouts_present_is_rejected(self) -> None:
        cand = write_export(self.base / "ep4", sharded=True)
        write_shard(cand / "model.safetensors", {"x": ("F32", (1,), struct.pack("<f", 1.0))})
        with self.assertRaises(parity.ParityError):
            parity.shard_files(cand)

    def test_self_comparison_is_refused(self) -> None:
        """Comparing a directory with itself passes trivially and proves nothing."""
        cand = write_export(self.base / "ep4")
        with self.assertRaises(parity.ParityError) as caught:
            parity.compare(cand, cand)
        self.assertIn("same directory", str(caught.exception))

    def test_symlinked_self_comparison_is_refused(self) -> None:
        cand = write_export(self.base / "ep4")
        link = self.base / "alias"
        link.symlink_to(cand)
        with self.assertRaises(parity.ParityError):
            parity.compare(link, cand)

    def test_two_exports_at_the_same_ep_prove_nothing(self) -> None:
        ref = write_export(self.base / "a", ep=4)
        cand = write_export(self.base / "b", ep=4)
        with self.assertRaises(parity.ParityError) as caught:
            parity.compare(ref, cand)
        self.assertIn("compares a layout with itself", str(caught.exception))

    def test_exports_of_different_checkpoints_are_refused(self) -> None:
        ref = write_export(self.base / "ep2", ep=2, source="/ckpts/iter_0004000", iteration=4000)
        cand = write_export(self.base / "ep4")
        with self.assertRaises(parity.ParityError) as caught:
            parity.compare(ref, cand)
        self.assertIn("different source checkpoints", str(caught.exception))

    def test_unstamped_export_cannot_be_certified(self) -> None:
        """Without the worker's stamp the EP and source are hearsay, so the pairing is unprovable."""
        ref = write_export(self.base / "ep2", ep=2)
        cand = write_export(self.base / "ep4", stamped=False)
        with self.assertRaises(parity.ParityError) as caught:
            parity.compare(ref, cand)
        self.assertIn("export_provenance.json", str(caught.exception))


class ParityEvidenceTests(unittest.TestCase):
    """validate_hf_export must only accept EP-invariance evidence that belongs to THIS export."""

    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ref = write_export(self.base / "ep2", ep=2)
        self.cand = write_export(self.base / "ep4")

    def _certify(self) -> dict:
        report = parity.compare(self.ref, self.cand)
        (self.cand / "export_parity.json").write_text(json.dumps(report))
        return report

    def test_absent_report_is_no_evidence(self) -> None:
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "absent")
        report = validator.check_shapes({"a": (1,)}, {"a": (1,)}, require_mtp=False, parity=evidence)
        self.assertFalse(report["numerical_parity_verified"])

    def test_matching_report_certifies_the_export(self) -> None:
        self._certify()
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertTrue(evidence["verified"], evidence)
        self.assertEqual(evidence["compared_eps"], [2, 4])
        report = validator.check_shapes({"a": (1,)}, {"a": (1,)}, require_mtp=False, parity=evidence)
        self.assertTrue(report["ep_invariance_verified"])
        # EP invariance is NOT parity with the SAVE: both sides ran the same mapping, so this stays false.
        self.assertFalse(report["numerical_parity_verified"])
        self.assertTrue(report["numerical_parity_gaps"])

    def test_report_from_a_different_export_is_stale(self) -> None:
        """Exporting another iteration over the same path keeps every shape and offset: only values change.

        A layout-level fingerprint would still match and would silently certify the new weights.
        """
        self._certify()
        write_export(self.base / "ep4", experts=(9.0, 8.0, 7.0, 6.0))
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "stale")

    def test_failed_report_is_no_evidence(self) -> None:
        report = self._certify()
        report["verdict"] = "FAIL"
        (self.cand / "export_parity.json").write_text(json.dumps(report))
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "failed")

    def test_report_copied_from_another_export_is_foreign(self) -> None:
        """A passing report is a file: copying it next to different weights must not certify them."""
        report = self._certify()
        other = write_export(
            self.base / "other", experts=(5.0, 6.0, 7.0, 8.0), source="/ckpts/iter_0009000", iteration=9000
        )
        (other / "export_parity.json").write_text(json.dumps(report))
        evidence = validator.ep_invariance_evidence(other)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "foreign")

    def test_same_ep_report_is_vacuous(self) -> None:
        """Even a hand-written PASS naming one EP twice must not count as evidence."""
        report = self._certify()
        report["reference_provenance"] = dict(report["candidate_provenance"])
        (self.cand / "export_parity.json").write_text(json.dumps(report))
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "vacuous")

    def test_unstamped_export_reports_unstamped(self) -> None:
        report = self._certify()
        (self.cand / "export_provenance.json").unlink()
        (self.cand / "export_parity.json").write_text(json.dumps(report))
        evidence = validator.ep_invariance_evidence(self.cand)
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "unstamped")


if __name__ == "__main__":
    unittest.main()
