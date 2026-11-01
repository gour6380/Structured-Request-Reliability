"""Verify the actual artifact-preservation contract using tiny temporary files."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from request_reliability.artifacts import ArtifactError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_notebook_preservation.py"
SPEC = importlib.util.spec_from_file_location("notebook_preservation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preservation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preservation)


class PreservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "data/frozen").mkdir(parents=True)
        (self.root / "reports").mkdir()
        (self.root / "data/frozen/weights").write_bytes(b"weights")
        self.report = self.root / "reports/results.json"
        self.report.write_bytes(b'{"valid": 1}')
        self.baseline = preservation.capture(self.root)

    def test_capture_and_verify_exact_existing_files(self) -> None:
        result = preservation.verify(self.root, self.baseline)
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["protected_file_count"], 2)
        self.assertEqual(result["new_paths"], [])

    def test_new_report_is_allowed_and_reported(self) -> None:
        (self.root / "reports/new-run.json").write_text("{}")
        result = preservation.verify(self.root, self.baseline)
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["new_paths"], ["reports/new-run.json"])

    def test_same_size_change_with_restored_mtime_still_fails(self) -> None:
        stat = self.report.stat()
        self.report.write_bytes(b'{"valid": 0}')
        os.utime(self.report, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        result = preservation.verify(self.root, self.baseline)
        self.assertFalse(result["unchanged"])
        self.assertEqual(result["changed"], ["reports/results.json"])

    def test_mtime_only_change_is_detected(self) -> None:
        stat = self.report.stat()
        os.utime(self.report, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        result = preservation.verify(self.root, self.baseline)
        self.assertEqual(result["changed"], ["reports/results.json"])

    def test_missing_old_file_fails(self) -> None:
        self.report.unlink()
        result = preservation.verify(self.root, self.baseline)
        self.assertFalse(result["unchanged"])
        self.assertEqual(result["missing"], ["reports/results.json"])

    def test_receipt_cannot_overwrite_or_touch_protected_directories(self) -> None:
        for output in (self.report, self.root / "data/frozen/new.json"):
            with self.subTest(output=output), self.assertRaises(ArtifactError):
                preservation.write_new_json(self.root, output, {})
        output = self.root / "verification/receipt.json"
        preservation.write_new_json(self.root, output, {})
        with self.assertRaises(FileExistsError):
            preservation.write_new_json(self.root, output, {})

    def test_symlink_and_out_of_scope_baselines_fail(self) -> None:
        self.baseline["files"]["../outside"] = {}
        with self.assertRaises(ArtifactError):
            preservation.verify(self.root, self.baseline)
        del self.baseline["files"]["../outside"]
        self.report.unlink()
        self.report.symlink_to(self.root / "data/frozen/weights")
        with self.assertRaises(ArtifactError):
            preservation.verify(self.root, self.baseline)


if __name__ == "__main__":
    unittest.main()
