"""Integrity checks use tiny synthetic files and never import a model framework."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from request_reliability.artifacts import (
    REQUIRED_FILES,
    ArtifactError,
    load_manifest,
    safe_relative,
    sha256_file,
    verify_bundle,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_handoff.py"
SPEC = importlib.util.spec_from_file_location("prepare_handoff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # macOS temporary-directory parents can be system symlink aliases.
        self.directory = Path(self.temporary.name).resolve()
        self.root = self.directory / "data/frozen"
        self.root.mkdir(parents=True)
        records = {}
        for name in REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "provenance/config.json": b"{}",
                "provenance/frozen_protocol.json": b'{"generation": {}}',
            }.get(name, b"sample")
            path.write_bytes(data)
            records[name] = {"sha256": sha256_file(path), "bytes": len(data)}
        self.manifest = {
            "schema_version": 1,
            "files": records,
            "origin": {},
            "config": {},
            "generation": {},
        }
        self.save_manifest()

    def save_manifest(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_verified_bundle_needs_no_source_project(self) -> None:
        result = verify_bundle(self.directory)
        self.assertTrue(result["valid"])
        self.assertTrue(result["full_hash_verification"])
        self.assertEqual(result["file_count"], len(REQUIRED_FILES))
        self.assertEqual(load_manifest(self.root), self.manifest)

    def test_same_size_corruption_is_detected(self) -> None:
        (self.root / "base_model/model.safetensors").write_bytes(b"broken")
        with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
            verify_bundle(self.root)
        self.assertFalse(verify_bundle(self.root, full=False)["full_hash_verification"])

    def test_manifest_config_cannot_override_copied_provenance(self) -> None:
        self.manifest["config"] = {"model_revision": "silently_changed"}
        self.save_manifest()
        with self.assertRaisesRegex(ArtifactError, "config differs"):
            verify_bundle(self.root)

    def test_manifest_generation_cannot_override_frozen_protocol(self) -> None:
        self.manifest["generation"] = {"max_new_tokens": 1}
        self.save_manifest()
        with self.assertRaisesRegex(ArtifactError, "generation differs"):
            verify_bundle(self.root, full=False)

    def test_missing_file_is_detected(self) -> None:
        (self.root / "reference/qlora.jsonl").unlink()
        with self.assertRaisesRegex(ArtifactError, "Missing regular"):
            verify_bundle(self.root)

    def test_omitting_reference_from_manifest_fails(self) -> None:
        del self.manifest["files"]["reference/lora.jsonl"]
        self.save_manifest()
        with self.assertRaisesRegex(ArtifactError, "required artifacts"):
            verify_bundle(self.root)

    def test_path_traversal_absolute_and_noncanonical_paths_fail(self) -> None:
        for name in ("../escape", "/etc/passwd", "a/../b", "a//b", "a/./b", "a\\b", ""):
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                safe_relative(name)
        self.manifest["files"]["../escape"] = {"sha256": "0" * 64, "bytes": 0}
        self.save_manifest()
        with self.assertRaises(ArtifactError):
            load_manifest(self.root)

    def test_symlink_file_is_rejected(self) -> None:
        path = self.root / "ontology.json"
        path.unlink()
        path.symlink_to(self.root / "system_prompt.txt")
        with self.assertRaisesRegex(ArtifactError, "Symlink"):
            verify_bundle(self.root)

    def test_symlink_directory_is_rejected(self) -> None:
        alias = self.directory / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ArtifactError, "Symlink"):
            load_manifest(alias)

    def test_hard_links_are_rejected(self) -> None:
        path = self.root / "ontology.json"
        path.unlink()
        os.link(self.root / "system_prompt.txt", path)
        with self.assertRaisesRegex(ArtifactError, "independent"):
            verify_bundle(self.root)

    def test_unlisted_nested_manifest_cannot_bypass_inventory(self) -> None:
        path = self.root / "unexpected/manifest.json"
        path.parent.mkdir()
        path.write_text("{}")
        with self.assertRaisesRegex(ArtifactError, "Unlisted"):
            verify_bundle(self.root)

    def test_invalid_metadata_fails(self) -> None:
        name = "ontology.json"
        for metadata in ({"sha256": "bad", "bytes": 6}, {"sha256": "0" * 64, "bytes": True}):
            with self.subTest(metadata=metadata):
                self.manifest["files"][name] = metadata
                self.save_manifest()
                with self.assertRaises(ArtifactError):
                    load_manifest(self.root)

    def test_source_and_target_cannot_overlap(self) -> None:
        for target in (self.directory, self.directory / "child", self.directory.parent):
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(ArtifactError, "separate sibling"),
            ):
                handoff.prepare_handoff(self.directory, target)

    def test_source_preservation_tracks_content_and_new_files(self) -> None:
        source = self.directory / "original"
        source.mkdir()
        script = source / "implementation.py"
        script.write_text("value = 1\n")
        env = source / ".venv"
        env.mkdir()
        (env / "package.py").write_text("excluded = True\n")
        snapshot = self.directory / "before.json"
        handoff.write_json(snapshot, handoff.source_fingerprint(source))
        self.assertEqual(handoff.verify_source_unchanged(snapshot)["file_count"], 1)
        script.write_text("value = 2\n")
        with self.assertRaisesRegex(ArtifactError, "Source changed"):
            handoff.verify_source_unchanged(snapshot)


if __name__ == "__main__":
    unittest.main()
