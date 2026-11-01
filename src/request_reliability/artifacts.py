"""Read-only integrity checks for independently materialized model bundles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SYSTEMS = ("base", "quantized_base", "lora", "qlora")
REQUIRED_FILES = frozenset(
    {
        "base_model/config.json",
        "base_model/generation_config.json",
        "base_model/model.safetensors",
        "quantized_base/config.json",
        "quantized_base/generation_config.json",
        "quantized_base/model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja",
        "ontology.json",
        "system_prompt.txt",
        "examples/test.jsonl",
        "examples/validation.jsonl",
        "examples/tokenized_test.jsonl",
        "provenance/config.json",
        "provenance/environment.json",
        "provenance/frozen_protocol.json",
        "provenance/artifact_manifest.json",
        "provenance/source_snapshot/manifest.json",
        "provenance/source_report.md",
        "licenses/MASSIVE-CC-BY-4.0.txt",
    }
    | {
        f"reference/{system}{suffix}"
        for system in SYSTEMS
        for suffix in (".jsonl", ".metrics.json")
    }
    | {
        f"adapters/{arm}/{name}"
        for arm in ("lora", "qlora")
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
)


class ArtifactError(ValueError):
    """A file or its provenance fails a bundle invariant."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    """Reject traversal instead of silently normalizing it."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"Invalid relative artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ArtifactError(f"Unsafe relative artifact path: {value!r}")
    return path


def checked_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = root
    for part in safe_relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ArtifactError(f"Symlinks are forbidden in artifact paths: {path}")
    if must_exist and not path.is_file():
        raise ArtifactError(f"Missing regular artifact file: {path}")
    return path


def checked_root(root: Path) -> Path:
    """Reject user-supplied traversal or symlinks, including parent directories."""
    root = Path(root).expanduser()
    if ".." in root.parts:
        raise ArtifactError(f"Parent traversal is forbidden: {root}")
    root = root.absolute()
    for component in (root, *root.parents):
        if component.is_symlink():
            raise ArtifactError(f"Symlink directory is forbidden: {component}")
    return root


def bundle_root(root: Path) -> Path:
    root = checked_root(root)
    # Public APIs accept either the standalone project or its bundle directory.
    if not (root / "manifest.json").exists() and (root / "data/frozen").exists():
        root = checked_root(root / "data/frozen")
    return root


def load_manifest(root: Path) -> dict[str, Any]:
    root = bundle_root(root)
    path = checked_path(root, "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ArtifactError("Unsupported frozen-bundle manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactError("Manifest must contain a nonempty files mapping")
    for name, metadata in files.items():
        safe_relative(name)
        if not isinstance(metadata, dict):
            raise ArtifactError(f"Invalid metadata for {name}")
        digest, size = metadata.get("sha256"), metadata.get("bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ArtifactError(f"Invalid SHA-256 for {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactError(f"Invalid byte count for {name}")
    for key in ("origin", "config", "generation"):
        if not isinstance(manifest.get(key), dict):
            raise ArtifactError(f"Manifest must contain {key}")
    return manifest


def verify_bundle(root: Path, full: bool = True) -> dict[str, Any]:
    """Verify every file's size and independence; hash every file when full=True.

    No source project is opened. full=False is a structural preflight, never a
    cryptographic integrity claim. Unlisted files and all symlinks are rejected.
    """
    root = bundle_root(root)
    manifest = load_manifest(root)
    files = manifest["files"]
    missing = sorted(REQUIRED_FILES - files.keys())
    if missing:
        raise ArtifactError(f"Manifest lacks required artifacts: {missing}")
    seen_inodes: set[tuple[int, int]] = set()
    total_bytes = 0
    for name, metadata in files.items():
        path = checked_path(root, name)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if stat.st_nlink != 1 or identity in seen_inodes:
            raise ArtifactError(f"Artifact is not an independent regular file: {name}")
        seen_inodes.add(identity)
        if stat.st_size != metadata["bytes"]:
            raise ArtifactError(f"Byte count mismatch: {name}")
        if full and sha256_file(path) != metadata["sha256"]:
            raise ArtifactError(f"SHA-256 mismatch: {name}")
        total_bytes += stat.st_size
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"Unexpected symlink: {path}")
        if path.is_file() and path != root / "manifest.json":
            name = path.relative_to(root).as_posix()
            if name not in files:
                raise ArtifactError(f"Unlisted artifact: {name}")
    copied_config = json.loads(
        checked_path(root, "provenance/config.json").read_text(encoding="utf-8")
    )
    copied_protocol = json.loads(
        checked_path(root, "provenance/frozen_protocol.json").read_text(encoding="utf-8")
    )
    if manifest["config"] != copied_config:
        raise ArtifactError("Manifest config differs from copied provenance/config.json")
    if not isinstance(copied_protocol, dict) or manifest["generation"] != copied_protocol.get(
        "generation"
    ):
        raise ArtifactError("Manifest generation differs from copied frozen evaluation protocol")
    return {
        "valid": True,
        "full_hash_verification": full,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "systems": list(SYSTEMS),
    }
