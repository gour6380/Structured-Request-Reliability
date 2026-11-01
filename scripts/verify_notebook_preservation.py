#!/usr/bin/env python3
"""Capture or check saved artifact preservation during a notebook refactor.

Only paths present at capture time are protected. New report files are allowed;
this check does not confer scientific validity on newly generated outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from request_reliability.artifacts import (  # noqa: E402
    ArtifactError,
    checked_path,
    checked_root,
    sha256_file,
)

PROTECTED_DIRECTORIES = ("data/frozen", "reports")


def file_record(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ArtifactError(f"File changed while reading: {path}")
    return {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def inventory(root: Path) -> dict[str, Path]:
    root = checked_root(root)
    files: dict[str, Path] = {}
    for name in PROTECTED_DIRECTORIES:
        directory = checked_path(root, name, must_exist=False)
        if not directory.is_dir():
            raise ArtifactError(f"Missing protected directory: {name}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ArtifactError(f"Symlink in protected artifact tree: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                files[relative] = checked_path(root, relative)
    return files


def capture(root: Path) -> dict[str, Any]:
    root = checked_root(root)
    files = {name: file_record(path) for name, path in inventory(root).items()}
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "project": str(root),
        "scope": (
            "Existing data/frozen and reports files: SHA-256, byte count and modification time"
        ),
        "protected_directories": list(PROTECTED_DIRECTORIES),
        "new_paths_policy": "New paths are reported but do not fail preservation of existing files",
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files.values()),
        "files": files,
    }


def verify(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    root = checked_root(root)
    if baseline.get("schema_version") != 1 or not isinstance(baseline.get("files"), dict):
        raise ArtifactError("Unsupported preservation baseline")
    files = baseline["files"]
    if not files:
        raise ArtifactError("Preservation baseline has no protected files")
    current = inventory(root)
    missing, changed = [], []
    for name, expected in sorted(files.items()):
        # Validate even missing or malicious paths before considering inventory.
        checked_path(root, name, must_exist=False)
        if not any(name.startswith(f"{prefix}/") for prefix in PROTECTED_DIRECTORIES):
            raise ArtifactError(f"Baseline path is outside its declared scope: {name}")
        if name not in current:
            missing.append(name)
        elif file_record(current[name]) != expected:
            changed.append(name)
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "unchanged": not missing and not changed,
        "project": str(root),
        "scope": baseline["scope"],
        "protected_file_count": len(files),
        "protected_total_bytes": sum(record["bytes"] for record in files.values()),
        "missing": missing,
        "changed": changed,
        "new_paths": sorted(current.keys() - files.keys()),
    }


def write_new_json(root: Path, path: Path, value: dict[str, Any]) -> None:
    root, path = checked_root(root), checked_root(path)
    if not path.is_relative_to(root):
        raise ArtifactError("Preservation receipts must be inside the standalone project")
    relative = path.relative_to(root).as_posix()
    if any(
        relative == prefix or relative.startswith(f"{prefix}/") for prefix in PROTECTED_DIRECTORIES
    ):
        raise ArtifactError("Refusing to write inside a protected artifact directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--baseline", type=Path, default=Path("verification/notebook-refactor-before.json")
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON receipt path, outside data/frozen and reports"
    )
    args = parser.parse_args()
    try:
        root = checked_root(args.project)
        baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
        if args.capture:
            if args.output:
                parser.error("--capture writes --baseline; do not also supply --output")
            result = capture(root)
            write_new_json(root, baseline_path, result)
            summary = {key: value for key, value in result.items() if key != "files"}
        else:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            result = verify(root, baseline)
            result["baseline_sha256"] = sha256_file(baseline_path)
            if args.output:
                output = args.output if args.output.is_absolute() else root / args.output
                write_new_json(root, output, result)
            summary = result
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if args.capture or result["unchanged"] else 1
    except (ArtifactError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"Preservation check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
