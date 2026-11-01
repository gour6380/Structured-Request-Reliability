#!/usr/bin/env python3
"""Copy a verified frozen run into this standalone project without source writes.

Only the standard library and this project's artifact checker are used. Recorded
source snapshots are copied as evidence and never imported or executed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from request_reliability.artifacts import (  # noqa: E402
    SYSTEMS,
    ArtifactError,
    checked_path,
    checked_root,
    safe_relative,
    sha256_file,
    verify_bundle,
)

EXCLUDED_PARTS = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        ".ipynb_checkpoints",
    }
)
EXCLUDED_NAMES = frozenset({".DS_Store"})
MAX_COPY_BYTES = 6_000_000_000


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactError(f"Refusing to write symlink: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected JSON object: {path}")
    return value


def source_fingerprint(source: Path) -> dict[str, Any]:
    """Hash all scientific/project files; explicitly exclude env/VCS/cache files."""
    source = checked_root(source)
    files: dict[str, dict[str, Any]] = {}
    for directory, directories, names in os.walk(source, followlinks=False):
        parent = Path(directory)
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_PARTS)
        for name in directories:
            if (parent / name).is_symlink():
                raise ArtifactError(f"Symlink in preservation scope: {parent / name}")
        for name in sorted(names):
            if name in EXCLUDED_NAMES or name.endswith((".pyc", ".pyo")):
                continue
            path = parent / name
            relative = path.relative_to(source).as_posix()
            checked_path(source, relative)
            stat = path.stat()
            files[relative] = {
                "sha256": sha256_file(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "source": str(source),
        "scope": (
            "All project regular-file contents and mtimes except listed environment/VCS/cache paths"
        ),
        "excluded_directory_names": sorted(EXCLUDED_PARTS),
        "excluded_file_names": sorted(EXCLUDED_NAMES),
        "excluded_suffixes": [".pyc", ".pyo"],
        "files": files,
        "total_bytes": sum(record["bytes"] for record in files.values()),
    }


def verify_source_unchanged(snapshot_path: Path) -> dict[str, Any]:
    before = read_json(snapshot_path)
    after = source_fingerprint(Path(before["source"]))
    left, right = before["files"], after["files"]
    added, removed = sorted(right.keys() - left.keys()), sorted(left.keys() - right.keys())
    changed = sorted(name for name in left.keys() & right.keys() if left[name] != right[name])
    result = {
        "verified_at": utc_now(),
        "unchanged": not (added or removed or changed),
        "source": before["source"],
        "scope": before["scope"],
        "file_count": len(left),
        "total_bytes": before["total_bytes"],
        "added": added,
        "removed": removed,
        "changed": changed,
        "before_snapshot_sha256": sha256_file(snapshot_path),
    }
    if not result["unchanged"]:
        raise ArtifactError(f"Source changed: {json.dumps(result)}")
    return result


def copy_plan(run: Path, source: Path) -> dict[str, Path]:
    plan: dict[str, Path] = {}

    def add(relative: str, destination: str) -> None:
        safe_relative(destination)
        if destination in plan:
            raise ArtifactError(f"Duplicate destination: {destination}")
        plan[destination] = checked_path(run, relative)

    def tree(relative: str, destination: str) -> None:
        directory = checked_path(run, relative, must_exist=False)
        if not directory.is_dir():
            raise ArtifactError(f"Missing source directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ArtifactError(f"Symlink in source artifact tree: {path}")
            if path.is_file():
                add(
                    path.relative_to(run).as_posix(),
                    f"{destination}/{path.relative_to(directory).as_posix()}",
                )

    for name in ("base_model", "quantized_base", "tokenizer"):
        tree(f"inputs/{name}", name)
    for arm in ("lora", "qlora"):
        tree(f"trials/{arm}/checkpoints/final/adapter", f"adapters/{arm}")
        for name in ("progress.json", "hashes.json"):
            add(f"trials/{arm}/checkpoints/final/{name}", f"provenance/{arm}/{name}")
        add(f"trials/{arm}/training_summary.json", f"provenance/{arm}/training_summary.json")
    for name in ("test", "validation", "tokenized_test"):
        add(f"preparation/{name}.jsonl", f"examples/{name}.jsonl")
    for name in ("ontology.json", "system_prompt.txt"):
        add(f"preparation/{name}", name)
    for system in SYSTEMS:
        add(f"evaluations/{system}/test/predictions.jsonl", f"reference/{system}.jsonl")
        add(f"evaluations/{system}/test/metrics.json", f"reference/{system}.metrics.json")
    for relative, destination in {
        "manifest.json": "artifact_manifest.json",
        "config.json": "config.json",
        "environment.json": "environment.json",
        "evaluations/frozen_protocol.json": "frozen_protocol.json",
        "reports/results.md": "source_report.md",
        "inputs/model_snapshot.json": "model_snapshot.json",
        "inputs/quantization.json": "quantization.json",
        "inputs/dataset/manifest.json": "dataset_manifest.json",
        "preparation/tokenization.json": "tokenization.json",
        "preparation/data.json": "data.json",
        "preparation/duplicate_audit.json": "duplicate_audit.json",
    }.items():
        add(relative, f"provenance/{destination}")
    tree("source_snapshot", "provenance/source_snapshot")
    plan["licenses/MASSIVE-CC-BY-4.0.txt"] = checked_path(source, "data/massive-1.1/en-US/LICENSE")
    return plan


def prepare_handoff(source: Path, target: Path) -> dict[str, Any]:
    source, target = checked_root(source), checked_root(target)
    if source == target or source in target.parents or target in source.parents:
        raise ArtifactError("Source and target must be separate sibling projects")
    if not source.is_dir() or not target.is_dir():
        raise ArtifactError("Source and target project directories must exist")
    run = checked_root(source / "runs/run_001")
    bundle = checked_root(target / "data/frozen")
    reports = checked_root(target / "reports")
    if bundle.exists():
        raise ArtifactError("Frozen bundle already exists; verify it instead of overwriting")
    before_path = reports / "source_before.json"
    if before_path.exists():
        raise ArtifactError("Source preservation record already exists; it must not be overwritten")
    artifact_manifest = read_json(checked_path(run, "manifest.json"))
    protocol = read_json(checked_path(run, "evaluations/frozen_protocol.json"))
    config = read_json(checked_path(run, "config.json"))
    snapshot_manifest = read_json(checked_path(run, "source_snapshot/manifest.json"))
    artifact_records = artifact_manifest["artifacts"]
    protocol_hashes = protocol["artifacts"]
    # Validate every recorded path, including entries outside this bounded export.
    for mapping in (artifact_records, protocol_hashes, snapshot_manifest):
        for name in mapping:
            safe_relative(name)
    if protocol.get("primary_checkpoint") != "final" or protocol.get("systems") != list(SYSTEMS):
        raise ArtifactError("Expected frozen final-checkpoint four-system protocol")
    if protocol["model_revision"] != config["model_revision"]:
        raise ArtifactError("Model revision differs between config and evaluation protocol")
    for arm in ("lora", "qlora"):
        record = protocol["training"][arm]
        progress = read_json(checked_path(run, f"trials/{arm}/checkpoints/final/progress.json"))
        if not progress.get("completed") or progress.get("epoch") != 5:
            raise ArtifactError(f"Final {arm} checkpoint has not completed five epochs")
        if (
            record["epochs_completed"] != 5
            or record["optimizer_updates"] != progress["global_step"]
        ):
            raise ArtifactError(f"Training identity mismatch: {arm}")
    plan = copy_plan(run, source)
    planned_bytes = sum(path.stat().st_size for path in plan.values())
    free_bytes = shutil.disk_usage(target).free
    if planned_bytes > MAX_COPY_BYTES or free_bytes < planned_bytes + 1_000_000_000:
        raise ArtifactError(f"Storage preflight failed: copy={planned_bytes}, free={free_bytes}")
    print(
        f"Fingerprinting source project; planned independent copy {planned_bytes:,} bytes",
        flush=True,
    )
    before = source_fingerprint(source)
    write_json(before_path, before)
    records: dict[str, dict[str, Any]] = {}
    attestations: dict[str, dict[str, Any]] = {}
    dataset = read_json(checked_path(run, "inputs/dataset/manifest.json"))
    for destination, path in plan.items():
        source_relative = path.relative_to(source).as_posix()
        observed = before["files"][source_relative]
        digest, size = observed["sha256"], observed["bytes"]
        expectations: dict[str, str] = {}
        if path.is_relative_to(run):
            relative = path.relative_to(run).as_posix()
            if relative in artifact_records:
                record = artifact_records[relative]
                expectations["original_artifact_manifest"] = record["sha256"]
                if record["bytes"] != size:
                    raise ArtifactError(f"Inherited manifest size mismatch: {relative}")
            if relative in protocol_hashes:
                expectations["frozen_evaluation_protocol"] = protocol_hashes[relative]
            if relative.startswith("source_snapshot/"):
                recorded_name = relative.removeprefix("source_snapshot/")
                if recorded_name in snapshot_manifest:
                    expectations["recorded_source_snapshot_manifest"] = snapshot_manifest[
                        recorded_name
                    ]
        else:
            expectations["dataset_license_manifest"] = dataset["files"]["LICENSE"]
        for authority, expected in expectations.items():
            if digest != expected:
                raise ArtifactError(f"Inherited hash mismatch ({authority}): {source_relative}")
        # Original metadata manifests/config/environment were not self-registered.
        if not expectations and destination not in {
            "provenance/artifact_manifest.json",
            "provenance/config.json",
            "provenance/environment.json",
            "provenance/source_snapshot/manifest.json",
        }:
            raise ArtifactError(f"Source artifact lacks recorded checksum: {source_relative}")
        records[destination] = {"sha256": digest, "bytes": size}
        attestations[destination] = {
            "source_relative_path": source_relative,
            "verified_against": list(expectations) or ["handoff_observation_only"],
        }
    print(f"Verified {len(plan)} selected source artifacts; copying independent bytes", flush=True)
    bundle.mkdir(parents=True)
    for destination, path in plan.items():
        output = checked_path(bundle, destination, must_exist=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as reader, output.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if (
            output.stat().st_ino == path.stat().st_ino
            and output.stat().st_dev == path.stat().st_dev
        ):
            raise ArtifactError(f"Source and destination share an inode: {destination}")
        if sha256_file(output) != records[destination]["sha256"]:
            raise ArtifactError(f"Independent copy hash mismatch: {destination}")
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "files": records,
        "origin": {
            "project_name": source.name,
            "run": "run_001",
            "model_id": config["model_id"],
            "model_revision": protocol["model_revision"],
            "checkpoint": "final",
            "epochs_completed": 5,
            "source_manifest_sha256": records["provenance/artifact_manifest.json"]["sha256"],
            "frozen_protocol_sha256": records["provenance/frozen_protocol.json"]["sha256"],
            "source_preservation_snapshot_sha256": sha256_file(before_path),
        },
        "config": config,
        "generation": protocol["generation"],
        "source_verification": attestations,
        "exclusions": [
            "optimizer/training states",
            "best-validation checkpoints",
            "training examples",
            "environment/cache",
        ],
        "evaluation_status": (
            "Existing test data and predictions are regression evidence, not an untouched test set."
        ),
    }
    write_json(bundle / "manifest.json", manifest)
    verified = verify_bundle(bundle)
    preservation = verify_source_unchanged(before_path)
    write_json(reports / "source_preservation.json", preservation)
    receipt = {
        "completed_at": utc_now(),
        "source": str(source),
        "destination": str(bundle),
        "verification": verified,
        "source_unchanged": preservation["unchanged"],
        "source_fingerprint_file_count": preservation["file_count"],
        "disk_free_before_bytes": free_bytes,
        "disk_free_after_bytes": shutil.disk_usage(target).free,
        "independent_byte_copy": True,
        "source_model_or_code_executed": False,
        "unattested_metadata": [
            key
            for key, record in attestations.items()
            if record["verified_against"] == ["handoff_observation_only"]
        ],
    }
    write_json(reports / "handoff_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original standalone project; read only")
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-source", type=Path, metavar="SNAPSHOT_JSON")
    args = parser.parse_args()
    try:
        if args.verify_source:
            result = verify_source_unchanged(args.verify_source)
        elif args.source:
            result = prepare_handoff(args.source, args.target)
        else:
            parser.error("Provide --source or --verify-source")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ArtifactError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"Handoff failed without modifying the source: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
