"""Notebook orchestration with read-only replay and deliberately selected live runs.

Scoring, model loading and generation belong to the existing package modules.
This layer selects saved evidence, delegates live work and records provenance.
Importing it never imports torch, loads a model, or creates an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import SYSTEMS, checked_path, checked_root, load_manifest, sha256_file


@dataclass(frozen=True)
class NotebookConfig:
    mode: str = "replay"
    device: str = "auto"
    preset: str = "smoke8"
    timeout_seconds: float = 60.0
    benchmark_requests: int = 8
    benchmark_repetitions: int = 3

    def __post_init__(self) -> None:
        if self.mode not in ("replay", "live"):
            raise ValueError("mode must be replay or live")
        if self.device not in ("auto", "mps", "cuda", "cpu"):
            raise ValueError("device must be auto, mps, cuda or cpu")
        if self.preset not in ("smoke8", "full"):
            raise ValueError("preset must be smoke8 or full (2974 reused test requests)")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
        ):
            raise ValueError("timeout_seconds must be finite and in (0, 600]")
        for name, maximum in (("benchmark_requests", 100), ("benchmark_repetitions", 10)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1–{maximum}")

    @property
    def actual_count(self) -> int:
        return 8 if self.preset == "smoke8" else 2974

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "actual_count": self.actual_count}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _notebook_source_sha256(path: Path) -> str:
    """Hash ordered cell types/sources, excluding outputs, IDs and execution metadata."""
    notebook = _read_json(path)
    cells = []
    for cell in notebook["cells"]:
        source = cell["source"]
        cells.append(
            {
                "cell_type": cell["cell_type"],
                "source": source if isinstance(source, str) else "".join(source),
            }
        )
    canonical = json.dumps(cells, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NotebookSession:
    """One notebook's settings and new outputs; existing evidence stays immutable."""

    def __init__(self, project_root: Path | str, config: NotebookConfig | None = None) -> None:
        self.project = checked_root(Path(project_root))
        self.config = config or NotebookConfig()
        metadata = tomllib.loads(checked_path(self.project, "pyproject.toml").read_text())
        if metadata.get("project", {}).get("name") != "structured-request-reliability":
            raise ValueError("Select the standalone structured-request-reliability project")
        for relative in (
            "src/request_reliability/__init__.py",
            "configs/regression.json",
            "uv.lock",
            "data/frozen/manifest.json",
        ):
            checked_path(self.project, relative)
        self.bundle = self.project / "data" / "frozen"
        self._results_dir: Path | None = None

    @property
    def actual_count(self) -> int:
        return self.config.actual_count

    def hardware(self) -> dict[str, Any]:
        """Inspect availability without loading weights or testing an allocation."""
        from .inference import configure_offline

        configure_offline(self.project)
        result: dict[str, Any] = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "requested_device": self.config.device,
            "model_loaded": False,
        }
        try:
            import torch

            mps, cuda = torch.backends.mps.is_available(), torch.cuda.is_available()
            selected = "mps" if mps else "cuda" if cuda else "cpu"
            result.update(
                mps_available=mps,
                cuda_available=cuda,
                auto_device=selected,
                selected_device=selected if self.config.device == "auto" else self.config.device,
            )
        except (ImportError, RuntimeError, OSError) as exc:
            result.update(device_status="unavailable", device_error=str(exc))
        try:
            import psutil

            memory = psutil.virtual_memory()
            result.update(
                memory_status="available",
                total_memory_gib=memory.total / 1024**3,
                available_memory_gib=memory.available / 1024**3,
            )
            try:
                result["swap_used_gib"] = psutil.swap_memory().used / 1024**3
            except (RuntimeError, OSError) as exc:
                result.update(swap_status="unavailable", swap_error=str(exc))
        except (ImportError, RuntimeError, OSError) as exc:
            result.update(memory_status="unavailable", memory_error=str(exc))
        result["memory_note"] = (
            "Point samples, not peak usage or a guarantee of fit. "
            "Live inference retains all existing admission and swap guards."
        )
        return result

    def verify(self, *, full: bool = True) -> dict[str, Any]:
        from .artifacts import verify_bundle

        return verify_bundle(self.bundle, full=full)

    def inventory(self) -> list[dict[str, Any]]:
        """Manifest inventory; call verify to establish actual file integrity."""
        manifest = load_manifest(self.bundle)
        return [
            {"artifact": name, "bytes": entry["bytes"], "sha256": entry["sha256"]}
            for name, entry in sorted(manifest["files"].items())
        ]

    def results_dir(self) -> Path:
        """Explicit write action: allocate a new directory and provenance once."""
        if self._results_dir is None:
            from .cli import save_json

            root = checked_root(self.project / "reports")
            root.mkdir(exist_ok=True)
            destination = root / f"notebook-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
            destination.mkdir(exist_ok=False)
            sources = [self.project / "pyproject.toml", self.project / "uv.lock"]
            sources += sorted((self.project / "src/request_reliability").rglob("*.py"))
            sources += sorted((self.project / "configs").rglob("*.json"))
            hashes = {
                path.relative_to(self.project).as_posix(): sha256_file(
                    checked_path(self.project, path.relative_to(self.project).as_posix())
                )
                for path in sources
            }
            notebook_hashes = {
                path.relative_to(self.project).as_posix(): _notebook_source_sha256(
                    checked_path(self.project, path.relative_to(self.project).as_posix())
                )
                for path in sorted((self.project / "notebooks").rglob("*.ipynb"))
                if ".ipynb_checkpoints" not in path.relative_to(self.project).parts
            }
            packages = {}
            for name in ("torch", "transformers", "peft", "bitsandbytes", "safetensors"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            save_json(
                destination / "session.json",
                {
                    "schema_version": 1,
                    "created_at": datetime.now(UTC).isoformat(),
                    "config": self.config.to_dict(),
                    "project": str(self.project),
                    "manifest_sha256": sha256_file(self.bundle / "manifest.json"),
                    "source_sha256": hashes,
                    "notebook_source_sha256": notebook_hashes,
                    "notebook_hash_policy": (
                        "SHA-256 of UTF-8 canonical JSON containing ordered "
                        "cell_type/source pairs; "
                        "source lists joined; keys sorted; no spaces. Outputs, cell IDs, notebook "
                        "metadata and execution metadata excluded. Runtime config recorded above."
                    ),
                    "python": platform.python_version(),
                    "packages": packages,
                    "qualification": "New notebook session; original evidence is read-only.",
                },
            )
            self._results_dir = destination
        return self._results_dir

    def _new_output(self, kind: str) -> Path:
        # The CLI creates its output with exist_ok=False. Re-running a cell is safe.
        return self.results_dir() / f"{kind}-{uuid4().hex[:8]}"

    def _saved_summaries(self) -> list[tuple[Path, dict[str, Any]]]:
        reports = checked_root(self.project / "reports")
        candidates = sorted(reports.rglob("summary.json"), key=lambda p: p.stat().st_mtime_ns)
        results = []
        for path in candidates:
            if "source_snapshot" not in path.parts:
                checked_path(reports, path.relative_to(reports).as_posix())
                results.append((path, _read_json(path)))
        return results

    @staticmethod
    def _system(system: str) -> None:
        if system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}")

    def _device_matches(self, evidence: dict[str, Any]) -> bool:
        # A failed load with no device identity cannot establish an explicit device match.
        return (
            self.config.device == "auto"
            or evidence.get("identity", {}).get("device") == self.config.device
        )

    @staticmethod
    def _missing(kind: str, **details: Any) -> dict[str, Any]:
        return {
            "status": "missing",
            "kind": kind,
            "inference_performed": False,
            "reason": "No matching saved evidence. Replay mode never starts missing inference.",
            **details,
        }

    def reference_audit(self, *, recompute: bool = False, save: bool = False) -> dict[str, Any]:
        """Re-score saved outputs only when requested; never generate predictions."""
        if recompute:
            from .regression import audit_reference

            result = audit_reference(self.bundle)
        else:
            path = self.project / "reports/reference-audit.json"
            if not path.is_file():
                return self._missing("reference audit")
            checked_path(self.project, "reports/reference-audit.json")
            result = {**_read_json(path), "evidence_path": str(path)}
        if save:
            from .cli import save_json

            output = self._new_output("reference-audit").with_suffix(".json")
            save_json(output, result)
            result = {**result, "output": str(output)}
        return result

    def comparison_table(self) -> list[dict[str, Any]]:
        """Full source comparison, always separate from local subset results."""
        audit = self.reference_audit()
        return [
            {
                "system": system,
                "scope": "Saved CUDA reference; reused test data; no new inference",
                "requests": row["metrics"]["example_count"],
                "entire_response_correct": row["metrics"]["joint_exact_match"],
                "schema_valid": row["metrics"]["schema_valid_rate"],
                "intent_accuracy": row["metrics"]["intent_accuracy"],
                "slot_f1": row["metrics"]["slot_f1"],
            }
            for system, row in audit.get("systems", {}).items()
        ]

    def status_table(self) -> list[dict[str, Any]]:
        """Recorded local attempts, including failures and their actual subset size."""
        rows = []
        for path, summary in self._saved_summaries():
            if "requested_count" not in summary:
                continue
            for system, evidence in summary.get("systems", {}).items():
                comparison = evidence.get("comparison", {})
                rows.append(
                    {
                        "system": system,
                        "status": evidence.get("status")
                        or ("passed" if evidence.get("acceptance", {}).get("passed") else "failed"),
                        "requested": summary["requested_count"],
                        "completed": comparison.get(
                            "example_count", evidence.get("completed_requests", 0)
                        ),
                        "text_agreement": comparison.get("text_agreement", {}).get("rate"),
                        "entire_response_correct": comparison.get("candidate_metrics", {}).get(
                            "joint_exact_match"
                        ),
                        "device": evidence.get("identity", {}).get("device", "not recorded"),
                        "error": evidence.get("error"),
                        "evidence_path": str(path),
                    }
                )
        return rows

    def compare(self, system: str = "qlora") -> dict[str, Any]:
        self._system(system)
        if self.config.mode == "replay":
            for path, summary in reversed(self._saved_summaries()):
                if summary.get("requested_count") == self.actual_count:
                    evidence = summary.get("systems", {}).get(system)
                    if evidence is not None and self._device_matches(evidence):
                        return {
                            **summary,
                            "systems": {system: evidence},
                            "passed": evidence.get("acceptance", {}).get("passed", False),
                            "status": "cached",
                            "evidence_path": str(path),
                            "inference_performed": False,
                        }
            return self._missing("regression", system=system, requested_count=self.actual_count)
        from .cli import run_compare

        args = argparse.Namespace(
            systems=[system],
            count=self.actual_count,
            device=self.config.device,
            timeout=self.config.timeout_seconds,
            output=str(self._new_output(f"compare-{system}")),
        )
        result, code = run_compare(args, self.bundle)
        return {
            **result,
            "status": "completed" if code == 0 else "failed",
            "exit_code": code,
            "inference_requested": True,
            "inference_performed": any(
                row.get("comparison", {}).get("example_count", row.get("completed_requests", 0)) > 0
                for row in result["systems"].values()
            ),
        }

    def predict(self, system: str, utterance: str) -> dict[str, Any]:
        self._system(system)
        if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 4096:
            raise ValueError("utterance must contain 1–4096 characters of non-whitespace text")
        if self.config.mode == "replay":
            # No frozen CUDA answer is presented as a fresh local prediction.
            reports = checked_root(self.project / "reports")
            candidates = sorted(
                reports.rglob(f"prediction-{system}-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            for path in candidates:
                checked_path(reports, path.relative_to(reports).as_posix())
                saved = _read_json(path)
                if (
                    saved.get("system") == system
                    and saved.get("utterance") == utterance
                    and self._device_matches(saved)
                ):
                    return {
                        **saved,
                        "recorded_status": saved.get("status"),
                        "status": "cached",
                        "evidence_path": str(path),
                        "inference_performed": False,
                    }
            return self._missing("single prediction", system=system, utterance=utterance)
        from .cli import save_json
        from .inference import FrozenEngine

        path = self._new_output(f"prediction-{system}").with_suffix(".json")
        try:
            with FrozenEngine(self.bundle, system, self.config.device) as engine:
                result = engine.predict(utterance, timeout_seconds=self.config.timeout_seconds)
            valid = result["schema_valid"] is True
            result = {
                **result,
                "status": "completed" if valid else "invalid_output",
                "exit_code": 0 if valid else 2,
                "inference_performed": True,
                "semantic_correctness": "not_evaluated; no target supplied",
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "inference_requested": True,
                "inference_performed": None,
                "exit_code": 1,
            }
        result.update(system=system, utterance=utterance, output=str(path))
        save_json(path, result)
        return result

    def benchmark(self, system: str = "qlora") -> dict[str, Any]:
        self._system(system)
        if self.config.mode == "replay":
            for path, summary in reversed(self._saved_summaries()):
                measurements = summary.get("measurements", [])
                request_ids = {str(row["id"]) for row in measurements}
                repetitions = {row["repetition"] for row in measurements}
                if (
                    summary.get("identity", {}).get("system") == system
                    and self._device_matches(summary)
                    and "p50_seconds" in summary
                    and len(request_ids) == self.config.benchmark_requests
                    and repetitions == set(range(self.config.benchmark_repetitions))
                    and len(measurements)
                    == self.config.benchmark_requests * self.config.benchmark_repetitions
                ):
                    return {
                        **summary,
                        "status": "cached",
                        "evidence_path": str(path),
                        "inference_performed": False,
                    }
            return self._missing("benchmark", system=system)
        from .cli import run_benchmark, save_json

        output = self._new_output(f"benchmark-{system}")
        args = argparse.Namespace(
            system=system,
            requests=self.config.benchmark_requests,
            repetitions=self.config.benchmark_repetitions,
            device=self.config.device,
            timeout=self.config.timeout_seconds,
            output=str(output),
        )
        try:
            result = run_benchmark(args, self.bundle)
            return {**result, "status": "completed", "inference_performed": True}
        except Exception as exc:
            result = {
                "status": "failed",
                "system": system,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output": str(output),
                "inference_requested": True,
                "inference_performed": None,
            }
            save_json(output / "failure.json", result)
            return result

    def report(self, results: dict[str, Any], *, write: bool = False) -> dict[str, Any]:
        """Compose only supplied results; save to a fresh file on explicit request."""
        lines = [
            "# Structured request reliability — notebook session",
            "",
            f"Mode: **{self.config.mode}**. Preset: **{self.config.preset}** "
            f"({self.actual_count} reused test requests).",
            "",
            "Cached source scores, local subset checks and fresh inference are separate evidence. "
            "A subset pass does not establish full-test, new-request "
            "or general hardware equivalence.",
            "",
        ]
        for label, result in results.items():
            lines.extend([f"## {label}", ""])
            if not isinstance(result, dict):
                lines.extend([str(result), ""])
                continue
            status = result.get("status", "recorded")
            lines.extend([f"Status: **{status}**.", ""])
            for key in ("evidence_path", "output", "reason", "error", "qualification"):
                if key in result:
                    lines.extend([f"{key.replace('_', ' ').capitalize()}: {result[key]}", ""])
            for system, evidence in result.get("systems", {}).items():
                comparison = evidence.get("comparison", {})
                if comparison:
                    lines.extend(
                        [
                            f"- {system}: {comparison['example_count']} requests; "
                            f"gate passed: {evidence.get('acceptance', {}).get('passed')}; "
                            f"text agreement: {comparison.get('text_agreement', {}).get('rate')}; "
                            "entire response correct: "
                            f"{comparison.get('candidate_metrics', {}).get('joint_exact_match')}.",
                            "",
                        ]
                    )
                elif evidence.get("error"):
                    lines.extend([f"- {system}: failed — {evidence['error']}", ""])
        markdown = "\n".join(lines)
        path = None
        if write:
            from .cli import save_json

            path = self._new_output("report").with_suffix(".md")
            with path.open("x", encoding="utf-8") as handle:
                handle.write(markdown)
            save_json(path.with_suffix(".json"), results)
        return {"markdown": markdown, "path": str(path) if path else None}
