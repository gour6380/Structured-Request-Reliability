"""Notebook replay, opt-in delegation and evidence preservation without model loads."""

import json
from pathlib import Path

import pytest

from request_reliability import cli, inference, workflow
from request_reliability.workflow import NotebookConfig, NotebookSession


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "structured-request-reliability"
    files = {
        "pyproject.toml": '[project]\nname = "structured-request-reliability"\n',
        "src/request_reliability/__init__.py": '"""Fixture package."""\n',
        "src/request_reliability/example.py": "value = 1\n",
        "configs/regression.json": '{"case_count": 8}',
        "uv.lock": "version = 1\n",
        "data/frozen/manifest.json": '{"schema_version": 1}',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    reports = root / "reports"
    reports.mkdir()
    (reports / "reference-audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "inference_performed": False,
                "systems": {
                    "qlora": {
                        "metrics": {
                            "example_count": 2974,
                            "joint_exact_match": 0.72,
                            "schema_valid_rate": 0.99,
                            "intent_accuracy": 0.8,
                            "slot_f1": 0.8,
                        }
                    }
                },
            }
        )
    )
    saved = reports / "old-smoke"
    saved.mkdir()
    (saved / "summary.json").write_text(
        json.dumps(
            {
                "requested_count": 8,
                "passed": False,
                "systems": {
                    "qlora": {
                        "comparison": {
                            "example_count": 8,
                            "text_agreement": {"rate": 1.0},
                            "candidate_metrics": {"joint_exact_match": 0.5},
                        },
                        "acceptance": {"passed": True},
                        "identity": {"device": "mps"},
                    },
                    "lora": {
                        "status": "failed",
                        "error": "Memory guard stopped loading",
                        "completed_requests": 0,
                        "acceptance": {"passed": False},
                    },
                },
            }
        )
    )
    return root


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_replay_never_calls_models_or_creates_outputs(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not load or generate")

    monkeypatch.setattr(inference, "FrozenEngine", forbidden)
    monkeypatch.setattr(cli, "run_compare", forbidden)
    monkeypatch.setattr(cli, "run_benchmark", forbidden)
    before = snapshot(project)
    session = NotebookSession(project)
    compared = session.compare("qlora")
    assert compared["status"] == "cached"
    assert compared["passed"] is True  # Sibling LoRA failure must not mark QLoRA failed.
    assert compared["inference_performed"] is False
    assert set(compared["systems"]) == {"qlora"}
    assert session.compare("lora")["systems"]["lora"]["status"] == "failed"
    assert session.compare("base")["status"] == "missing"
    assert session.benchmark()["status"] == "missing"
    assert session.predict("qlora", "hello")["status"] == "missing"
    assert session.comparison_table()[0]["requests"] == 2974
    assert session.comparison_table()[0]["entire_response_correct"] == 0.72
    assert session.status_table()[0]["completed"] == 8
    assert session.status_table()[0]["entire_response_correct"] == 0.5
    assert session.report({"comparison": compared})["path"] is None
    assert snapshot(project) == before


def test_full_replay_does_not_present_smoke_as_full_evidence(project):
    session = NotebookSession(project, NotebookConfig(preset="full"))
    result = session.compare()
    assert result["status"] == "missing"
    assert result["requested_count"] == 2974


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "train"},
        {"device": "metal"},
        {"preset": "full2974"},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"benchmark_requests": 101},
        {"benchmark_requests": True},
        {"benchmark_repetitions": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        NotebookConfig(**kwargs)


def test_project_path_must_be_week9_not_week8(project):
    (project / "pyproject.toml").write_text('[project]\nname = "structured-request-adaptation"')
    with pytest.raises(ValueError, match="structured-request-reliability"):
        NotebookSession(project)


def test_live_compare_delegates_one_system_and_records_provenance(project, monkeypatch):
    before = snapshot(project)
    calls = []

    def fake_compare(args, bundle):
        calls.append((args, bundle))
        Path(args.output).mkdir(exist_ok=False)
        return {"systems": {args.systems[0]: {"comparison": {"example_count": args.count}}}}, 0

    monkeypatch.setattr(cli, "run_compare", fake_compare)
    session = NotebookSession(project, NotebookConfig(mode="live", device="mps"))
    first = session.compare("qlora")
    session.compare("lora")
    assert first["status"] == "completed"
    assert first["inference_performed"] is True
    assert calls[0][0].systems == ["qlora"]
    assert calls[0][0].count == 8
    assert calls[0][0].device == "mps"
    assert calls[0][1] == project / "data/frozen"
    assert calls[0][0].output != calls[1][0].output
    metadata = json.loads((session.results_dir() / "session.json").read_text())
    assert metadata["config"]["mode"] == "live"
    assert metadata["config"]["actual_count"] == 8
    assert set(metadata["source_sha256"]) == {
        "src/request_reliability/__init__.py",
        "src/request_reliability/example.py",
        "pyproject.toml",
        "uv.lock",
        "configs/regression.json",
    }
    for relative, content in before.items():
        assert (project / relative).read_bytes() == content


def test_live_benchmark_keeps_request_and_repetition_bounds(project, monkeypatch):
    calls = []

    def fake_benchmark(args, bundle):
        calls.append(args)
        return {"measured_requests": args.requests * args.repetitions, "output": args.output}

    monkeypatch.setattr(cli, "run_benchmark", fake_benchmark)
    session = NotebookSession(
        project,
        NotebookConfig(
            mode="live",
            benchmark_requests=3,
            benchmark_repetitions=2,
        ),
    )
    assert session.benchmark("qlora")["measured_requests"] == 6
    assert calls[0].system == "qlora"
    assert calls[0].requests == 3
    assert calls[0].repetitions == 2


def test_replay_benchmark_matches_configured_scope(project):
    folder = project / "reports/old-benchmark"
    folder.mkdir()
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "identity": {"system": "qlora", "device": "mps"},
                "p50_seconds": 1.0,
                "measurements": [{"id": "a", "repetition": 0}, {"id": "b", "repetition": 0}],
            }
        )
    )
    assert NotebookSession(project).benchmark()["status"] == "missing"
    session = NotebookSession(
        project,
        NotebookConfig(
            benchmark_requests=2,
            benchmark_repetitions=1,
        ),
    )
    result = session.benchmark()
    assert result["status"] == "cached"
    assert result["identity"]["device"] == "mps"
    assert result["inference_performed"] is False


def test_replay_single_prediction_requires_exact_saved_request(project):
    path = project / "reports/prediction-qlora-example.json"
    path.write_text(
        json.dumps(
            {
                "system": "qlora",
                "utterance": "hello",
                "text": "{}",
                "status": "completed",
                "inference_performed": True,
            }
        )
    )
    session = NotebookSession(project)
    assert session.predict("qlora", "hello")["status"] == "cached"
    assert session.predict("qlora", "hello")["inference_performed"] is False
    assert session.predict("qlora", "different request")["status"] == "missing"


@pytest.mark.parametrize(
    "device, expected",
    [("auto", "cached"), ("mps", "cached"), ("cpu", "missing"), ("cuda", "missing")],
)
def test_replay_respects_explicit_device_for_all_live_evidence(project, device, expected):
    prediction = project / "reports/prediction-qlora-example.json"
    prediction.write_text(
        json.dumps(
            {
                "system": "qlora",
                "utterance": "hello",
                "identity": {"device": "mps"},
                "status": "completed",
                "schema_valid": True,
            }
        )
    )
    benchmark = project / "reports/old-benchmark"
    benchmark.mkdir()
    (benchmark / "summary.json").write_text(
        json.dumps(
            {
                "identity": {"system": "qlora", "device": "mps"},
                "p50_seconds": 1.0,
                "measurements": [{"id": "a", "repetition": 0}],
            }
        )
    )
    session = NotebookSession(
        project,
        NotebookConfig(
            device=device,
            benchmark_requests=1,
            benchmark_repetitions=1,
        ),
    )
    assert session.compare("qlora")["status"] == expected
    assert session.predict("qlora", "hello")["status"] == expected
    assert session.benchmark("qlora")["status"] == expected


def test_explicit_device_replay_does_not_infer_unknown_device_from_failure(project):
    session = NotebookSession(project, NotebookConfig(device="mps"))
    assert session.compare("lora")["status"] == "missing"
    # Historical failed attempts remain visible with their actual identity limitation.
    assert session.status_table()[1]["device"] == "not recorded"


@pytest.mark.parametrize(
    "valid, status, code", [(True, "completed", 0), (False, "invalid_output", 2)]
)
def test_prediction_status_distinguishes_schema_validity_without_claiming_accuracy(
    project,
    monkeypatch,
    valid,
    status,
    code,
):
    class Engine:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def predict(self, utterance, *, timeout_seconds):
            return {
                "text": "output",
                "schema_valid": valid,
                "failure_reasons": [] if valid else ["invalid_json"],
            }

    monkeypatch.setattr(inference, "FrozenEngine", Engine)
    result = NotebookSession(project, NotebookConfig(mode="live")).predict("qlora", "hello")
    assert result["status"] == status
    assert result["exit_code"] == code
    assert result["schema_valid"] is valid
    assert result["inference_performed"] is True
    assert result["semantic_correctness"].startswith("not_evaluated")


def test_notebook_provenance_ignores_execution_but_detects_source_change(project):
    notebook = project / "notebooks/structured_request_reliability.ipynb"
    notebook.parent.mkdir()
    contents = {
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "source": ["MODE = 'replay'\n"],
                "outputs": [],
                "metadata": {},
                "execution_count": None,
            }
        ],
    }
    notebook.write_text(json.dumps(contents))

    def saved_hash():
        session = NotebookSession(project)
        metadata = json.loads((session.results_dir() / "session.json").read_text())
        return metadata["notebook_source_sha256"][notebook.relative_to(project).as_posix()]

    first = saved_hash()
    contents["metadata"] = {"widgets": {"runtime": "changed"}}
    contents["cells"][0].update(
        source="MODE = 'replay'\n",
        execution_count=1,
        outputs=[{"output_type": "stream", "name": "stdout", "text": "executed"}],
        metadata={"execution": {"iopub.status.busy": "a new timestamp"}},
    )
    notebook.write_text(json.dumps(contents, indent=2))
    assert saved_hash() == first
    contents["cells"][0]["source"] = "MODE = 'live'\n"
    notebook.write_text(json.dumps(contents))
    assert saved_hash() != first


def test_notebook_provenance_includes_nested_public_copy_and_excludes_checkpoints(project):
    notebook_paths = [
        "notebooks/structured_request_reliability.ipynb",
        "notebooks/public/structured_request_reliability.ipynb",
    ]
    content = {"cells": [{"cell_type": "code", "source": "MODE = 'replay'\n"}]}
    for relative in notebook_paths:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    checkpoint = project / "notebooks/public/.ipynb_checkpoints/notebook-checkpoint.ipynb"
    checkpoint.parent.mkdir()
    checkpoint.write_text("notebook checkpoints are ignored even when incomplete")

    session = NotebookSession(project)
    metadata = json.loads((session.results_dir() / "session.json").read_text())
    recorded = metadata["notebook_source_sha256"]
    assert set(recorded) == set(notebook_paths)
    assert recorded[notebook_paths[0]] == recorded[notebook_paths[1]]

    content["cells"][0]["source"] = "MODE = 'live'\n"
    (project / notebook_paths[1]).write_text(json.dumps(content))
    next_session = NotebookSession(project)
    updated = json.loads((next_session.results_dir() / "session.json").read_text())
    assert updated["notebook_source_sha256"][notebook_paths[1]] != recorded[notebook_paths[1]]
    assert updated["notebook_source_sha256"][notebook_paths[0]] == recorded[notebook_paths[0]]


def test_live_prediction_releases_engine_even_on_error(project, monkeypatch):
    events = []

    class Engine:
        def __init__(self, bundle, system, device):
            events.append((system, device))

        def __enter__(self):
            events.append("enter")
            return self

        def predict(self, utterance, *, timeout_seconds):
            events.append((utterance, timeout_seconds))
            raise TimeoutError("bounded request timed out")

        def __exit__(self, *args):
            events.append("close")

    monkeypatch.setattr(inference, "FrozenEngine", Engine)
    session = NotebookSession(project, NotebookConfig(mode="live", timeout_seconds=5))
    result = session.predict("qlora", "hello")
    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutError"
    assert events == [("qlora", "auto"), "enter", ("hello", 5), "close"]
    assert json.loads(Path(result["output"]).read_text())["status"] == "failed"


def test_hardware_memory_failure_is_unavailable_not_zero(project, monkeypatch):
    import psutil

    def denied():
        raise PermissionError("sandbox denied sysctl")

    monkeypatch.setattr(psutil, "virtual_memory", denied)
    monkeypatch.setattr(inference, "FrozenEngine", lambda *args: pytest.fail("model loaded"))
    result = NotebookSession(project).hardware()
    assert result["memory_status"] == "unavailable"
    assert "available_memory_gib" not in result
    assert result["model_loaded"] is False


def test_source_audit_recompute_delegates_without_writing(project, monkeypatch):
    from request_reliability import regression

    before = snapshot(project)
    monkeypatch.setattr(
        regression,
        "audit_reference",
        lambda bundle: {
            "status": "passed",
            "inference_performed": False,
            "example_count": 2974,
        },
    )
    session = NotebookSession(project)
    assert session.reference_audit(recompute=True)["example_count"] == 2974
    assert snapshot(project) == before


def test_explicit_report_writes_never_overwrite_prior_evidence(project):
    before = snapshot(project)
    session = NotebookSession(project)
    first = session.report({"lora": session.compare("lora")}, write=True)
    second = session.report({"lora": session.compare("lora")}, write=True)
    assert first["path"] != second["path"]
    assert "Memory guard stopped loading" in Path(first["path"]).read_text()
    other_session = NotebookSession(project)
    assert other_session.results_dir() != session.results_dir()
    for relative, content in before.items():
        assert (project / relative).read_bytes() == content


def test_inventory_and_verification_use_shared_artifact_code(project, monkeypatch):
    from request_reliability import artifacts

    monkeypatch.setattr(
        workflow,
        "load_manifest",
        lambda bundle: {
            "files": {"example.bin": {"bytes": 4, "sha256": "a" * 64}},
        },
    )
    monkeypatch.setattr(
        artifacts,
        "verify_bundle",
        lambda bundle, full: {
            "valid": True,
            "full_hash_verification": full,
        },
    )
    session = NotebookSession(project)
    assert session.inventory() == [{"artifact": "example.bin", "bytes": 4, "sha256": "a" * 64}]
    assert session.verify()["full_hash_verification"] is True
    assert session.verify(full=False)["full_hash_verification"] is False
