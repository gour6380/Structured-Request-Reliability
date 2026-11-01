"""Public benchmark evidence rejects mismatched runs and excludes private payloads."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "public_export", Path(__file__).resolve().parents[1] / "scripts/export_public_artifacts.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def evidence():
    ids = [str(value) for value in range(8)]
    manifest = {"config": {"model_id": "fixture", "model_revision": "frozen"}}
    benchmark = {
        "identity": {
            "system": "qlora",
            "device": "mps",
            "compute_dtype": "torch.bfloat16",
            "manifest_sha256": "digest",
            "model_id": "fixture",
            "model_revision": "frozen",
            "python": "3.13.11",
            "packages": {
                name: "1.0"
                for name in ("torch", "transformers", "peft", "bitsandbytes", "safetensors")
            },
            "private_path": "/Users/private/model",
        },
        "measured_requests": 24,
        "warmup_requests": 2,
        "model_load_seconds": 1.0,
        "first_request_seconds": 2.0,
        "p50_seconds": 12.5,
        "p95_seconds": 23.0,
        "requests_per_second": 24 / 300,
        "generated_tokens_per_second": 48 / 300,
        "output": "/Users/private/reports",
        "utterance": "private request",
        "error": "private error",
        "measurements": [
            {
                "id": ids[index % 8],
                "repetition": index // 8,
                "seconds": float(index + 1),
                "generated_tokens": 2,
                "schema_valid": index != 0,
                "generated_token_ids": [123, 456],
                "prediction_text": "private output",
                "memory": {
                    "rss_bytes": 10,
                    "mps_allocated_bytes": 20,
                    "mps_driver_bytes": 30,
                    "system_swap_used_bytes": 40,
                },
            }
            for index in range(24)
        ],
    }
    preflight = {
        "bundle": {"manifest_sha256": "digest"},
        "system_swap_used_bytes": 40,
        "platform": "macOS-arm64",
        "machine": "arm64",
        "memory_total_bytes": 16 * 1024**3,
        "executable": "/Users/private/.venv/bin/python",
        "prefix": "/Users/private/.venv",
    }
    return benchmark, preflight, ids, manifest, "digest"


def test_public_benchmark_excludes_private_payloads_and_keeps_invalid_denominator(evidence):
    result = exporter.benchmark_results(*evidence)
    encoded = json.dumps(result)
    assert result["schema_valid_count"] == 23
    assert result["schema_invalid_count"] == 1
    assert result["measured_requests"] == 24
    assert result["p50_seconds"] == 12.5
    assert result["p95_seconds"] == 23.0
    assert result["total_timed_seconds"] == 300.0
    for forbidden in (
        "/Users/",
        "private request",
        "private output",
        "private error",
        "generated_token_ids",
        "prediction_text",
        "measurements",
        '"output"',
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize("change", ["manifest", "device", "scope", "order", "aggregate"])
def test_public_benchmark_rejects_unreviewed_identity_scope_or_metrics(evidence, change):
    benchmark = evidence[0]
    if change == "manifest":
        benchmark["identity"]["manifest_sha256"] = "different"
    elif change == "device":
        benchmark["identity"]["device"] = "cpu"
    elif change == "scope":
        benchmark["measurements"].pop()
    elif change == "order":
        benchmark["measurements"][0]["id"] = "unexpected"
    else:
        benchmark["p95_seconds"] += 1
    with pytest.raises(ValueError):
        exporter.benchmark_results(*evidence)
