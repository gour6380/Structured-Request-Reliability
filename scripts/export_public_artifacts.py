"""Export selected aggregate evidence and a clean notebook without running models.

Run with this project's .venv/bin/python. Raw evidence and the working notebook
are read only. Every exported fact is selected explicitly; source dictionaries
are never copied wholesale into the public results.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("base", "quantized_base", "lora", "qlora")
WORKING_NOTEBOOK = "notebooks/structured_request_reliability.ipynb"
BENCHMARK_SOURCE = "reports/benchmark-qlora-validation-20261101T084410Z/summary.json"
BENCHMARK_PREFLIGHT = "verification/bounded-benchmark-2026-11-01-preflight.json"
BENCHMARK_TESTS = "verification/bounded-benchmark-2026-11-01-tests.xml"
VALIDATION_SOURCE = "data/frozen/examples/validation.jsonl"
DATA_AUDIT_SOURCE = "data/frozen/provenance/data.json"
SOURCES = (
    "reports/reference-audit.json",
    "reports/mps-reference-smoke/qlora/report.json",
    "reports/mps-reference-smoke/summary.json",
    "reports/mps-lora-isolated/summary.json",
    "reports/local-api-check.json",
    "verification/notebook-execution.json",
    "verification/notebook-tests.xml",
    "data/frozen/manifest.json",
    WORKING_NOTEBOOK,
    BENCHMARK_SOURCE,
    BENCHMARK_PREFLIGHT,
    VALIDATION_SOURCE,
    DATA_AUDIT_SOURCE,
)
METRICS = (
    "example_count",
    "joint_exact_match",
    "schema_valid_rate",
    "intent_accuracy",
    "intent_macro_f1",
    "slot_f1",
)
COUNTS = (
    "json_valid",
    "structure_valid",
    "schema_valid",
    "joint_exact_match",
    "intent_correct",
    "slot_tp",
    "slot_fp",
    "slot_fn",
    "missing_slots",
    "extra_slots",
    "wrong_values",
)
FAILURES = (
    "invalid_json",
    "invalid_structure",
    "unknown_intent",
    "unknown_slot_type",
    "wrong_intent",
    "missing_slots",
    "extra_slots",
    "wrong_values",
)


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def selected(value: dict, names: tuple[str, ...]) -> dict:
    return {name: value[name] for name in names}


def benchmark_results(
    benchmark: dict,
    preflight: dict,
    validation_ids: list[str],
    manifest: dict,
    manifest_sha256: str,
) -> dict:
    """Validate the reviewed run, then expose aggregates without raw requests or paths."""
    identity = benchmark["identity"]
    if (
        identity["system"] != "qlora"
        or identity["device"] != "mps"
        or identity["compute_dtype"] != "torch.bfloat16"
        or identity["manifest_sha256"] != manifest_sha256
        or preflight["bundle"]["manifest_sha256"] != manifest_sha256
        or any(identity[key] != manifest["config"][key] for key in ("model_id", "model_revision"))
    ):
        raise ValueError("Benchmark identity differs from the reviewed QLoRA/MPS frozen protocol")
    measurements = benchmark["measurements"]
    expected = [(value, repetition) for repetition in range(3) for value in validation_ids]
    if (
        len(validation_ids) != 8
        or len(set(validation_ids)) != 8
        or benchmark["measured_requests"] != 24
        or benchmark["warmup_requests"] != 2
        or len(measurements) != 24
        or [(str(row["id"]), row["repetition"]) for row in measurements] != expected
        or any(type(row["repetition"]) is not int for row in measurements)
    ):
        raise ValueError("Benchmark scope differs from the first eight validation inputs ×3")
    times = []
    for row in measurements:
        duration = row["seconds"]
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration <= 0
            or type(row["generated_tokens"]) is not int
            or row["generated_tokens"] <= 0
            or type(row["schema_valid"]) is not bool
        ):
            raise ValueError("Benchmark contains invalid measurement values")
        times.append(duration)
    total_seconds = sum(times)
    total_tokens = sum(row["generated_tokens"] for row in measurements)
    computed = {
        "p50_seconds": statistics.median(times),
        "p95_seconds": sorted(times)[math.ceil(len(times) * 0.95) - 1],
        "requests_per_second": len(times) / total_seconds,
        "generated_tokens_per_second": total_tokens / total_seconds,
    }
    for name, value in computed.items():
        if not math.isclose(benchmark[name], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Benchmark aggregate disagrees with measurements: {name}")
    for name in ("model_load_seconds", "first_request_seconds"):
        if not math.isfinite(benchmark[name]) or benchmark[name] <= 0:
            raise ValueError(f"Invalid benchmark duration: {name}")
    memory = {}
    for name in (
        "rss_bytes",
        "mps_allocated_bytes",
        "mps_driver_bytes",
        "system_swap_used_bytes",
    ):
        values = [row["memory"][name] for row in measurements]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"Invalid benchmark memory counter: {name}")
        memory[name] = {"minimum_bytes": min(values), "maximum_bytes": max(values)}
    return {
        "scope": (
            "First eight frozen validation inputs, sequential MPS inference, three repetitions"
        ),
        "distinct_validation_inputs": 8,
        "repetitions": 3,
        "measured_requests": len(measurements),
        "separate_first_requests": 1,
        "excluded_warmup_requests": 2,
        "model_load_seconds": benchmark["model_load_seconds"],
        "first_request_seconds": benchmark["first_request_seconds"],
        **computed,
        "total_timed_seconds": total_seconds,
        "total_generated_tokens": total_tokens,
        "schema_valid_count": sum(row["schema_valid"] for row in measurements),
        "schema_invalid_count": sum(not row["schema_valid"] for row in measurements),
        "memory_point_sample_ranges": memory,
        "preflight_system_swap_used_bytes": preflight["system_swap_used_bytes"],
        "identity": selected(
            identity,
            (
                "system",
                "model_id",
                "model_revision",
                "manifest_sha256",
                "device",
                "compute_dtype",
                "python",
            ),
        ),
        "packages": selected(
            identity["packages"],
            ("torch", "transformers", "peft", "bitsandbytes", "safetensors"),
        ),
        "hardware": selected(preflight, ("platform", "machine", "memory_total_bytes")),
        "timing_boundary": (
            "Tokenization, transfer, generation, synchronization, decoding and schema validation; "
            "model loading and subsequent memory snapshots excluded."
        ),
        "qualifications": [
            "Eight repeated inputs are not 24 independent examples or a population-tail estimate.",
            "Schema validity is not task correctness; this run computes no accuracy score.",
            "Memory counters overlap; point samples are not peaks or a continuous paging trace.",
            "OS disk caches were not flushed; load and first request are not storage-cold results.",
            "Throughput excludes load, warmups and orchestration; not concurrent serving capacity.",
            "Generated-token counts include terminal tokens.",
            "No native baseline, LoRA or CUDA performance comparison was measured.",
        ],
    }


def output_for(notebook: dict, source_fragment: str) -> dict:
    for cell in notebook["cells"]:
        if any(line.lstrip().startswith(source_fragment) for line in cell["source"]):
            for output in cell.get("outputs", []):
                if "application/json" in output.get("data", {}):
                    return output["data"]["application/json"]
    raise ValueError(f"Missing saved JSON output for {source_fragment}")


def evidence_status(value: dict) -> dict:
    return {
        name: value[name]
        for name in ("status", "recorded_status", "passed", "schema_valid", "inference_performed")
        if name in value
    }


def latest_notebook_audit(notebook: dict) -> dict:
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell["execution_count"] for cell in code]
    settings = output_for(notebook, "config = NotebookConfig")
    stream = "".join(
        "".join(output.get("text", []))
        for cell in code
        for output in cell.get("outputs", [])
        if output["output_type"] == "stream"
    )
    adaptation_kernel = "structured-request-adaptation/.venv/bin/python" in stream
    own_kernel = str(ROOT / ".venv/bin/python") in stream
    kernel_role = (
        "adaptation_project_python"
        if adaptation_kernel
        else ("reliability_project_python" if own_kernel else "unverified")
    )
    baselines = output_for(notebook, "baseline_comparisons = {}")
    current = {
        "mode": settings["mode"],
        "preset": settings["preset"],
        "code_cells": len(code),
        "completed_code_cells": sum(value is not None for value in counts),
        "sequential_execution_counts": counts == list(range(1, len(code) + 1)),
        "saved_error_outputs": sum(
            output["output_type"] == "error" for cell in code for output in cell.get("outputs", [])
        ),
        "kernel_role": kernel_role,
        "python_version": notebook["metadata"]["language_info"]["version"],
        "reliability_environment_verified_by_this_execution": own_kernel and not adaptation_kernel,
        "inference_performed": False,
        "saved_outcomes": {
            "qlora": evidence_status(output_for(notebook, "qlora_comparison = session.compare")),
            "lora": evidence_status(output_for(notebook, "lora_comparison = session.compare")),
            "base": evidence_status(baselines["base"]),
            "quantized_base": evidence_status(baselines["quantized_base"]),
            "prediction": evidence_status(output_for(notebook, "prediction = session.predict")),
            "benchmark": evidence_status(output_for(notebook, "benchmark = session.benchmark")),
            "api": evidence_status(output_for(notebook, "api_check =")),
        },
        "notebook_session_report_directories": len(list((ROOT / "reports").glob("notebook-*"))),
        "qualification": (
            "The latest saved notebook is a completed replay using the adaptation project "
            "Python kernel. It is not an isolated reliability environment check or native run."
            if adaptation_kernel
            else "Saved replay only; not new native inference or a performance benchmark."
        ),
    }
    if settings["mode"] != "replay" or any(
        outcome.get("inference_performed") is True for outcome in current["saved_outcomes"].values()
    ):
        raise ValueError("This public export expects the saved replay notebook, not live outputs")
    return current


def build_results(notebook: dict) -> dict:
    reference = read(SOURCES[0])
    native = read(SOURCES[1])
    api = read("reports/local-api-check.json")
    execution = read("verification/notebook-execution.json")
    manifest = read("data/frozen/manifest.json")
    data_audit = read(DATA_AUDIT_SOURCE)
    comparisons = native["comparison"]
    lora_attempts = []
    for source in (SOURCES[2], SOURCES[3]):
        summary = read(source)
        lora = summary["systems"]["lora"]
        if lora["error"] != "System swap grew by more than 128 MiB; stop this run":
            raise ValueError("Review the new LoRA failure before exposing its error publicly")
        lora_attempts.append(
            {
                "source": source,
                "status": lora["status"],
                "requested_requests": summary["requested_count"],
                "completed_requests": lora["completed_requests"],
                "passed": lora["acceptance"]["passed"],
                "error_type": lora["error_type"],
                "reason_code": "swap_growth_guard",
                "swap_growth_limit_mib": 128,
            }
        )
    suites = list(ET.parse(ROOT / "verification/notebook-tests.xml").getroot().iter("testsuite"))
    junit = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "errors", "failures", "skipped")
    }
    recorded_tests = execution["software_tests"]
    if junit["tests"] != recorded_tests["passed"] + recorded_tests["subtests_passed"]:
        raise ValueError("JUnit count disagrees with the notebook verification receipt")
    identity = selected(
        native["identity"],
        (
            "model_id",
            "model_revision",
            "manifest_sha256",
            "device",
            "compute_dtype",
            "source_device",
        ),
    )
    if identity["manifest_sha256"] != digest("data/frozen/manifest.json"):
        raise ValueError("Native result does not identify the supplied frozen manifest")
    validation_ids = [
        str(json.loads(line)["id"])
        for line in (ROOT / VALIDATION_SOURCE).read_text().splitlines()
        if line.strip()
    ][:8]
    return {
        "schema_version": 1,
        "source_reference": {
            "status": reference["status"],
            "scope": "Recomputed saved CUDA predictions; reused test data; no new inference",
            "examples_per_system": reference["example_count"],
            "systems": {
                system: {
                    **selected(reference["systems"][system]["metrics"], METRICS),
                    "counts": selected(reference["systems"][system]["metrics"]["counts"], COUNTS),
                    "failure_counts": {
                        key: reference["systems"][system]["metrics"]["failure_counts"][key]
                        for key in FAILURES
                        if key in reference["systems"][system]["metrics"]["failure_counts"]
                    },
                    "whole_record_correct_count": reference["systems"][system]["metrics"]["counts"][
                        "joint_exact_match"
                    ],
                    "schema_valid_count": reference["systems"][system]["metrics"]["counts"][
                        "schema_valid"
                    ],
                }
                for system in SYSTEMS
            },
            "novel_input_subset": {
                "scope": (
                    "Saved CUDA test responses whose normalized input is absent from both "
                    "train and validation; still inspected source-experiment data"
                ),
                "examples_per_system": data_audit["novel_input_counts"]["test"],
                "systems": {
                    system: {
                        **selected(
                            reference["systems"][system]["metrics"]["novel_input_subset"],
                            METRICS,
                        ),
                        "whole_record_correct_count": reference["systems"][system]["metrics"][
                            "novel_input_subset"
                        ]["counts"]["joint_exact_match"],
                        "schema_valid_count": reference["systems"][system]["metrics"][
                            "novel_input_subset"
                        ]["counts"]["schema_valid"],
                    }
                    for system in SYSTEMS
                },
            },
            "data_overlap_audit": {
                "normalization": "NFKC, casefold, collapse whitespace; audit only",
                "all_official_rows_retained": True,
                "duplicate_group_counts": {
                    kind: selected(
                        data_audit["duplicate_audit_counts"][kind],
                        (
                            "duplicate_group_count",
                            "cross_split_group_count",
                            "conflicting_group_count",
                        ),
                    )
                    for kind in ("exact", "normalized")
                },
                "test_input_overlap_count": data_audit["split_counts"]["test"]
                - data_audit["novel_input_counts"]["test"],
                "novel_test_input_count": data_audit["novel_input_counts"]["test"],
                "qualification": (
                    "Observed cross-split overlap; novel-input filtering does not make the "
                    "inspected test set a new untouched evaluation."
                ),
            },
        },
        "native_qlora": {
            "scope": "First eight requests in frozen test-file order; MPS regression smoke check",
            "example_count": comparisons["example_count"],
            "prompt_token_parity": native["prompt_token_parity"],
            "text_agreement": selected(
                comparisons["text_agreement"], ("equal", "compared", "rate")
            ),
            "token_agreement": selected(
                comparisons["token_agreement"], ("equal", "compared", "rate")
            ),
            "metrics": selected(comparisons["candidate_metrics"], METRICS),
            "whole_record_correct_count": comparisons["candidate_metrics"]["counts"][
                "joint_exact_match"
            ],
            "acceptance": selected(
                native["acceptance"],
                (
                    "passed",
                    "full_reference_evaluated",
                    "full_equivalence_demonstrated",
                ),
            ),
            "limitation": (
                "Eight matching outputs include four correct responses; "
                "full-test equivalence is untested."
            ),
        },
        "native_lora_attempts": lora_attempts,
        "native_qlora_benchmark": benchmark_results(
            read(BENCHMARK_SOURCE),
            read(BENCHMARK_PREFLIGHT),
            validation_ids,
            manifest,
            digest("data/frozen/manifest.json"),
        ),
        "api": {
            "scope": "Previously recorded in-process ASGI check; not a TCP or load test",
            **selected(
                api,
                (
                    "passed",
                    "health_status",
                    "invalid_input_status",
                    "prediction_statuses",
                    "response_matches_reference",
                    "repeat_parsed_output_equal",
                    "request_ids_distinct",
                    "closed_after_shutdown",
                ),
            ),
            "system": api["identity"]["system"],
            "device": api["identity"]["device"],
            "counters": selected(
                api["counters"],
                (
                    "accepted",
                    "succeeded",
                    "failed",
                    "timed_out",
                    "cancelled",
                    "busy_rejected",
                ),
            ),
        },
        "notebook_audit": {
            "latest_saved_execution": latest_notebook_audit(notebook),
            "earlier_project_environment_verification": {
                **selected(
                    execution,
                    (
                        "status",
                        "mode",
                        "cell_count",
                        "executed_code_cells",
                        "seconds",
                        "notebook_sha256",
                        "inference_performed",
                    ),
                ),
                "error_count": len(execution["errors"]),
                "notebook_hash_matches_latest_saved_copy": execution["notebook_sha256"]
                == digest(WORKING_NOTEBOOK),
                "qualification": (
                    "Earlier project-environment replay audit; a separate execution "
                    "from the latest saved adaptation-project-kernel replay."
                ),
            },
        },
        "software_validation": {
            "scope": "Recorded notebook-refactor test run; not a new test run by this exporter",
            "test_cases_passed": recorded_tests["passed"],
            "subtests_passed": recorded_tests["subtests_passed"],
            "junit": junit,
            "ruff": recorded_tests["ruff"],
            "dependency_check": recorded_tests["dependency_check"],
        },
        "protocol": {
            "model_identity": identity,
            "checkpoint": manifest["origin"]["checkpoint"],
            "epochs_completed": manifest["origin"]["epochs_completed"],
            **selected(
                manifest["config"],
                (
                    "locale",
                    "max_sequence_length",
                    "max_new_tokens",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                    "target_modules",
                    "evaluation_batch_size",
                    "seed",
                ),
            ),
            "training": selected(
                manifest["config"],
                (
                    "epochs",
                    "microbatch_size",
                    "gradient_accumulation",
                    "learning_rate",
                    "warmup_ratio",
                    "weight_decay",
                    "expected_split_counts",
                ),
            ),
            "generation": {
                **selected(
                    manifest["generation"],
                    (
                        "do_sample",
                        "num_beams",
                        "use_cache",
                        "decoding",
                        "padding",
                        "terminal_token_policy",
                    ),
                ),
                "inherited_configuration": selected(
                    manifest["generation"]["inherited_configuration"],
                    (
                        "bos_token_id",
                        "do_sample",
                        "eos_token_id",
                        "pad_token_id",
                        "repetition_penalty",
                        "temperature",
                        "top_k",
                        "top_p",
                        "transformers_version",
                    ),
                ),
                "qualification": (
                    "Explicit generation fields override inherited values. Sampling controls "
                    "are inactive with do_sample=false; EOS/PAD use the saved tokenizer."
                ),
            },
            "evaluation_batch_note": (
                "Source CUDA evaluation uses batch size 32; native smoke and API "
                "use one request at a time."
            ),
        },
        "provenance": {
            "sources": {source: digest(source) for source in SOURCES},
            "exporter": {
                "path": "scripts/export_public_artifacts.py",
                "sha256": digest("scripts/export_public_artifacts.py"),
            },
            "qualification": (
                "Relative source paths and SHA-256 only. Raw private source artifacts "
                "are not included in the public export."
            ),
        },
    }


def public_notebook(notebook: dict) -> dict:
    result = copy.deepcopy(notebook)
    settings = {}
    for cell in result["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            for node in ast.parse("".join(cell["source"])).body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    name = node.targets[0].id
                    if name in (
                        "MODE",
                        "INSTALL_DEPENDENCIES",
                        "RUN_LORA",
                        "RUN_BASELINES",
                        "RUN_BENCHMARKS",
                        "RUN_API_CHECK",
                        "WRITE_SESSION_REPORT",
                    ):
                        settings[name] = ast.literal_eval(node.value)
        else:
            cell["source"] = [
                line.replace("(../README.md)", "(../../README.md)").replace(
                    "(../reports/technical-report.md)", "(../../reports/technical-report.md)"
                )
                for line in cell["source"]
            ]
    if settings.get("MODE") != "replay" or any(
        value is not False for name, value in settings.items() if name != "MODE"
    ):
        raise ValueError(
            "Public notebook must preserve replay mode with every live/write/install flag false"
        )
    result["metadata"] = {
        "kernelspec": {
            "display_name": "Structured Request Reliability — Python 3.13",
            "language": "python",
            "name": "structured-request-reliability",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    return result


def render_chart(results: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = results["source_reference"]["systems"]
    x = list(range(len(SYSTEMS)))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    for offset, field, label, color in (
        (-0.19, "joint_exact_match", "Entire response correct", "#236B8E"),
        (0.19, "schema_valid_rate", "Schema valid", "#D79526"),
    ):
        bars = ax.bar(
            [value + offset for value in x],
            [100 * values[s][field] for s in SYSTEMS],
            width=0.38,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_xticks(x, ["Base", "NF4 base", "LoRA", "QLoRA"])
    ax.set(
        title="Original CUDA reference · 2,974 requests per system",
        ylabel="Requests (%)",
        ylim=(0, 116),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(
        destination,
        dpi=180,
        metadata={"Software": "Structured Request Reliability public exporter"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-only", action="store_true", help="Preserve the public notebook and chart"
    )
    args = parser.parse_args()
    notebook = read(WORKING_NOTEBOOK)
    results = build_results(notebook)
    latest_junit = "verification/github-tests.xml"
    if (ROOT / latest_junit).is_file():
        root = ET.parse(ROOT / latest_junit).getroot()
        suites = list(root.iter("testsuite"))
        totals = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "errors", "failures", "skipped")
        }
        tests = len(list(root.iter("testcase")))
        clean_pass = not any(totals[key] for key in ("errors", "failures", "skipped"))
        results["software_validation"]["latest_github_preparation_gate"] = {
            "scope": "Latest saved GitHub-preparation JUnit receipt; not rerun by this exporter",
            "status": "passed" if clean_pass else "not_all_passed",
            "test_cases_passed": tests if clean_pass else None,
            "subtests_passed": totals["tests"] - tests if clean_pass else None,
            "junit": totals,
        }
        results["provenance"]["sources"][latest_junit] = digest(latest_junit)
    if (ROOT / BENCHMARK_TESTS).is_file():
        root = ET.parse(ROOT / BENCHMARK_TESTS).getroot()
        suites = list(root.iter("testsuite"))
        totals = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "errors", "failures", "skipped")
        }
        tests = len(list(root.iter("testcase")))
        clean_pass = not any(totals[key] for key in ("errors", "failures", "skipped"))
        results["software_validation"]["pre_benchmark_gate"] = {
            "scope": "Saved pre-benchmark JUnit receipt; not rerun by this exporter",
            "status": "passed" if clean_pass else "not_all_passed",
            "test_cases_passed": tests if clean_pass else None,
            "subtests_passed": totals["tests"] - tests if clean_pass else None,
            "junit": totals,
        }
        results["provenance"]["sources"][BENCHMARK_TESTS] = digest(BENCHMARK_TESTS)
    clean = public_notebook(notebook)
    encoded = (
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    clean_encoded = json.dumps(clean, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    for forbidden in (
        "/Users/",
        "/private/",
        "file://",
        "prediction_text",
        "generated_token_ids",
        "utterance",
    ):
        if forbidden in encoded:
            raise ValueError(f"Public aggregate contains a disallowed field or path: {forbidden}")
    if "/Users/" in clean_encoded or "/private/" in clean_encoded:
        raise ValueError("Notebook source contains a private absolute path; review before export")
    public = ROOT / "reports/public"
    notebooks = ROOT / "notebooks/public"
    public.mkdir(parents=True, exist_ok=True)
    notebooks.mkdir(parents=True, exist_ok=True)
    (public / "results.json").write_text(encoded, encoding="utf-8")
    if args.results_only:
        print("Exported public aggregate results; preserved the public notebook and chart.")
        return
    (notebooks / "structured_request_reliability.ipynb").write_text(clean_encoded, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="request-reliability-chart-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        # Reuse the existing font inventory when available, without modifying it.
        for font_cache in (ROOT / ".cache/matplotlib").glob("fontlist-*.json"):
            shutil.copyfile(font_cache, Path(temporary) / font_cache.name)
        render_chart(results, public / "reference-scores.png")
    print("Exported public aggregate results, reference chart and clean replay notebook.")


if __name__ == "__main__":
    main()
