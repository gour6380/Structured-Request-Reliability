"""Explicit commands; imports never run a model or contact a remote service."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def output_directory(project: Path, kind: str, requested: str | None) -> Path:
    path = (
        Path(requested).resolve()
        if requested
        else project / "reports" / f"{kind}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_compare(args: argparse.Namespace, bundle: Path) -> tuple[dict[str, Any], int]:
    from .inference import FrozenEngine
    from .regression import build_regression_report, fixed_reference_cases

    project = bundle.parents[1]
    output = output_directory(project, "regression", args.output)
    cases = fixed_reference_cases(bundle, count=args.count)
    thresholds = json.loads((project / "configs" / "regression.json").read_text())
    thresholds["case_count"] = args.count
    summaries: dict[str, Any] = {}
    for system in args.systems:
        predictions: list[dict[str, Any]] = []
        try:
            with FrozenEngine(bundle, system, args.device) as engine:
                identity = dict(engine.identity)
                # Confirm prompt reconstruction before any generation.
                tokenized = {
                    str(row["id"]): row
                    for row in read_jsonl(bundle / "examples" / "tokenized_test.jsonl")
                }
                for case in cases:
                    prompt = engine.prompt_ids(case["utterance"])
                    if prompt != tokenized[str(case["id"])]["prompt_ids"]:
                        raise ValueError("Prompt tokens differ from frozen reference; stop")
                    result = engine.predict(case["utterance"], timeout_seconds=args.timeout)
                    predictions.append(
                        {
                            "id": str(case["id"]),
                            "utterance": case["utterance"],
                            "target": case["target"],
                            "novel_input": case.get("novel_input", False),
                            **result,
                        }
                    )
                    destination = output / system
                    destination.mkdir(exist_ok=True)
                    (destination / "predictions.jsonl").write_text(
                        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
                    )
                memory = engine.memory_snapshot()
            report = build_regression_report(bundle, predictions, system, thresholds)
            report.update(identity=identity, memory=memory, prompt_token_parity=True)
            summaries[system] = report
        except Exception as exc:
            summaries[system] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_requests": len(predictions),
                "acceptance": {"passed": False},
            }
        save_json(output / system / "report.json", summaries[system])
    passed = all(row.get("acceptance", {}).get("passed", False) for row in summaries.values())
    result = {
        "output": str(output),
        "kind": "frozen-reference regression; reused test data",
        "requested_count": args.count,
        "systems": summaries,
        "passed": passed,
        "scope": "This bounded replay does not demonstrate full-test or general equivalence.",
    }
    save_json(output / "summary.json", result)
    return result, 0 if passed else 2


def run_benchmark(args: argparse.Namespace, bundle: Path) -> dict[str, Any]:
    from .inference import FrozenEngine

    if not 1 <= args.requests <= 100 or not 1 <= args.repetitions <= 10:
        raise ValueError("Benchmark bounds: 1–100 requests and 1–10 repetitions")
    rows = read_jsonl(bundle / "examples" / "validation.jsonl")[: args.requests]
    output = output_directory(bundle.parents[1], "benchmark", args.output)
    with FrozenEngine(bundle, args.system, args.device) as engine:
        identity = dict(engine.identity)
        first = engine.predict(rows[0]["utterance"], timeout_seconds=args.timeout)
        for row in rows[: min(2, len(rows))]:
            engine.predict(row["utterance"], timeout_seconds=args.timeout)
        measurements = []
        for repetition in range(args.repetitions):
            for row in rows:
                prediction = engine.predict(row["utterance"], timeout_seconds=args.timeout)
                measurements.append(
                    {
                        "id": row["id"],
                        "repetition": repetition,
                        "seconds": prediction["elapsed_seconds"],
                        "generated_tokens": prediction["generated_tokens"],
                        "schema_valid": prediction["schema_valid"],
                        "memory": engine.memory_snapshot(),
                    }
                )
    times = sorted(row["seconds"] for row in measurements)
    result = {
        "identity": identity,
        "model_load_seconds": identity["load_seconds"],
        "first_request_seconds": first["elapsed_seconds"],
        "warmup_requests": min(2, len(rows)),
        "measured_requests": len(measurements),
        "p50_seconds": statistics.median(times),
        "p95_seconds": times[max(0, math.ceil(len(times) * 0.95) - 1)],
        "requests_per_second": len(times) / sum(times),
        "generated_tokens_per_second": sum(x["generated_tokens"] for x in measurements)
        / sum(times),
        "measurements": measurements,
        "timing_boundary": (
            "Tokenization, input transfer, generation, synchronization, decoding and validation; "
            "no model loading."
        ),
        "memory_note": "Per-request point samples only; no peak-memory claim.",
        "cold_note": "Load and first request are separate; OS disk caches were not flushed.",
        "output": str(output),
    }
    save_json(output / "summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle", type=Path, default=Path("data/frozen"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Hash-check independently materialized frozen inputs")
    audit = commands.add_parser("audit", help="Re-score all saved reference predictions offline")
    audit.add_argument("--output", type=Path, default=Path("reports/reference-audit.json"))
    compare = commands.add_parser(
        "compare", help="Run bounded live regression against frozen outputs"
    )
    compare.add_argument("--systems", nargs="+", choices=SYSTEM_NAMES, default=["lora", "qlora"])
    compare.add_argument("--count", type=int, default=8)
    compare.add_argument("--output")
    predict = commands.add_parser("predict", help="Make one local request")
    predict.add_argument("--system", choices=SYSTEM_NAMES, default="qlora")
    predict.add_argument("utterance")
    bench = commands.add_parser(
        "benchmark", help="Separate loading/first/warm request measurements"
    )
    bench.add_argument("--system", choices=SYSTEM_NAMES, default="qlora")
    bench.add_argument("--requests", type=int, default=8)
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--output")
    serve = commands.add_parser("serve", help="Run a single-worker local HTTP service")
    serve.add_argument("--system", choices=SYSTEM_NAMES, default="qlora")
    serve.add_argument("--port", type=int, default=8000)
    for child in (compare, predict, bench, serve):
        child.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
        child.add_argument("--timeout", type=float, default=60.0)
    return result


SYSTEM_NAMES = ["base", "quantized_base", "lora", "qlora"]


def main() -> int:
    args = parser().parse_args()
    bundle = args.bundle.resolve()
    try:
        from .artifacts import verify_bundle

        code = 0
        if args.command == "verify":
            result = verify_bundle(bundle, full=True)
        elif args.command == "audit":
            from .regression import audit_reference

            verify_bundle(bundle, full=True)
            result = audit_reference(bundle)
            save_json(args.output, result)
        elif args.command == "compare":
            if not 1 <= args.count <= 2974:
                raise ValueError("Count must be 1–2974; 2974 explicitly runs the full reused test")
            result, code = run_compare(args, bundle)
        elif args.command == "benchmark":
            result = run_benchmark(args, bundle)
        else:
            from .inference import FrozenEngine

            if args.command == "predict":
                with FrozenEngine(bundle, args.system, args.device) as engine:
                    result = engine.predict(args.utterance, timeout_seconds=args.timeout)
                code = 0 if result["schema_valid"] else 2
            else:
                import uvicorn

                from .service import create_app

                engine = FrozenEngine(bundle, args.system, args.device).load()
                uvicorn.run(
                    create_app(engine, request_timeout_seconds=args.timeout),
                    host="127.0.0.1",
                    port=args.port,
                    workers=1,
                    access_log=False,
                )
                return 0
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return code
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
