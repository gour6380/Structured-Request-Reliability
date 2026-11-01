"""Bounded real-model ASGI check; uses no TCP listener and never trains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from request_reliability.cli import save_json
from request_reliability.inference import FrozenEngine
from request_reliability.service import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=Path("reports/local-api-check.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing API evidence")
    bundle = Path("data/frozen").resolve()
    reference = json.loads(
        (bundle / "reference" / f"{args.system}.jsonl").read_text().splitlines()[0]
    )
    engine = FrozenEngine(bundle, args.system, args.device).load()
    identity = dict(engine.identity)
    with TestClient(create_app(engine, request_timeout_seconds=60)) as client:
        health = client.get("/health")
        invalid = client.post("/predict", json={"utterance": " "})
        first = client.post("/predict", json={"utterance": reference["utterance"]})
        second = client.post("/predict", json={"utterance": reference["utterance"]})
        if first.status_code != 200 or second.status_code != 200:
            raise RuntimeError("Real-model API prediction failed")
        a, b = first.json(), second.json()
        record = {
            "identity": identity,
            "transport": "in-process ASGI TestClient; not a live network/load test",
            "reference_id": reference["id"],
            "health_status": health.json()["status"],
            "invalid_input_status": invalid.status_code,
            "prediction_statuses": [first.status_code, second.status_code],
            "response_matches_reference": a["result"] == reference["parsed_prediction"],
            "repeat_parsed_output_equal": a["result"] == b["result"],
            "request_ids_distinct": a["request_id"] != b["request_id"],
            "durations_seconds": [a["elapsed_seconds"], b["elapsed_seconds"]],
            "counters": client.get("/health").json()["counters"],
            "memory": engine.memory_snapshot(),
        }
    record["closed_after_shutdown"] = engine.identity["loaded"] is False
    record["passed"] = all(
        [
            record["health_status"] == "ready",
            record["invalid_input_status"] == 422,
            record["response_matches_reference"],
            record["repeat_parsed_output_equal"],
            record["request_ids_distinct"],
            record["closed_after_shutdown"],
        ]
    )
    save_json(args.output, record)
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
