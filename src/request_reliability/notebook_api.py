"""A scoped HTTP-contract check for an existing notebook event loop.

This helper opens no TCP listener and writes no files. Model loading is explicit;
the service lifespan owns cleanup after requests, including timed-out workers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .inference import FrozenEngine
from .regression import fixed_reference_cases


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Notebook API check failed: {message}")


async def check_api(
    bundle: Path,
    system: str = "qlora",
    device: str = "auto",
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Load one model, check its in-process API and release it before returning.

    Call with ``await check_api(...)`` inside a notebook; never ``asyncio.run``.
    A failed status, inconsistent repeat or failed cleanup raises an exception.
    Success checks the HTTP contract for one reused example; it does not certify
    semantic correctness, full-dataset equivalence or persistent-server behavior.
    """
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be finite and in (0, 600]")

    import httpx

    from .service import create_app

    case = fixed_reference_cases(Path(bundle).resolve(), count=1)[0]
    engine = FrozenEngine(Path(bundle).resolve(), system, device)
    engine.load()
    lifespan_owns_engine = False
    try:
        app = create_app(engine, request_timeout_seconds=timeout_seconds)
        async with app.router.lifespan_context(app):
            lifespan_owns_engine = True
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://notebook.local"
            ) as client:
                before = await client.get("/health")
                _require(before.status_code == 200, "initial health endpoint did not return 200")
                before_body = before.json()
                _require(before_body.get("status") == "ready", "model was not ready")
                _require(before_body.get("busy") is False, "model was already busy")
                _require(
                    before_body.get("identity", {}).get("loaded") is True,
                    "health did not identify a loaded model",
                )
                payload = {"utterance": case["utterance"]}
                first = await client.post("/predict", json=payload)
                _require(first.status_code == 200, f"first prediction returned {first.status_code}")
                repeated = await client.post("/predict", json=payload)
                _require(
                    repeated.status_code == 200,
                    f"repeated prediction returned {repeated.status_code}",
                )
                invalid = await client.post("/predict", json={"utterance": " \t\n"})
                _require(invalid.status_code == 422, "whitespace input was not rejected with 422")
                first_body, repeated_body, invalid_body = (
                    first.json(),
                    repeated.json(),
                    invalid.json(),
                )
                _require(
                    invalid_body.get("error", {}).get("code") == "invalid_input",
                    "invalid input did not receive its structured error code",
                )
                _require(
                    first_body.get("result") is not None
                    and first_body["result"] == repeated_body.get("result"),
                    "repeated parsed outputs disagreed",
                )
                _require(
                    first_body.get("identity")
                    == repeated_body.get("identity")
                    == before_body["identity"],
                    "model identity changed during the check",
                )
                request_ids = [
                    first_body.get("request_id"),
                    repeated_body.get("request_id"),
                    invalid_body.get("error", {}).get("request_id"),
                ]
                _require(
                    all(isinstance(value, str) and value for value in request_ids)
                    and len(set(request_ids)) == 3,
                    "request IDs were missing or not distinct",
                )
                after = await client.get("/health")
                _require(after.status_code == 200, "final health endpoint did not return 200")
                after_body = after.json()
                _require(
                    after_body.get("status") == "ready" and after_body.get("busy") is False,
                    "model did not become idle after the requests",
                )
                _require(
                    after_body.get("counters", {}).get("accepted") == 2
                    and after_body.get("counters", {}).get("succeeded") == 2,
                    "successful inference count was not exactly two",
                )
                report = {
                    "status": "passed",
                    "inference_performed": True,
                    "transport": "in_process_asgi",
                    "persistent_tcp_server": False,
                    "case_id": str(case["id"]),
                    "system": system,
                    "identity": first_body["identity"],
                    "prediction": first_body["result"],
                    "request_ids": request_ids,
                    "http_statuses": {
                        "health_before": before.status_code,
                        "first_prediction": first.status_code,
                        "repeat_prediction": repeated.status_code,
                        "invalid_input": invalid.status_code,
                        "health_after": after.status_code,
                    },
                    "repeat_parsed_agreement": True,
                    "distinct_request_ids": True,
                    "final_counters": after_body["counters"],
                    "qualification": (
                        "HTTP contract and repeated parsed output on one reused frozen example; "
                        "not a correctness evaluation or TCP serving benchmark."
                    ),
                }
    finally:
        # Once entered, lifespan queues close behind any outstanding worker.
        # Closing directly here after a cancellation would race that worker.
        if not lifespan_owns_engine:
            engine.close()
    _require(engine.identity.get("loaded") is False, "model was not released on lifespan exit")
    report["model_released"] = True
    return report
