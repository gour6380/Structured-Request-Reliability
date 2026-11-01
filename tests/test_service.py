"""HTTP failure, worker ownership and a tiny CPU inference-boundary regression."""

import asyncio
import logging
import threading
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from request_reliability.service import create_app


class FakeEngine:
    identity = {"system": "lora", "adapter_sha256": "a" * 64, "base_revision": "frozen"}

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False
        self.exception: Exception | None = None
        self.result: dict[str, Any] = {
            "system": "lora",
            "text": "PRIVATE_RAW_OUTPUT",
            "parsed": {"intent": "alarm_set", "slots": []},
            "schema_valid": True,
            "failure_reasons": [],
            "generated_token_ids": [1, 2],
            "prompt_tokens": 12,
            "generated_tokens": 2,
            "elapsed_seconds": 0.1,
            "identity": self.identity,
        }

    def predict(self, utterance: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        self.calls.append(utterance)
        if self.exception:
            raise self.exception
        return self.result

    def close(self) -> None:
        self.closed = True


class BlockingEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.maximum_active = 0

    def predict(self, utterance: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.started.set()
        try:
            if not self.release.wait(timeout=3):
                raise RuntimeError("test worker was never released")
            return super().predict(utterance, timeout_seconds=timeout_seconds)
        finally:
            self.active -= 1


def test_success_is_typed_preserves_input_and_does_not_echo_raw_output(caplog):
    engine = FakeEngine()
    with caplog.at_level(logging.INFO), TestClient(create_app(engine)) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/predict", json={"utterance": "  private request  "})
        assert response.status_code == 200
        body = response.json()
        assert body["result"] == {"intent": "alarm_set", "slots": []}
        assert body["identity"] == engine.identity
        assert "text" not in body and "generated_token_ids" not in body
        assert engine.calls == ["  private request  "]
        assert client.get("/health").json()["counters"]["succeeded"] == 1
    assert engine.closed
    assert "private request" not in caplog.text
    assert "PRIVATE_RAW_OUTPUT" not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"utterance": ""},
        {"utterance": " \t\n"},
        {"utterance": "x" * 4097},
        {"utterance": 1},
        {"utterance": None},
        {"utterance": "hello", "extra": "PRIVATE_INPUT"},
    ],
)
def test_invalid_input_never_reaches_engine(payload):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"
        assert "PRIVATE_INPUT" not in response.text
        assert engine.calls == []


def test_malformed_json_does_not_echo_input():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post(
            "/predict", content='{"PRIVATE_INPUT":', headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert "PRIVATE_INPUT" not in response.text


def test_health_reports_unloaded_model_without_triggering_inference():
    engine = FakeEngine()
    engine.identity = {"loaded": False, "system": "lora"}
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").json()["status"] == "unavailable"
        assert engine.calls == []


@pytest.mark.parametrize(
    "exception,status,code",
    [
        (RuntimeError("PRIVATE_MODEL_PATH"), 503, "model_unavailable"),
        (ValueError("PRIVATE_PROMPT"), 422, "context_limit"),
        (TimeoutError("PRIVATE_PROMPT"), 504, "inference_timeout"),
    ],
)
def test_engine_failures_use_safe_structured_errors(exception, status, code):
    engine = FakeEngine()
    engine.exception = exception
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict", json={"utterance": "hello"})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert "PRIVATE" not in response.text
        if status == 503:
            assert client.get("/health").json()["status"] == "unavailable"
        engine.exception = None
        assert client.post("/predict", json={"utterance": "retry"}).status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_valid": False},
        {"parsed": {"intent": "alarm_set", "slots": [{"type": "time", "value": ""}]}},
        {"parsed": {"intent": 1, "slots": []}},
        {"parsed": {"intent": "alarm_set", "slots": [], "extra": "PRIVATE_OUTPUT"}},
        {"elapsed_seconds": float("nan")},
    ],
)
def test_invalid_model_output_never_becomes_success(changes):
    engine = FakeEngine()
    engine.result.update(changes)
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict", json={"utterance": "hello"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_model_output"
        assert "PRIVATE" not in response.text


async def wait_until_idle(client: httpx.AsyncClient) -> None:
    for _ in range(100):
        if not (await client.get("/health")).json()["busy"]:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("completed worker retained admission lock")


def test_timeout_keeps_model_owned_until_worker_exits():
    async def scenario():
        engine = BlockingEngine()
        app = create_app(engine, request_timeout_seconds=0.025)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                try:
                    timed_out = await client.post("/predict", json={"utterance": "first"})
                    assert engine.started.is_set()
                    assert timed_out.status_code == 504
                    assert (await client.get("/health")).json()["status"] == "busy"
                    rejected = await client.post("/predict", json={"utterance": "second"})
                    assert rejected.status_code == 503
                    assert rejected.json()["error"]["code"] == "busy"
                finally:
                    engine.release.set()
                await wait_until_idle(client)
                assert (
                    await client.post("/predict", json={"utterance": "third"})
                ).status_code == 200
                assert engine.calls == ["first", "third"]
                assert engine.maximum_active == 1
        assert engine.closed

    asyncio.run(scenario())


def test_request_cancellation_keeps_model_owned_until_worker_exits():
    async def scenario():
        engine = BlockingEngine()
        app = create_app(engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post("/predict", json={"utterance": "first"}))
                try:
                    assert await asyncio.to_thread(engine.started.wait, 1)
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    rejected = await client.post("/predict", json={"utterance": "second"})
                    assert rejected.status_code == 503
                    assert rejected.json()["error"]["code"] == "busy"
                    assert (await client.get("/health")).json()["counters"]["cancelled"] == 1
                finally:
                    engine.release.set()
                await wait_until_idle(client)
                assert (
                    await client.post("/predict", json={"utterance": "third"})
                ).status_code == 200
                assert engine.maximum_active == 1
        assert engine.closed

    asyncio.run(scenario())


def test_shutdown_does_not_close_model_while_native_work_is_active():
    async def scenario():
        engine = BlockingEngine()
        app = create_app(engine, request_timeout_seconds=0.025)
        lifecycle = app.router.lifespan_context(app)
        await lifecycle.__aenter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json={"utterance": "first"})
            assert response.status_code == 504
            shutdown = asyncio.create_task(lifecycle.__aexit__(None, None, None))
            try:
                await asyncio.sleep(0.01)
                assert not shutdown.done()
                assert not engine.closed
                assert engine.active == 1
            finally:
                engine.release.set()
                await shutdown
            assert engine.closed
            assert engine.active == 0

    asyncio.run(scenario())


def test_frozen_prediction_identity_survives_engine_close(tmp_path, monkeypatch):
    """Use tiny CPU tensors and a fake generator; never load model weights."""
    import torch

    from request_reliability.inference import FrozenEngine

    class TinyGenerator:
        def generate(self, *, input_ids, **kwargs):
            return torch.cat((input_ids, torch.tensor([[8, 9]])), dim=1)

    class TinyTokenizer:
        eos_token_id = 9
        pad_token_id = 0

        def decode(self, tokens, **kwargs):
            assert tokens == [8]
            return '{"intent":"alarm_set","slots":[]}'

    engine = FrozenEngine(tmp_path / "unused-bundle", "lora", "cpu")
    engine.device = "cpu"
    engine.model = TinyGenerator()
    engine.tokenizer = TinyTokenizer()
    engine.identity = {"loaded": True, "system": "lora", "manifest_sha256": "f" * 64}
    engine.config = {"max_new_tokens": 2}
    engine.ontology = {"intents": ["alarm_set"], "slots": ["time"]}
    monkeypatch.setattr(engine, "prompt_ids", lambda utterance: [1, 2, 3])
    monkeypatch.setattr(engine, "_memory_guard", lambda: None)
    result = engine.predict("set an alarm")
    engine.close()

    assert result["schema_valid"] is True
    assert result["identity"] is not engine.identity
    assert result["identity"]["loaded"] is True
    assert engine.identity["loaded"] is False
    assert result["identity"]["manifest_sha256"] == "f" * 64


@pytest.mark.parametrize("timeout", [0, -1, 601, float("inf"), float("nan")])
def test_timeout_configuration_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        create_app(FakeEngine(), request_timeout_seconds=timeout)
