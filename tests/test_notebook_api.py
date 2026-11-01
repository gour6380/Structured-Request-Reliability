"""Notebook API lifecycle checks with fake engines and tiny temporary examples."""

import asyncio
import json
import threading

import pytest

from request_reliability import notebook_api


class Engine:
    def __init__(self):
        self.identity = {"system": "qlora", "loaded": False, "manifest_sha256": "a" * 64}
        self.calls = 0
        self.closed = 0
        self.invalid_output = False
        self.change_repeat = False
        self.started = threading.Event()
        self.release = None
        self.active = False

    def load(self):
        self.identity["loaded"] = True
        return self

    def predict(self, utterance, *, timeout_seconds):
        self.active = True
        self.started.set()
        try:
            if self.release is not None:
                assert self.release.wait(2)
            self.calls += 1
            return {
                "system": "qlora",
                "identity": dict(self.identity),
                "schema_valid": not self.invalid_output,
                "parsed": {
                    "intent": "alarm_cancel"
                    if self.change_repeat and self.calls > 1
                    else "alarm_set",
                    "slots": [],
                },
                "prompt_tokens": 3,
                "generated_tokens": 2,
                "elapsed_seconds": 0.01,
            }
        finally:
            self.active = False

    def close(self):
        assert not self.active, "closing model while inference is active"
        self.closed += 1
        self.identity["loaded"] = False


@pytest.fixture
def setup_api(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    row = {
        "id": "first",
        "utterance": "set an alarm",
        "novel_input": False,
        "target": {"intent": "alarm_set", "slots": []},
    }
    (examples / "test.jsonl").write_text(json.dumps(row) + "\n")
    (examples / "tokenized_test.jsonl").write_text(
        json.dumps({**row, "prompt_ids": [1, 2, 3]}) + "\n"
    )
    engine = Engine()
    monkeypatch.setattr(notebook_api, "FrozenEngine", lambda *args: engine)
    return tmp_path, engine


def test_scoped_check_passes_and_closes_once(setup_api):
    bundle, engine = setup_api
    report = asyncio.run(notebook_api.check_api(bundle))
    assert report["status"] == "passed"
    assert report["model_released"] is True
    assert report["identity"]["loaded"] is True
    assert report["transport"] == "in_process_asgi"
    assert report["persistent_tcp_server"] is False
    assert report["repeat_parsed_agreement"] is True
    assert len(set(report["request_ids"])) == 3
    assert report["case_id"] == "first"
    assert engine.calls == 2
    assert engine.closed == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["invalid_output", "change_repeat"])
def test_nonpassing_behavior_raises_and_releases_model(setup_api, failure):
    bundle, engine = setup_api
    setattr(engine, failure, True)
    with pytest.raises(RuntimeError, match="Notebook API check failed"):
        asyncio.run(notebook_api.check_api(bundle))
    assert engine.closed == 1
    assert engine.identity["loaded"] is False


def test_cancellation_waits_for_worker_before_releasing_model(setup_api):
    bundle, engine = setup_api
    engine.release = threading.Event()

    async def scenario():
        task = asyncio.create_task(notebook_api.check_api(bundle))
        try:
            assert await asyncio.to_thread(engine.started.wait, 1)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert engine.closed == 0
        finally:
            engine.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert engine.closed == 1
    assert not engine.active


def test_invalid_timeout_rejected_before_loading(setup_api):
    bundle, engine = setup_api
    with pytest.raises(ValueError):
        asyncio.run(notebook_api.check_api(bundle, timeout_seconds=0))
    assert engine.identity["loaded"] is False
    assert engine.calls == 0
    assert engine.closed == 0
