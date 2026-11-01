"""CLI failure semantics without loading an accelerator model."""

import json

from request_reliability import cli, inference


def test_invalid_prediction_is_nonzero_and_preserves_diagnostics(monkeypatch, capsys):
    class Engine:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def predict(self, utterance, *, timeout_seconds):
            return {
                "schema_valid": False,
                "text": "bad output",
                "failure_reasons": ["invalid_json"],
            }

    monkeypatch.setattr(inference, "FrozenEngine", Engine)
    monkeypatch.setattr("sys.argv", ["request-reliability", "predict", "hello"])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["failure_reasons"] == ["invalid_json"]


def test_verified_memory_efficient_system_is_local_default():
    assert cli.parser().parse_args(["predict", "hello"]).system == "qlora"
    assert cli.parser().parse_args(["serve"]).system == "qlora"
