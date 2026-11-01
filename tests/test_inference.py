"""Token and input contracts without loading a full model."""

from pathlib import Path

import pytest

from request_reliability.inference import (
    FrozenEngine,
    decode_continuation,
    require_available_memory,
)


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return repr(tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0] == {"role": "system", "content": "frozen\n"}
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        return {"input_ids": [1, 2, 3]}


def test_decoding_keeps_internal_special_tokens():
    assert decode_continuation(Tokenizer(), [1, 9, 2, 9, 0]) == "[1, 9, 2]"


def engine():
    value = FrozenEngine(Path("/tmp/missing-bundle"), "lora")
    value.tokenizer = Tokenizer()
    value.system_prompt = "frozen\n"
    value.config = {"max_new_tokens": 2, "max_sequence_length": 5}
    return value


def test_prompt_uses_frozen_system_text_and_accepts_boundary():
    assert engine().prompt_ids("hello") == [1, 2, 3]


def test_context_overflow_is_not_truncated():
    value = engine()
    value.config["max_sequence_length"] = 4
    with pytest.raises(ValueError, match="context"):
        value.prompt_ids("hello")


@pytest.mark.parametrize("text", ["", "  ", "x" * 4097, None, 7])
def test_bad_input_rejected(text):
    with pytest.raises(ValueError):
        engine().prompt_ids(text)


def test_missing_model_fails_without_loading():
    value = FrozenEngine(Path("/tmp/missing-bundle"), "lora")
    with pytest.raises(RuntimeError, match="not been loaded"):
        value.prompt_ids("hello")


def test_no_unknown_model_or_device_fallback():
    with pytest.raises(ValueError, match="Unknown system"):
        FrozenEngine(Path("/tmp"), "invented")


def test_dense_loader_rejects_insufficient_memory_before_allocating():
    with pytest.raises(RuntimeError, match="6 GiB"):
        require_available_memory("lora", 5 * 1024**3)
    require_available_memory("lora", 6 * 1024**3)


def test_quantized_loader_has_its_own_memory_preflight():
    with pytest.raises(RuntimeError, match="3 GiB"):
        require_available_memory("qlora", 2 * 1024**3)
    require_available_memory("qlora", 3 * 1024**3)
