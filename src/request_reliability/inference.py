"""Inference-only model loading. No downloads, training, or source-project imports.

Precision and prompt semantics follow the copied frozen source snapshot.
Cross-device outputs are measured, never assumed bitwise equivalent.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

SYSTEMS = ("base", "quantized_base", "lora", "qlora")


def require_available_memory(system: str, available_bytes: int) -> None:
    """Conservative MPS admission check; observed swap stops remain a second guard."""
    minimum_gib = 3 if system in ("quantized_base", "qlora") else 6
    if available_bytes < minimum_gib * 1024**3:
        raise RuntimeError(
            f"{system} requires at least {minimum_gib} GiB available unified memory before "
            "loading; close other applications. This preflight is not a guarantee of fit."
        )


def configure_offline(project: Path) -> None:
    """Call before importing Hugging Face; isolate all writable runtime caches."""
    cache = project.resolve() / ".cache"
    for key, value in {
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_XET_CACHE": str(cache / "huggingface" / "xet"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }.items():
        os.environ[key] = value


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested not in ("auto", "mps", "cpu", "cuda"):
        raise ValueError("Device must be auto, mps, cpu or cuda")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; select CPU explicitly for a separate run")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return requested


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _rotary_forward(module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    if x.device.type == "mps":
        with torch.autocast(device_type="mps", enabled=False):
            return module._original_frozen_rope(x, *args, **kwargs)
    return module._original_frozen_rope(x, *args, **kwargs)


def install_rotary_guard(model: Any) -> list[str]:
    """Retain FP32 Qwen2 rotary math inside MPS BF16 autocast."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    names = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen2RotaryEmbedding):
            if not hasattr(module, "_original_frozen_rope"):
                module._original_frozen_rope = module.forward
                module.forward = MethodType(_rotary_forward, module)
            names.append(name)
    return names


def decode_continuation(tokenizer: Any, tokens: list[int]) -> str:
    stop = len(tokens)
    terminals = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
    while stop and tokens[stop - 1] in terminals:
        stop -= 1
    return tokenizer.decode(
        tokens[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


class FrozenEngine:
    """One frozen model per instance; callers must close it before loading another."""

    def __init__(self, bundle: Path, system: str, device: str = "auto") -> None:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system {system}")
        self.bundle = bundle.resolve()
        self.system = system
        self.requested_device = device
        self.model: Any = None
        self.tokenizer: Any = None
        self._lock = threading.Lock()
        self.identity: dict[str, Any] = {"system": system, "loaded": False}
        self.load_seconds: float | None = None

    def load(self) -> FrozenEngine:
        from .artifacts import load_manifest, verify_bundle

        if self.model is not None:
            return self
        configure_offline(self.bundle.parents[1])
        verify_bundle(self.bundle, full=True)
        self.manifest = load_manifest(self.bundle)
        self.config = self.manifest["config"]
        self.ontology = json.loads((self.bundle / "ontology.json").read_text())
        self.system_prompt = (self.bundle / "system_prompt.txt").read_text()
        import psutil
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = choose_device(self.requested_device)
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        if self.device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This frozen BF16 protocol requires a BF16-capable CUDA device")
        if self.device == "mps":
            if int(platform.mac_ver()[0].split(".")[0]) < 14:
                raise RuntimeError("MPS BF16 requires macOS 14+")
            require_available_memory(self.system, psutil.virtual_memory().available)
            torch.mps.set_per_process_memory_fraction(0.65)
        self.swap_start = psutil.swap_memory().used
        started = time.perf_counter()
        backbone = "quantized_base" if self.system in ("quantized_base", "qlora") else "base_model"
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.bundle / backbone,
                local_files_only=True,
                trust_remote_code=False,
                dtype=self.dtype,
                device_map={"": self.device},
                attn_implementation="sdpa",
            )
            # Match the recorded comparison's shared non-projection precision.
            for name, parameter in model.named_parameters():
                projection = (
                    name.endswith(".weight")
                    and name.split(".")[-2] in self.config["target_modules"]
                )
                if not projection and parameter.is_floating_point():
                    parameter.data = parameter.data.to(torch.float32)
            self.guarded_modules = install_rotary_guard(model)
            if self.system in ("lora", "qlora"):
                from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
                from safetensors.torch import load_file

                adapter = self.bundle / "adapters" / self.system
                adapter_config = LoraConfig.from_pretrained(adapter, local_files_only=True)
                model = get_peft_model(model, adapter_config, autocast_adapter_dtype=True)
                result = set_peft_model_state_dict(
                    model, load_file(adapter / "adapter_model.safetensors", device="cpu")
                )
                missing = [key for key in result.missing_keys if "lora_" in key]
                if missing or result.unexpected_keys:
                    raise ValueError("Frozen adapter tensor keys do not match the base")
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.bundle / "tokenizer", local_files_only=True, trust_remote_code=False
            )
            self.model = model
            synchronize(self.device)
            self.load_seconds = time.perf_counter() - started
            self._memory_guard()
            self.identity = {
                "system": self.system,
                "loaded": True,
                "model_id": self.config["model_id"],
                "model_revision": self.config["model_revision"],
                "manifest_sha256": hashlib.sha256(
                    (self.bundle / "manifest.json").read_bytes()
                ).hexdigest(),
                "device": self.device,
                "compute_dtype": str(self.dtype),
                "source_device": "cuda",
                "cross_device_equivalence": "measured separately, not assumed",
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "peft", "bitsandbytes", "safetensors")
                },
                "rotary_guarded_modules": self.guarded_modules,
                "load_seconds": self.load_seconds,
            }
            return self
        except BaseException:
            self.model = None
            del model
            self.close()
            raise

    def _memory_guard(self) -> None:
        import psutil

        if self.device == "mps":
            if psutil.swap_memory().used - self.swap_start > 128 * 1024**2:
                raise RuntimeError("System swap grew by more than 128 MiB; stop this run")
            if psutil.virtual_memory().available < 512 * 1024**2:
                raise RuntimeError("Available unified memory below 512 MiB; stop this run")

    def prompt_ids(self, utterance: str) -> list[int]:
        if not isinstance(utterance, str) or not utterance.strip() or len(utterance) > 4096:
            raise ValueError("Request must contain 1–4096 characters and non-whitespace text")
        if self.tokenizer is None:
            raise RuntimeError("Model has not been loaded")
        ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": utterance},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) + self.config["max_new_tokens"] > self.config["max_sequence_length"]:
            raise ValueError("Request exceeds the frozen token context; truncation is disabled")
        return ids

    def predict(self, utterance: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        from .metrics import validate_prediction

        if self.model is None or not self.identity["loaded"]:
            raise RuntimeError("Model unavailable")
        if not 0 < timeout_seconds <= 600:
            raise ValueError("Timeout must be in (0, 600] seconds")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Model is busy")
        start = time.perf_counter()
        deadline = start + timeout_seconds
        engine = self

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                engine._memory_guard()
                return time.perf_counter() >= deadline

        try:
            self._memory_guard()
            prompt = self.prompt_ids(utterance)
            ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            synchronize(self.device)
            context = (
                nullcontext()
                if self.device == "cpu"
                else torch.autocast(device_type=self.device, dtype=self.dtype)
            )
            with torch.inference_mode(), context:
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([Deadline()]),
                )
            synchronize(self.device)
            tokens = output[0, len(prompt) :].detach().cpu().tolist()
            text = decode_continuation(self.tokenizer, tokens)
            validated = validate_prediction(text, self.ontology["intents"], self.ontology["slots"])
            elapsed = time.perf_counter() - start
            if time.perf_counter() >= deadline:
                raise TimeoutError("Inference exceeded its cooperative deadline")
            return {
                "system": self.system,
                "text": text,
                "prediction_text": text,
                "parsed": validated["parsed_prediction"],
                "schema_valid": validated["schema_valid"],
                "failure_reasons": validated["failure_reasons"],
                "generated_token_ids": tokens,
                "prompt_tokens": len(prompt),
                "generated_tokens": len(tokens),
                "elapsed_seconds": elapsed,
                "identity": dict(self.identity),
            }
        finally:
            self._lock.release()

    def memory_snapshot(self) -> dict[str, Any]:
        import psutil
        import torch

        result = {
            "rss_bytes": psutil.Process().memory_info().rss,
            "system_swap_used_bytes": psutil.swap_memory().used,
            "measurement": "point sample; not a peak; overlapping counters must not be summed",
        }
        if self.device == "mps":
            result.update(
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
        elif self.device == "cuda":
            result.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
            )
        return result

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.identity["loaded"] = False
        gc.collect()
        if hasattr(self, "device"):
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    def __enter__(self) -> FrozenEngine:
        return self.load()

    def __exit__(self, *args: Any) -> None:
        self.close()
