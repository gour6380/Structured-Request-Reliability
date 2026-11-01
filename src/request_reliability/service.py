"""A local, single-model HTTP boundary that never queues inference requests.

HTTP timeouts and cancellation do not terminate Python/native inference threads.
Worker ownership therefore remains held until the actual call exits. The engine
must implement its own cooperative generation deadline for prompt termination.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)
MAX_INPUT_CHARACTERS = 4096
API_VERSION = "1.0.0"


class InferenceEngine(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def predict(self, utterance: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]: ...

    def close(self) -> None: ...


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    utterance: StrictStr = Field(min_length=1, max_length=MAX_INPUT_CHARACTERS)

    @field_validator("utterance")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An utterance must contain non-whitespace characters")
        # Preserve the exact input for the frozen prompt contract.
        return value


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: StrictStr
    value: StrictStr = Field(min_length=1)


class ParsedPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: StrictStr
    slots: list[Slot]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str
    result: ParsedPrediction
    system: str
    identity: dict[str, Any]
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    api_version: str
    status: Literal["ready", "busy", "unavailable", "stopping"]
    busy: bool
    identity: dict[str, Any]
    counters: dict[str, int]


def create_app(engine: InferenceEngine, *, request_timeout_seconds: float = 30.0) -> FastAPI:
    """Create an app without loading a model; the caller prepares the engine.

    Run one Uvicorn worker. Multiple processes each create an independent model
    and are outside this in-process concurrency and memory bound.
    """
    if not math.isfinite(request_timeout_seconds) or not 0 < request_timeout_seconds <= 600:
        raise ValueError("request_timeout_seconds must be finite and in (0, 600]")

    admission = asyncio.Lock()
    state: dict[str, Any] = {"stopping": False, "unavailable": False, "executor": None}
    counters = {
        name: 0
        for name in ("accepted", "succeeded", "failed", "timed_out", "cancelled", "busy_rejected")
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="request-inference")
        state.update(executor=executor, stopping=False)
        try:
            yield
        finally:
            state["stopping"] = True
            # close is queued behind the existing inference worker. Never free
            # model storage while native inference might still be using it.
            try:
                close_future = asyncio.wrap_future(executor.submit(engine.close))
                # Even cancellation of shutdown must not cancel queued cleanup.
                await asyncio.shield(close_future)
            finally:
                executor.shutdown(wait=False, cancel_futures=False)
                state["executor"] = None

    app = FastAPI(title="Structured Request Reliability", version=API_VERSION, lifespan=lifespan)

    def error_response(status: int, code: str, message: str, request_id: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error=ErrorDetail(code=code, message=message, request_id=request_id)
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default validation detail can echo the original request.
        # Omit both that detail and exception strings from responses and logs.
        return error_response(
            422,
            "invalid_input",
            "Provide one nonempty utterance of at most 4096 characters.",
            uuid4().hex,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        identity = dict(engine.identity)
        status = (
            "stopping"
            if state["stopping"]
            else "busy"
            if admission.locked()
            else "unavailable"
            if state["unavailable"] or identity.get("loaded") is False
            else "ready"
        )
        return HealthResponse(
            api_version=API_VERSION,
            status=status,
            busy=admission.locked(),
            identity=identity,
            counters=dict(counters),
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={code: {"model": ErrorResponse} for code in (422, 502, 503, 504)},
    )
    async def predict(payload: PredictionRequest) -> PredictionResponse | JSONResponse:
        request_id = uuid4().hex
        if state["stopping"] or state["executor"] is None:
            return error_response(
                503, "model_unavailable", "The inference service is unavailable.", request_id
            )
        if admission.locked():
            counters["busy_rejected"] += 1
            return error_response(
                503, "busy", "The model is processing another request.", request_id
            )
        # Acquiring an unlocked asyncio.Lock does not yield. Combined with the
        # check above, this admits a single call without creating a wait queue.
        await admission.acquire()
        counters["accepted"] += 1
        started = time.monotonic()
        outcome = "failed"
        try:
            worker = asyncio.wrap_future(
                state["executor"].submit(
                    engine.predict,
                    payload.utterance,
                    timeout_seconds=request_timeout_seconds,
                )
            )
        except Exception:
            admission.release()
            counters["failed"] += 1
            state["unavailable"] = True
            return error_response(
                503, "model_unavailable", "The inference service is unavailable.", request_id
            )

        def worker_finished(future: asyncio.Future[Any]) -> None:
            # Read late exceptions so a timed-out/cancelled request cannot cause
            # an unhandled-future warning that leaks an exception string.
            if not future.cancelled():
                error = future.exception()
                if error is not None and not isinstance(error, (ValueError, TimeoutError)):
                    state["unavailable"] = True
            admission.release()

        worker.add_done_callback(worker_finished)
        try:
            raw = await asyncio.wait_for(asyncio.shield(worker), timeout=request_timeout_seconds)
            state["unavailable"] = False
            if not isinstance(raw, dict) or raw.get("schema_valid") is not True:
                counters["failed"] += 1
                outcome = "invalid_model_output"
                return error_response(502, outcome, "Model output failed validation.", request_id)
            try:
                response = PredictionResponse(
                    request_id=request_id,
                    result=ParsedPrediction.model_validate(raw["parsed"]),
                    system=raw["system"],
                    identity=raw["identity"],
                    prompt_tokens=raw["prompt_tokens"],
                    generated_tokens=raw["generated_tokens"],
                    elapsed_seconds=raw["elapsed_seconds"],
                )
            except (KeyError, TypeError, ValidationError):
                counters["failed"] += 1
                outcome = "invalid_model_output"
                return error_response(502, outcome, "Model output failed validation.", request_id)
            counters["succeeded"] += 1
            outcome = "succeeded"
            return response
        except TimeoutError:
            counters["timed_out"] += 1
            outcome = "inference_timeout"
            return error_response(
                504, outcome, "Inference exceeded the request deadline.", request_id
            )
        except ValueError:
            counters["failed"] += 1
            outcome = "context_limit"
            return error_response(
                422, outcome, "The request exceeds the model context budget.", request_id
            )
        except asyncio.CancelledError:
            counters["cancelled"] += 1
            outcome = "cancelled"
            raise
        except Exception:
            counters["failed"] += 1
            state["unavailable"] = True
            outcome = "model_unavailable"
            return error_response(
                503, outcome, "The model could not complete inference.", request_id
            )
        finally:
            LOGGER.info(
                "inference request_id=%s outcome=%s elapsed_seconds=%.6f",
                request_id,
                outcome,
                time.monotonic() - started,
            )

    return app
