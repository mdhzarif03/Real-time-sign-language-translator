from __future__ import annotations

import ipaddress
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .features import observation_to_features
from .schemas import Hypothesis, Observation, Status

APP_VERSION = "0.1.0"
MAX_MESSAGE_BYTES = 262_144
MAX_STREAM_HZ = 20

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.runtime = None
    application.state.model_status = Status(state="model_unavailable", message="No compatible sign-language model is configured.")
    manifest_path = os.getenv("SIGNFLOW_MODEL_MANIFEST")
    if manifest_path:
        try:
            from backend.recognition.runtime import TemporalRuntime

            runtime, status = await run_in_threadpool(TemporalRuntime.load, manifest_path)
            application.state.runtime = runtime
            application.state.model_status = status
        except Exception as error:
            application.state.model_status = Status(state="model_unavailable", message=f"Local recognition runtime could not start: {error}")
    yield


app = FastAPI(title="SignFlow local inference", version=APP_VERSION, docs_url=None, redoc_url=None, lifespan=lifespan)


def loopback_peer(websocket: WebSocket) -> bool:
    peer = websocket.client.host if websocket.client else ""
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def local_origin(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    return origin.lower() in {
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    }


def model_status(application: FastAPI, language: str | None = None) -> Status:
    runtime: Any = application.state.runtime
    if runtime is not None:
        return runtime.status_for(language)
    return application.state.model_status


@app.get("/health")
async def health() -> JSONResponse:
    state = model_status(app)
    return JSONResponse({"status": "ok", "recognition": state.model_dump(), "version": APP_VERSION})


@app.websocket("/api/v1/stream/{stream_id}")
async def stream(websocket: WebSocket, stream_id: str) -> None:
    if not loopback_peer(websocket) or not local_origin(websocket):
        await websocket.close(code=1008, reason="Local clients only")
        return
    await websocket.accept()
    await websocket.send_json(model_status(app).model_dump())
    history: deque[tuple[Any, int]] = deque(maxlen=512)
    last_sequence = -1
    last_timestamp = -1
    revision = 0
    language: str | None = None
    receive_times: deque[float] = deque()
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
            await websocket.close(code=1009, reason="Message too large")
            return
        now = time.monotonic()
        receive_times.append(now)
        while receive_times and now - receive_times[0] > 1.0:
            receive_times.popleft()
        if len(receive_times) > MAX_STREAM_HZ:
            await websocket.send_json(Status(state="degraded", message="Input rate exceeded the local stream limit.").model_dump())
            await websocket.close(code=1008, reason="Stream rate limit exceeded")
            return
        try:
            observation = Observation.model_validate_json(raw)
        except ValidationError:
            await websocket.close(code=1003, reason="Invalid observation message")
            return
        if observation.stream_id != stream_id or observation.sequence <= last_sequence or observation.timestamp_ms <= last_timestamp:
            await websocket.close(code=1008, reason="Invalid stream identity or sequence")
            return
        if language is None:
            language = observation.sign_language
            await websocket.send_json(model_status(app, language).model_dump())
        elif observation.sign_language != language:
            await websocket.close(code=1008, reason="Restart the stream after changing sign language")
            return
        last_sequence = observation.sequence
        last_timestamp = observation.timestamp_ms
        runtime = app.state.runtime
        if runtime is None or model_status(app, language).state != "ready":
            continue
        history.append((observation_to_features(observation), observation.timestamp_ms))
        if len(history) >= 8 and observation.sequence % 4 == 3:
            hypothesis: Hypothesis | None = await run_in_threadpool(runtime.predict, tuple(history), stream_id, revision)
            if hypothesis is not None:
                revision += 1
                await websocket.send_json(hypothesis.model_dump())
