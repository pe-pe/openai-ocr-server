"""FastAPI application: OpenAI-compatible chat-completions OCR API.

Routes:
  GET    /health
  GET    /v1/models
  POST   /v1/chat/completions

The adapter (librechat-ocr-adapter) posts one base64 PNG image per request and
reads OCR text from choices[0].message.content. Generation-only parameters
(temperature, top_p, max_tokens) are accepted and ignored.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .backend import Backend, OCRResult
from .config import ServerConfig
from .registry import ModelRegistry

logger = logging.getLogger("ocr_server")


class OCRError(Exception):
    """Structured service error mapped to a JSON {message, type} response."""

    def __init__(self, message: str, status_code: int = 500, type_: str = "internal_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type_


def _error(message: str, status_code: int, type_: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"message": message, "type": type_})


async def _ocr_error_handler(request: Request, exc: OCRError) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    backend = getattr(request.state, "backend", None)
    logger.warning(
        "ocr_error request_id=%s backend=%s type=%s message=%s",
        rid,
        backend or "-",
        exc.type,
        exc.message,
    )
    return _error(exc.message, exc.status_code, exc.type)


def create_app(
    config: ServerConfig | None = None, registry: ModelRegistry | None = None
) -> FastAPI:
    if config is None:
        from .config import load_config

        config = load_config()

    registry = registry or ModelRegistry(config)

    semaphore = asyncio.Semaphore(config.concurrency)
    active_requests = 0
    active_requests_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start eager model loading in the background so the port opens
        # immediately; /health reports non-200 until loading finishes.
        load_task = asyncio.create_task(registry.load_all())
        try:
            yield
        finally:
            # Graceful shutdown: wait for in-flight requests to complete
            logger.info("shutdown_started waiting_for_inflight_requests")
            # Cancel model loading if still in progress (don't wait for HF downloads)
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                logger.info("model_loading_cancelled_during_shutdown")
            except Exception:
                logger.exception("model_loading_failed_during_shutdown")

            # Wait for active requests to drain (with timeout)
            shutdown_timeout = 3.0  # seconds
            start_time = time.monotonic()
            while True:
                async with active_requests_lock:
                    if active_requests == 0:
                        break
                    remaining = active_requests
                elapsed = time.monotonic() - start_time
                if elapsed >= shutdown_timeout:
                    logger.warning(
                        "shutdown_timeout active_requests=%d elapsed=%.1fs",
                        remaining,
                        elapsed,
                    )
                    break
                logger.info("shutdown_waiting active_requests=%d elapsed=%.1fs", remaining, elapsed)
                await asyncio.sleep(0.5)

            # Also wait for semaphore to be fully released (all permits available)
            # This ensures no inference is running in the thread pool
            while semaphore._value < config.concurrency:  # type: ignore[attr-defined]
                elapsed = time.monotonic() - start_time
                if elapsed >= shutdown_timeout:
                    logger.warning("shutdown_timeout waiting_for_semaphore")
                    break
                await asyncio.sleep(0.1)

            logger.info("shutdown_complete")

    app = FastAPI(title="openai-ocr-server", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def _request_tracking(request: Request, call_next):
        nonlocal active_requests
        request.state.request_id = str(uuid.uuid4())
        request.state.backend = None
        async with active_requests_lock:
            active_requests += 1
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            async with active_requests_lock:
                active_requests -= 1
            ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request_id=%s method=%s path=%s status=%s backend=%s ms=%.1f",
                request.state.request_id,
                request.method,
                request.url.path,
                getattr(response, "status_code", 0),
                request.state.backend or "-",
                ms,
            )
            if response is not None and hasattr(response, "headers"):
                response.headers["X-Request-ID"] = request.state.request_id

    @app.exception_handler(OCRError)
    async def _handle_ocr_error(request: Request, exc: OCRError):
        return await _ocr_error_handler(request, exc)

    @app.get("/health")
    async def health():
        if registry.ready:
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "starting"})

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model"} for model_id in registry.model_ids],
        }

    def _extract_image_url(data: dict[str, Any]) -> str:
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OCRError("messages is required", 400, "invalid_request")
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            parts = (
                content
                if isinstance(content, list)
                else ([content] if isinstance(content, str) else [])
            )
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if isinstance(url, str) and url:
                    return url
        raise OCRError("no image_url part found", 422, "unprocessable")

    def _decode_image(data_url: str) -> Image.Image:
        prefix = "data:image/png;base64,"
        if not data_url.startswith(prefix):
            raise OCRError(
                "unsupported image URL; expected data:image/png;base64",
                422,
                "unprocessable",
            )
        b64 = data_url[len(prefix) :]
        try:
            raw = base64.b64decode(b64, validate=True)
        except ValueError, binascii.Error:
            raise OCRError("invalid base64 image data", 422, "unprocessable")
        try:
            with BytesIO(raw) as buf:
                image = Image.open(buf)
                image.load()
        except UnidentifiedImageError:
            raise OCRError("invalid PNG image", 422, "unprocessable")
        width, height = image.size
        if width * height > config.max_image_pixels:
            raise OCRError(
                f"image too large: {width}x{height} pixels exceeds {config.max_image_pixels}",
                413,
                "too_large",
            )
        return image

    async def _run_inference(backend: Backend, image: Image.Image) -> OCRResult:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(backend.infer, image),
                    timeout=config.page_timeout_s,
                )
            except TimeoutError:
                raise OCRError(
                    f"inference did not finish within {config.page_timeout_s:.0f}s",
                    504,
                    "timeout",
                )
            except OCRError:
                raise
            except Exception as exc:  # backend failure -> explicit error, never empty success
                logger.exception("backend_inference_failed")
                raise OCRError(
                    f"backend inference failed: {type(exc).__name__}", 502, "upstream_error"
                )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        rid = request.state.request_id

        if not registry.ready:
            raise OCRError(
                "service starting up, models not yet loaded", 503, "upstream_unavailable"
            )

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > config.max_body_mb * 1024 * 1024:
                    raise OCRError("request body too large", 413, "too_large")
            except ValueError:
                pass

        body = await request.body()
        if len(body) > config.max_body_mb * 1024 * 1024:
            raise OCRError("request body too large", 413, "too_large")

        try:
            data = json.loads(body)
        except ValueError, TypeError:
            raise OCRError("invalid JSON body", 400, "invalid_request")
        if not isinstance(data, dict):
            raise OCRError("request body must be a JSON object", 400, "invalid_request")

        model_id = data.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise OCRError("model is required", 400, "invalid_request")
        if not registry.contains(model_id):
            raise OCRError(f"unknown model: {model_id}", 400, "invalid_request")
        backend: Backend = registry.get(model_id)
        request.state.backend = model_id

        image_url = _extract_image_url(data)
        image = _decode_image(image_url)

        result: OCRResult = await _run_inference(backend, image)

        content = result.markdown if result.markdown else result.text

        return {
            "id": f"chatcmpl-{rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app
