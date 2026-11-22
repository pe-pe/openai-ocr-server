"""Server configuration parsed from a single YAML blob in OPENAI_OCR_SERVER_CONFIG."""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    id: str
    backend: str = "indra"
    model_path: str | None = None
    config: dict[str, Any] = {}

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model id must be non-empty")
        return v.strip()


class ServerConfig(BaseModel):
    models: list[ModelConfig]
    concurrency: int = 2
    engine_threads: int = 1
    page_timeout_s: float = 180.0
    max_body_mb: int = 100
    max_image_pixels: int = 50_000_000

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        if not v:
            raise ValueError("at least one model must be configured")
        return v

    @field_validator("concurrency", "engine_threads")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("page_timeout_s")
    @classmethod
    def _positive_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


def _default_config() -> ServerConfig:
    """Sensible defaults when OPENAI_OCR_SERVER_CONFIG is not set.

    A single indra-ocr model with no model_path lets Indra download its default
    weights on first load (cached by huggingface_hub).
    """
    return ServerConfig(
        models=[ModelConfig(id="indra-ocr", backend="indra", model_path=None, config={})],
        concurrency=2,
        engine_threads=1,
        page_timeout_s=180.0,
        max_body_mb=100,
        max_image_pixels=50_000_000,
    )


def load_config() -> ServerConfig:
    """Parse OPENAI_OCR_SERVER_CONFIG (YAML) into a validated ServerConfig.

    The variable is optional: when unset or empty, a default single-model
    config is returned so the service starts without explicit configuration.
    """
    raw = os.environ.get("OPENAI_OCR_SERVER_CONFIG")
    if not raw:
        return _default_config()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise TypeError("OPENAI_OCR_SERVER_CONFIG must contain a YAML mapping")
    return ServerConfig.model_validate(data)
