"""Server configuration parsed from a single YAML blob in OPENAI_OCR_SERVER_CONFIG."""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    id: str
    backend: str = "indra-ocr"
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
    """Load the embedded default-config.yaml.

    The default config registers an Indra OCR model and a RapidDoc model and
    stores their weights under the conventional /models volume.
    """
    raw = resources.files("ocr_server").joinpath("default-config.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise TypeError("default-config.yaml must contain a YAML mapping")
    return ServerConfig.model_validate(data)


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
