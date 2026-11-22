"""Model registry: model_id -> warm backend instance.

Every configured model ID is registered at startup and its Indra engine is
loaded eagerly so inference can run immediately once the server is up.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .backend import Backend, create_backend
from .config import ModelConfig, ServerConfig

logger = logging.getLogger("ocr_server")


@dataclass
class ModelStatus:
    """Per-model loading status."""

    model_id: str
    backend: str
    loaded: bool = False
    error: str | None = None


class ModelRegistry:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._backends: dict[str, Backend] = {}
        self._model_status: dict[str, ModelStatus] = {}
        self.ready = False

    @property
    def model_ids(self) -> list[str]:
        return [model.id for model in self._config.models]

    def contains(self, model_id: str) -> bool:
        return model_id in self._backends

    def get(self, model_id: str) -> Backend:
        backend = self._backends.get(model_id)
        if backend is None:
            raise KeyError(model_id)
        return backend

    def get_status(self, model_id: str) -> ModelStatus | None:
        """Get loading status for a specific model."""
        return self._model_status.get(model_id)

    def all_statuses(self) -> list[ModelStatus]:
        """Get loading status for all models."""
        return list(self._model_status.values())

    async def load_all(self) -> None:
        """Load every configured model eagerly (off the event loop) in parallel.

        Failed models are logged but don't prevent other models from loading.
        The registry is marked ready if at least one model loads successfully.
        """
        # Map model_id to model config for quick lookup
        model_map = {model.id: model for model in self._config.models}

        async def load_one(model: ModelConfig) -> tuple[str, Backend | None, str | None]:
            try:
                backend = create_backend(model.backend)
                await asyncio.to_thread(backend.load, model.model_path, model.config)
                return model.id, backend, None
            except Exception as exc:
                logger.exception(
                    "model_load_failed model_id=%s backend=%s", model.id, model.backend
                )
                return model.id, None, f"{type(exc).__name__}: {exc}"

        # Load all models in parallel
        results = await asyncio.gather(
            *[load_one(model) for model in self._config.models],
            return_exceptions=False,
        )

        loaded_count = 0
        for model_id, backend, error in results:
            model_config = model_map[model_id]
            status = ModelStatus(
                model_id=model_id,
                backend=model_config.backend,
                loaded=backend is not None,
                error=error,
            )
            self._model_status[model_id] = status
            if backend is not None:
                self._backends[model_id] = backend
                loaded_count += 1
                logger.info("model_loaded model_id=%s backend=%s", model_id, status.backend)
            else:
                logger.error("model_load_failed model_id=%s error=%s", model_id, error)

        self.ready = loaded_count > 0
        if self.ready:
            logger.info(
                "registry_ready model_ids=%s", [m for m in self.model_ids if self.contains(m)]
            )
        else:
            logger.error("registry_ready_failed no_models_loaded")
