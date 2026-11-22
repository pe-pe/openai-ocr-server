"""Model registry: model_id -> warm backend instance.

Every configured model ID is registered at startup and its Indra engine is
loaded eagerly so inference can run immediately once the server is up.
"""

from __future__ import annotations

import asyncio
import logging

from .backend import Backend, create_backend
from .config import ServerConfig

logger = logging.getLogger("ocr_server")


class ModelRegistry:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._backends: dict[str, Backend] = {}
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

    async def load_all(self) -> None:
        """Load every configured model eagerly (off the event loop)."""
        for model in self._config.models:
            backend = create_backend(model.backend)
            await asyncio.to_thread(backend.load, model.model_path, model.config)
            self._backends[model.id] = backend
            logger.info(
                "model_loaded model_id=%s backend=%s",
                model.id,
                model.backend,
            )
        self.ready = True
        logger.info("registry_ready model_ids=%s", self.model_ids)
