"""Agnostic OCR backend interface with the Indra OCR implementation.

The HTTP layer owns validation, decoding and response formatting.  The backend
layer owns model-specific preprocessing, inference and conversion of native
output into a common :class:`OCRResult`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    markdown: str | None = None


class Backend:
    """Single warm model instance reused across requests."""

    def load(self, model_path: str | None, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def infer(self, image: Image.Image) -> OCRResult:
        raise NotImplementedError

    def ready(self) -> bool:
        raise NotImplementedError


class IndraBackend(Backend):
    def __init__(self) -> None:
        self._ocr = None
        self._ready = False

    def load(self, model_path: str | None, config: dict[str, Any]) -> None:
        from indra_ocr import load

        threads = int(config.get("threads", 1))
        # Bind Indra/onnxruntime internal CPU threads so the engine does not
        # oversubscribe the host alongside the asyncio semaphore concurrency.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        converter = load(
            path=model_path,
            threads=threads,
            mode="ocr",
            image_ocr=True,
        )
        self._ocr = converter.ocr
        self._ready = True

    def infer(self, image: Image.Image) -> OCRResult:
        if not self._ready or self._ocr is None:
            raise RuntimeError("indra backend not loaded")
        gray = np.asarray(image.convert("L"))
        page = self._ocr(gray, turn=True)
        return OCRResult(text=page.text, markdown=page.markdown)

    def ready(self) -> bool:
        return self._ready


BACKENDS: dict[str, type[Backend]] = {"indra": IndraBackend}


def create_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend: {name}")
