"""Agnostic OCR backend interface with the Indra OCR and RapidDoc implementations.

The HTTP layer owns validation, decoding and response formatting.  The backend
layer owns model-specific preprocessing, inference and conversion of native
output into a common :class:`OCRResult`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("ocr_server")


@dataclass
class OCRResult:
    text: str
    markdown: str | None = None


def _prepare_model_dir(model_path: str | None, backend: str) -> Path | None:
    """Create the configured model directory, falling back to the default cache."""
    if not model_path:
        return None
    directory = Path(model_path).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "%s_model_dir_unavailable path=%s error=%s; using backend default cache",
            backend,
            directory,
            type(exc).__name__,
        )
        return None
    return directory


def _ensure_indra_models(directory: Path) -> bool:
    """Populate an Indra model directory, returning False if it is not writable."""
    from huggingface_hub import snapshot_download
    from indra_ocr import models as indra_models

    required = (*indra_models.FILES[:2], *indra_models.FILES[3:])
    if all((directory / name).exists() for name in required):
        return True
    try:
        snapshot_download(
            indra_models.HF_REPO,
            revision=indra_models.HF_REVISION,
            allow_patterns=["*.onnx", "meta.json", "lm.npz", "lm_*.npz"],
            ignore_patterns=list(indra_models.MATH_FILES),
            local_dir=directory,
        )
    except OSError as exc:
        logger.warning(
            "indra_model_dir_not_writable path=%s error=%s; using default cache",
            directory,
            type(exc).__name__,
        )
        return False
    return True


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

        path = None
        directory = _prepare_model_dir(model_path, "indra-ocr")
        if directory is not None and _ensure_indra_models(directory):
            path = str(directory)

        threads = int(config.get("threads", 1))
        converter = load(
            path=path,
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


_RAPID_DOC_LOCK = threading.Lock()


class RapidDocBackend(Backend):
    """Warm RapidDoc document-parsing pipeline reused across requests.

    RapidDoc accepts document paths or byte streams, not decoded PIL images, so
    each page is encoded as a PNG byte stream before inference. RapidDoc keeps a
    process-wide model cache and does not document its pipeline as thread-safe,
    so inference is serialized across all RapidDoc model instances.
    """

    def __init__(self) -> None:
        self._engine = None
        self._ready = False

    def load(self, model_path: str | None, config: dict[str, Any]) -> None:
        directory = _prepare_model_dir(model_path, "rapid-doc")
        if directory is not None:
            os.environ["RAPID_MODELS_DIR"] = str(directory)
        device = config.get("device")
        if device:
            os.environ["MINERU_DEVICE_MODE"] = str(device)
        if config.get("orientation"):
            os.environ["USE_DOC_ORIENTATION_CLASSIFY"] = "true"

        from rapid_doc import RapidDoc

        self._engine = RapidDoc(
            layout_config=config.get("layout"),
            ocr_config=config.get("ocr"),
            formula_config=config.get("formula"),
            table_config=config.get("table"),
            checkbox_config=config.get("checkbox"),
            image_config=config.get("image"),
            parse_method=str(config.get("parse_method", "auto")),
            formula_enable=bool(config.get("formula_enable", True)),
            table_enable=bool(config.get("table_enable", True)),
            lang=str(config.get("lang", "en")),
            image_output_mode=str(config.get("image_output_mode", "data_uri")),
            preload_model=bool(config.get("preload", True)),
        )
        self._ready = True

    def infer(self, image: Image.Image) -> OCRResult:
        if not self._ready or self._engine is None:
            raise RuntimeError("rapid-doc backend not loaded")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "page.png"
        with _RAPID_DOC_LOCK:
            output = self._engine(buffer)
        markdown = output.markdown or ""
        return OCRResult(text=markdown, markdown=markdown or None)

    def ready(self) -> bool:
        return self._ready


BACKENDS: dict[str, type[Backend]] = {
    "indra-ocr": IndraBackend,
    "rapid-doc": RapidDocBackend,
}


def create_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend: {name}")
