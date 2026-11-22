import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_server.app import create_app
from ocr_server.backend import Backend, OCRResult
from ocr_server.config import ModelConfig, ServerConfig
from ocr_server.registry import ModelRegistry


class FakeBackend(Backend):
    def __init__(self, text="fake ocr text"):
        self.text = text
        self._ready = False

    def load(self, model_path, config):
        self._ready = True

    def infer(self, image):
        return OCRResult(text=self.text)

    def ready(self):
        return self._ready


@pytest.fixture
def config():
    return ServerConfig(
        models=[ModelConfig(id="fake-ocr", backend="indra", model_path=None, config={})],
        concurrency=2,
        engine_threads=1,
        page_timeout_s=5,
        max_body_mb=1,
        max_image_pixels=1_000_000,
    )


@pytest.fixture
def registry(config):
    reg = ModelRegistry(config)
    reg._backends["fake-ocr"] = FakeBackend()
    reg.ready = True
    return reg


@pytest.fixture
def app(config, registry):
    async def _noop():
        return None

    registry.load_all = _noop
    return create_app(config=config, registry=registry)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_data_url():
    img = Image.new("L", (64, 64), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
