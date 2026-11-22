import base64
import io
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ocr_server.app import create_app
from ocr_server.config import ModelConfig, ServerConfig


def _make_app():
    cfg = ServerConfig(
        models=[
            ModelConfig(id="indra-ocr", backend="indra-ocr", model_path=None, config={"threads": 1})
        ],
        concurrency=1,
        engine_threads=1,
        page_timeout_s=180,
        max_body_mb=100,
        max_image_pixels=50_000_000,
    )
    return create_app(config=cfg)


def test_e2e_indra_ocr():
    img = Image.new("L", (400, 100), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Hello World", fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    app = _make_app()
    with TestClient(app) as client:
        for _ in range(120):
            if client.get("/health").status_code == 200:
                break
            time.sleep(1)
        else:
            pytest.fail("service did not become ready")

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "indra-ocr",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": url}}],
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "indra-ocr"
        assert body["choices"][0]["finish_reason"] == "stop"
        text = body["choices"][0]["message"]["content"]
        assert text.strip(), "OCR returned empty content"


@pytest.mark.skipif(
    not os.environ.get("OCR_E2E_RAPID_DOC"),
    reason="set OCR_E2E_RAPID_DOC=1 to run the RapidDoc end-to-end test",
)
def test_e2e_rapid_doc():
    img = Image.new("L", (400, 100), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Hello World", fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    cfg = ServerConfig(
        models=[
            ModelConfig(
                id="rapid-doc",
                backend="rapid-doc",
                model_path=os.environ.get("OCR_E2E_RAPID_DOC_MODELS"),
                config={
                    "lang": "en",
                    "parse_method": "ocr",
                    "formula_enable": False,
                    "table_enable": False,
                },
            )
        ],
        concurrency=1,
        engine_threads=1,
        page_timeout_s=600,
        max_body_mb=100,
        max_image_pixels=50_000_000,
    )
    app = create_app(config=cfg)
    with TestClient(app) as client:
        for _ in range(900):
            if client.get("/health").status_code == 200:
                break
            time.sleep(1)
        else:
            pytest.fail("service did not become ready")

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "rapid-doc",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": url}}],
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "rapid-doc"
        assert body["choices"][0]["finish_reason"] == "stop"
        text = body["choices"][0]["message"]["content"]
        assert text.strip(), "RapidDoc returned empty content"
