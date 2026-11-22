import base64
import io

from PIL import Image

from tests.conftest import FakeBackend


def test_health_ready(client, registry):
    assert registry.ready is True
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_not_ready(client, registry):
    registry.ready = False
    registry._backends = {}
    try:
        r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "starting"
    finally:
        registry.ready = True
        registry._backends["fake-ocr"] = FakeBackend()


def test_models_list(client, registry):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "fake-ocr" in ids


def test_unknown_model_400(client, png_data_url):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "no-such-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": png_data_url}}],
                }
            ],
        },
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["type"] == "invalid_request"
    assert "unknown model" in body["message"]


def test_chat_completions_returns_ocr_text(client, png_data_url):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": png_data_url},
                        }
                    ],
                }
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-ocr"
    assert isinstance(body["created"], int) and body["created"] > 0
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "fake ocr text"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["id"].startswith("chatcmpl-")


def test_invalid_json_400(client):
    r = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400, r.text
    assert r.json()["type"] == "invalid_request"


def test_missing_image_422(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "fake-ocr", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 422, r.text
    assert r.json()["type"] == "unprocessable"


def test_invalid_base64_422(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,!!!notbase64!!!"},
                        }
                    ],
                }
            ],
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["type"] == "unprocessable"


def test_body_too_large_413(client):
    r = client.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"content-length": str(2 * 1024 * 1024)},
    )
    assert r.status_code == 413, r.text
    assert r.json()["type"] == "too_large"


def test_image_too_many_pixels_413(client):
    img = Image.new("L", (1200, 1200), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-ocr",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
            ],
        },
    )
    assert r.status_code == 413, r.text
    assert r.json()["type"] == "too_large"
