import pytest

from ocr_server.config import ServerConfig


def test_valid_config():
    cfg = ServerConfig.model_validate(
        {
            "models": [{"id": "indra-ocr", "backend": "indra", "model_path": "/models/indra"}],
            "concurrency": 2,
            "engine_threads": 1,
            "page_timeout_s": 180,
            "max_body_mb": 100,
            "max_image_pixels": 50_000_000,
        }
    )
    assert cfg.models[0].id == "indra-ocr"
    assert cfg.models[0].backend == "indra"
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000


def test_defaults():
    cfg = ServerConfig.model_validate({"models": [{"id": "m", "model_path": "/m"}]})
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000


def test_empty_models_rejected():
    with pytest.raises(ValueError):
        ServerConfig.model_validate({"models": []})


def test_load_config_defaults_when_env_missing(monkeypatch):
    from ocr_server.config import load_config

    monkeypatch.delenv("OPENAI_OCR_SERVER_CONFIG", raising=False)
    cfg = load_config()
    assert len(cfg.models) == 1
    assert cfg.models[0].id == "indra-ocr"
    assert cfg.models[0].backend == "indra"
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180.0
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000
