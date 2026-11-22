import pytest

from ocr_server.config import ServerConfig


def test_valid_config():
    cfg = ServerConfig.model_validate(
        {
            "models": [
                {
                    "id": "indra-ocr",
                    "backend": "indra-ocr",
                    "model_path": "/models/indra-ocr",
                }
            ],
            "concurrency": 2,
            "engine_threads": 1,
            "page_timeout_s": 180,
            "max_body_mb": 100,
            "max_image_pixels": 50_000_000,
        }
    )
    assert cfg.models[0].id == "indra-ocr"
    assert cfg.models[0].backend == "indra-ocr"
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000


def test_defaults():
    cfg = ServerConfig.model_validate({"models": [{"id": "m", "model_path": "/m"}]})
    assert cfg.models[0].backend == "indra-ocr"
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000


def test_rapid_doc_model_config():
    cfg = ServerConfig.model_validate(
        {
            "models": [
                {
                    "id": "rapid-doc",
                    "backend": "rapid-doc",
                    "model_path": "/models/rapid-doc",
                    "config": {"lang": "en", "formula_enable": False, "table_enable": False},
                }
            ]
        }
    )
    model = cfg.models[0]
    assert model.backend == "rapid-doc"
    assert model.model_path == "/models/rapid-doc"
    assert model.config["lang"] == "en"
    assert model.config["formula_enable"] is False
    assert model.config["table_enable"] is False


def test_empty_models_rejected():
    with pytest.raises(ValueError):
        ServerConfig.model_validate({"models": []})


def test_load_config_defaults_when_env_missing(monkeypatch):
    from ocr_server.config import load_config

    monkeypatch.delenv("OPENAI_OCR_SERVER_CONFIG", raising=False)
    cfg = load_config()
    assert [model.id for model in cfg.models] == ["indra-ocr", "rapid-doc"]
    assert cfg.models[0].backend == "indra-ocr"
    assert cfg.models[0].model_path == "/models/indra-ocr"
    assert cfg.models[1].backend == "rapid-doc"
    assert cfg.models[1].model_path == "/models/rapid-doc"
    assert cfg.models[1].config["lang"] == "en"
    assert cfg.models[1].config["image_output_mode"] == "url"
    assert cfg.concurrency == 2
    assert cfg.engine_threads == 1
    assert cfg.page_timeout_s == 180.0
    assert cfg.max_body_mb == 100
    assert cfg.max_image_pixels == 50_000_000
