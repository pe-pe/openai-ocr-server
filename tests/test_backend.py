import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ocr_server.backend import IndraBackend, RapidDocBackend, create_backend


class FakeRapidDoc:
    instance = None
    markdown = "# RapidDoc OCR"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.received = None
        FakeRapidDoc.instance = self

    def __call__(self, inputs):
        data = inputs.read()
        self.received = (inputs.name, data)
        return SimpleNamespace(markdown=FakeRapidDoc.markdown)


class FakeIndraLoader:
    def __init__(self):
        self.path = "unset"

    def __call__(self, **kwargs):
        self.path = kwargs["path"]
        converter = SimpleNamespace(
            ocr=lambda image, turn: SimpleNamespace(text="indra-ocr", markdown=None)
        )
        return converter


def _install_fake_rapid_doc(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapid_doc", SimpleNamespace(RapidDoc=FakeRapidDoc))


def _install_fake_indra(monkeypatch):
    loader = FakeIndraLoader()
    models = SimpleNamespace(
        HF_REPO="test/indra-ocr",
        HF_REVISION="v1",
        FILES=("det.onnx", "rec.onnx", "order.onnx", "meta.json"),
        MATH_FILES=("mathcls.onnx",),
    )
    monkeypatch.setitem(sys.modules, "indra_ocr", SimpleNamespace(load=loader, models=models))
    return loader


def _install_fake_snapshot_download(monkeypatch, download):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))


def _write_indra_files(directory: Path) -> None:
    for name in ("det.onnx", "rec.onnx", "order.onnx", "meta.json"):
        (directory / name).write_bytes(b"model")


def test_create_backend_rapid_doc():
    assert isinstance(create_backend("rapid-doc"), RapidDocBackend)


def test_create_backend_indra():
    assert isinstance(create_backend("indra-ocr"), IndraBackend)


def test_create_backend_unknown():
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("no-such-backend")


def test_rapid_doc_load_configures_engine(monkeypatch, tmp_path):
    _install_fake_rapid_doc(monkeypatch)
    monkeypatch.delenv("RAPID_MODELS_DIR", raising=False)
    monkeypatch.delenv("MINERU_DEVICE_MODE", raising=False)
    monkeypatch.delenv("USE_DOC_ORIENTATION_CLASSIFY", raising=False)
    model_dir = tmp_path / "rapid-doc"

    backend = RapidDocBackend()
    backend.load(
        str(model_dir),
        {
            "lang": "en",
            "device": "cuda:0",
            "orientation": True,
            "formula_enable": False,
            "table_enable": False,
            "parse_method": "ocr",
            "preload": False,
        },
    )

    engine = FakeRapidDoc.instance
    assert engine.kwargs["lang"] == "en"
    assert engine.kwargs["parse_method"] == "ocr"
    assert engine.kwargs["formula_enable"] is False
    assert engine.kwargs["table_enable"] is False
    assert engine.kwargs["preload_model"] is False
    assert engine.kwargs["image_output_mode"] == "url"
    assert os.environ["RAPID_MODELS_DIR"] == str(model_dir)
    assert os.environ["MINERU_DEVICE_MODE"] == "cuda:0"
    assert os.environ["USE_DOC_ORIENTATION_CLASSIFY"] == "true"
    assert backend.ready() is True


def test_rapid_doc_default_lang_is_en(monkeypatch):
    _install_fake_rapid_doc(monkeypatch)

    backend = RapidDocBackend()
    backend.load(None, {"preload": False})

    assert FakeRapidDoc.instance.kwargs["lang"] == "en"


def test_rapid_doc_unusable_model_dir_falls_back(monkeypatch, tmp_path):
    _install_fake_rapid_doc(monkeypatch)
    monkeypatch.delenv("RAPID_MODELS_DIR", raising=False)
    not_a_dir = tmp_path / "models"
    not_a_dir.write_text("not a directory", encoding="utf-8")

    backend = RapidDocBackend()
    backend.load(str(not_a_dir), {"preload": False})

    assert "RAPID_MODELS_DIR" not in os.environ
    assert backend.ready() is True


def test_rapid_doc_infer_returns_markdown(monkeypatch):
    _install_fake_rapid_doc(monkeypatch)

    backend = RapidDocBackend()
    backend.load(None, {"preload": False})

    result = backend.infer(Image.new("L", (8, 8), 255))

    name, data = FakeRapidDoc.instance.received
    assert name == "page.png"
    assert data.startswith(b"\x89PNG")
    assert result.text == "# RapidDoc OCR"
    assert result.markdown == "# RapidDoc OCR"


def test_rapid_doc_image_output_mode_passes_through(monkeypatch):
    _install_fake_rapid_doc(monkeypatch)

    backend = RapidDocBackend()
    backend.load(None, {"image_output_mode": "data_uri", "preload": False})

    assert FakeRapidDoc.instance.kwargs["image_output_mode"] == "data_uri"


def test_rapid_doc_infer_before_load_raises():
    backend = RapidDocBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.infer(Image.new("L", (8, 8), 255))


def test_indra_downloads_models_into_empty_dir(monkeypatch, tmp_path):
    loader = _install_fake_indra(monkeypatch)
    model_dir = tmp_path / "indra-ocr"
    calls = []

    def snapshot_download(repo_id, revision, allow_patterns, ignore_patterns, local_dir):
        calls.append((repo_id, revision, Path(local_dir)))
        _write_indra_files(Path(local_dir))
        return Path(local_dir)

    _install_fake_snapshot_download(monkeypatch, snapshot_download)

    backend = IndraBackend()
    backend.load(str(model_dir), {"threads": 2})

    assert calls == [("test/indra-ocr", "v1", model_dir)]
    assert loader.path == str(model_dir)
    assert backend.ready() is True


def test_indra_uses_existing_models_without_download(monkeypatch, tmp_path):
    loader = _install_fake_indra(monkeypatch)
    model_dir = tmp_path / "indra-ocr"
    model_dir.mkdir()
    _write_indra_files(model_dir)

    def snapshot_download(**_kwargs):
        raise AssertionError("download should not be called")

    _install_fake_snapshot_download(monkeypatch, snapshot_download)

    backend = IndraBackend()
    backend.load(str(model_dir), {})

    assert loader.path == str(model_dir)
    assert backend.ready() is True


def test_indra_unusable_model_dir_falls_back(monkeypatch, tmp_path):
    loader = _install_fake_indra(monkeypatch)
    not_a_dir = tmp_path / "models"
    not_a_dir.write_text("not a directory", encoding="utf-8")

    backend = IndraBackend()
    backend.load(str(not_a_dir), {})

    assert loader.path is None
    assert backend.ready() is True


def test_indra_read_only_model_dir_falls_back(monkeypatch, tmp_path):
    loader = _install_fake_indra(monkeypatch)
    model_dir = tmp_path / "indra-ocr"

    def snapshot_download(*_args, **_kwargs):
        raise PermissionError("read-only")

    _install_fake_snapshot_download(monkeypatch, snapshot_download)

    backend = IndraBackend()
    backend.load(str(model_dir), {})

    assert loader.path is None
    assert backend.ready() is True
