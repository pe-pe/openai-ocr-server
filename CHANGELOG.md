# Changelog

All notable changes to `openai-ocr-server` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **RapidDoc backend**: second CPU OCR engine alongside Indra, configured with
  `backend: rapid-doc` and selectable through the request `model` field.
  - Layout-aware Markdown output with optional formula and table recognition.
  - Per-model configuration for language, parse method, component model paths,
    device, orientation correction, image output mode and eager preloading.
  - RapidDoc inference is serialized across RapidDoc models because its
    pipeline is not documented as thread-safe.
- **Embedded default config**: `src/ocr_server/default-config.yaml` is packaged
  in the image and used when `OPENAI_OCR_SERVER_CONFIG` is unset. It registers
  an `indra-ocr` model and a `rapid-doc` model with `lang: en`.
- **Model directory bootstrap**: configured `model_path` directories are
  created automatically. Indra and RapidDoc download missing weights into an
  empty `/models` volume; if the path is not writable, they fall back to their
  default caches with a warning.
- **Docker model volume**: the image creates and declares `/models` as a
  volume, so model downloads persist across container recreation.
- Unit tests for the RapidDoc backend, Indra model-directory bootstrapping, and
  an opt-in RapidDoc end-to-end test (`OCR_E2E_RAPID_DOC=1`, enabled in CI).

### Changed
- The Indra backend name is `indra-ocr`, matching its model ID and
  `/models/indra-ocr` path.
- `EXPOSE` in the Dockerfile now derives from the `PORT` build-time env value
  (default 8000); runtime `PORT` overrides still control uvicorn.
- Pillow requirement relaxed from `>=13.0` to `>=11.0`; `rapid-doc` currently
  constrains Pillow below version 13.
- Documentation (README, DESIGN) updated to describe RapidDoc in place of the
  previously planned PaddleOCR backend, including an Indra backend section and
  a writable `/models` Docker volume instead of a read-only mount.

## [v0.1.1] - 2026-11-22

### Fixed
- **Graceful shutdown**: Server now tracks active requests and waits for in-flight
  requests and semaphore to drain on SIGTERM (30s timeout) instead of hanging
  indefinitely requiring SIGKILL
- **Created timestamp**: Chat completion responses now use `int(time.time())`
  instead of hardcoded `0`
- **Model loading race condition**: Models load in parallel with `asyncio.gather`,
  partial failures don't block other models; added per-model status tracking
  (`ModelStatus`)
- **BytesIO leak**: Image decoding now uses context manager to ensure cleanup

### Removed
- Hardcoded thread env vars (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`) from Dockerfile and code - no longer needed

### Changed
- **Dependencies updated** to latest stable versions:
  - fastapi: 0.115 → 0.165
  - uvicorn: 0.32 → 0.54
  - pydantic: 2.9 → 2.14
  - pydantic-settings: 2.3 → 2.16
  - Pillow: 10 → 13.0
  - numpy: 1.26 → 2.5
  - pytest: 8 → 9.2
  - httpx: 0.27 → 0.28
  - ruff: 0.9 → 0.17

- **Python version**: 3.13 → 3.14 (requires-python >=3.14)
  - Dockerfile: python:3.14-slim
  - CI workflow: Python 3.14

## [v0.1.0] - 2026-11-22

### Added
- Initial release of `openai-ocr-server`
- OpenAI-compatible `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`
- Indra OCR backend behind an agnostic `Backend` interface
- Single `OPENAI_OCR_SERVER_CONFIG` YAML config (optional, defaults to one `indra-ocr` model)
- Eager model loading at startup; `/health` returns 200 only when ready
- CPU-bound inference off the event loop (`asyncio.to_thread`) gated by `asyncio.Semaphore`
- Per-page inference deadline with explicit timeout errors
- Markdown output preferred when available, plain text fallback
- Docker image for `linux/amd64`
- CI/CD pipeline: ruff lint/format, pytest, Docker build + smoke test, GHCR deploy
- Dependabot for pip, Docker, and GitHub Actions dependencies
- Release workflow triggered on `v*` tags
