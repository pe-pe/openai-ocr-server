# Changelog

All notable changes to `openai-ocr-server` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
