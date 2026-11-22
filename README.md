# openai-ocr-server

Self-hosted OCR inference service with an OpenAI-compatible chat-completions
API (`POST /v1/chat/completions`, `GET /v1/models`, `GET /health`). It is a
drop-in replacement for the vLLM upstream used by
[librechat-ocr-adapter](https://github.com/virtuos/librechat-ocr-adapter).

The initial backend is Indra OCR (CPU), served behind a single agnostic
backend interface. Only one base64 PNG image per request is handled in the
MVP; multi-image and server-side PDF are deferred.

## Configuration

All configuration is a single YAML document passed through the
`OPENAI_OCR_SERVER_CONFIG` environment variable. The variable is optional:
when unset, a default single-model config (`indra-ocr`, concurrency 2,
engine threads 1, 180s page timeout, 100MB body limit, 50M pixel limit) is
used.

```yaml
models:
  - id: indra-ocr
    backend: indra
    model_path: /models/indra   # local folder; Indra downloads the default weights when omitted
    config: {}
      threads: 1
concurrency: 2                 # simultaneous page inference
engine_threads: 1              # CPU threads bound per engine instance
page_timeout_s: 180            # per-page inference deadline
max_body_mb: 100               # request body limit
max_image_pixels: 50000000     # decoded image pixel limit
```

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn ocr_server.main:create_app --factory
```

`OPENAI_OCR_SERVER_CONFIG` is optional; when unset, a default single-model
config (`indra-ocr`, concurrency 2, 1 engine thread, 180s page timeout,
100MB body limit, 50M pixel limit) is used. To override:

```bash
OPENAI_OCR_SERVER_CONFIG="$(cat config.yaml)" .venv/bin/uvicorn ocr_server.main:create_app --factory
```

## Docker

```bash
docker build -t openai-ocr-server .
docker run --rm -p 8000:8000 openai-ocr-server
```

The env var is optional. To override with a config file:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_OCR_SERVER_CONFIG="$(cat config.yaml)" \
  -v /models:/models:ro \
  openai-ocr-server
```

Point the adapter at it:

```yaml
ocr-adapter:
  environment:
    OCR_VLLM_URL: http://openai-ocr-server:8000
    OCR_MODEL: indra-ocr
```

### docker-compose example (server + adapter)

```yaml
services:
  openai-ocr-server:
    image: openai-ocr-server:0.1.0

  ocr-adapter:
    image: ghcr.io/virtuos/librechat-ocr-adapter:0.1.1
    environment:
      OCR_VLLM_URL: http://openai-ocr-server:8000
      OCR_MODEL: indra-ocr
      OCR_PUBLIC_URL: http://ocr-adapter:8000
    volumes:
      - type: tmpfs
        target: /uploads
        tmpfs:
          size: 512m
          mode: "1777"
    depends_on:
      - openai-ocr-server
```

`OPENAI_OCR_SERVER_CONFIG` is intentionally omitted: the default config (one `indra-ocr` model, concurrency 2, 1 engine thread, 180s page timeout, 100MB body limit, 50M pixel limit) is used. Indra downloads its default weights on first load.

Neither service publishes a port: both are reachable only over the private Docker network by service name (`openai-ocr-server:8000`, `ocr-adapter:8000`).

## Security

Do not publish the port publicly without authentication and TLS at a trusted
proxy. Image bytes and recognized content are never logged; only request IDs,
selected backend, timings and error categories are.
