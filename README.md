# openai-ocr-server

Self-hosted OCR inference service with an OpenAI-compatible chat-completions
API (`POST /v1/chat/completions`, `GET /v1/models`, `GET /health`). It is a
drop-in replacement for the vLLM upstream used by
[librechat-ocr-adapter](https://github.com/virtuos/librechat-ocr-adapter).

The service ships two CPU backends behind a single agnostic backend
interface: Indra OCR (the default when `backend` is omitted) and RapidDoc.
RapidDoc adds layout-aware document parsing with optional formula and table
recognition on top of OCR. The embedded default configuration registers both
models. Only one base64 PNG image per request is handled in the MVP;
multi-image and server-side PDF are deferred.

## Configuration

All configuration is a single YAML document passed through the
`OPENAI_OCR_SERVER_CONFIG` environment variable. When set, it replaces the
default. When unset, the embedded
[`default-config.yaml`](src/ocr_server/default-config.yaml) is used. The default config is embedded in the
Docker image.

```yaml
models:
  - id: indra-ocr
    backend: indra-ocr
    model_path: /models/indra-ocr
    config:
      threads: 1
  - id: rapid-doc
    backend: rapid-doc
    model_path: /models/rapid-doc
    config:
      lang: en
      parse_method: auto             # auto, txt or ocr
      formula_enable: true
      table_enable: true
      image_output_mode: data_uri    # embed extracted figures in Markdown
concurrency: 2                 # simultaneous page inference
engine_threads: 1              # CPU threads bound per Indra engine instance
page_timeout_s: 180            # per-page inference deadline
max_body_mb: 100               # request body limit
max_image_pixels: 50000000     # decoded image pixel limit
```

Each entry in `models` maps a public model ID (sent as `model` by clients) to
a backend. `backend: indra-ocr` is the default when the field is omitted. If a
configured `model_path` cannot be created or written (for example a local
non-root run without `/models`), the backend falls back to its default cache
and logs a warning.

### Indra OCR backend

`backend: indra-ocr` is the default backend and runs Indra OCR on CPU. It returns
recognized text in reading order and Markdown when available.

Supported `config` keys:

| Key | Default | Description |
| --- | --- | --- |
| `threads` | `1` | ONNX Runtime CPU threads per Indra engine instance. |

`model_path` points to a local folder containing `det.onnx`, `rec.onnx` and
`meta.json`; `order.onnx` is optional. If the folder is missing or empty,
Indra downloads its default weights into it. When `model_path` is omitted, the
Hugging Face cache is used instead.

### RapidDoc backend

`backend: rapid-doc` runs the [RapidDoc](https://github.com/RapidAI/RapidDoc)
pipeline: OCR plus layout analysis, with formula and table recognition
optional. It runs on CPU through ONNX Runtime. Models are downloaded on first
load unless `model_path` (exported as `RAPID_MODELS_DIR`) or per-component
model paths are supplied.

Supported `config` keys:

| Key | Default | Description |
| --- | --- | --- |
| `lang` | `en` | RapidDoc OCR language (`en`, `ch`, ...). |
| `parse_method` | `auto` | `auto`, `txt` or `ocr`. Image inputs have no text layer, so `auto` selects OCR. |
| `formula_enable` | `true` | Enable formula recognition. Disable to reduce model downloads and startup time. |
| `table_enable` | `true` | Enable table recognition. Disable to reduce model downloads and startup time. |
| `image_output_mode` | `data_uri` | `data_uri` embeds extracted figures in the returned Markdown; `url` leaves relative image links. |
| `layout` | `{}` | RapidDoc `layout_config` overrides, e.g. `model_type` or `model_dir_or_path`. |
| `ocr` | `{}` | RapidDoc `ocr_config` overrides for the underlying RapidOCR models. |
| `formula` | `{}` | RapidDoc `formula_config` overrides. |
| `table` | `{}` | RapidDoc `table_config` overrides. |
| `checkbox` | `{}` | RapidDoc `checkbox_config` overrides (checkbox detection is off by default). |
| `image` | `{}` | RapidDoc `image_config` overrides. |
| `device` | `cpu` | Exported as `MINERU_DEVICE_MODE` before import (for example `cuda:0`). |
| `orientation` | `false` | Exported as `USE_DOC_ORIENTATION_CLASSIFY` to correct rotated pages. |
| `preload` | `true` | Load the RapidDoc pipeline during startup instead of on first request. |

RapidDoc uses one process-wide model cache, so `RAPID_MODELS_DIR` is shared by
all RapidDoc models in the process. RapidDoc inference is serialized across
all RapidDoc models because the pipeline is not documented as thread-safe.
Different backends can still run concurrently up to `concurrency`.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn ocr_server.main:create_app --factory
```

`OPENAI_OCR_SERVER_CONFIG` is optional; when unset, the embedded `default-config.yaml` is used. To override:

```bash
OPENAI_OCR_SERVER_CONFIG="$(cat config.yaml)" .venv/bin/uvicorn ocr_server.main:create_app --factory
```

## Testing

Fast unit tests do not download RapidDoc models:

```bash
.venv/bin/pytest
```

The RapidDoc end-to-end test downloads the layout model on first run and is
opt-in locally; CI enables it with `OCR_E2E_RAPID_DOC=1`:

```bash
OCR_E2E_RAPID_DOC=1 .venv/bin/pytest tests/test_e2e_ocr.py::test_e2e_rapid_doc
```

Set `OCR_E2E_RAPID_DOC_MODELS=/path/to/models` to reuse an existing RapidDoc
model directory.

## Docker

```bash
docker build -t openai-ocr-server .
docker run --rm -p 8000:8000 openai-ocr-server
```

The container listens on `$PORT` (default 8000). To use another port:
`docker run --rm -e PORT=9000 -p 9000:9000 openai-ocr-server`. `EXPOSE`
remains build-time metadata; the runtime `PORT` value controls uvicorn.

The env var is optional. To override with a config file and persist the
downloaded models:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_OCR_SERVER_CONFIG="$(cat config.yaml)" \
  -v /models:/models \
  openai-ocr-server
```

The `/models` mount must be writable: if it is empty, Indra and RapidDoc
download their weights into `/models/indra-ocr` and `/models/rapid-doc` on first
load.

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
    volumes:
      - ocr-models:/models

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

volumes:
  ocr-models:
```

`OPENAI_OCR_SERVER_CONFIG` is intentionally omitted.

Set `OCR_MODEL` to `indra-ocr` or `rapid-doc` to choose the backend used by the
adapter.

Neither service publishes a port: both are reachable only over the private Docker network by service name (`openai-ocr-server:8000`, `ocr-adapter:8000`).

## Security

Do not publish the port publicly without authentication and TLS at a trusted
proxy. Image bytes and recognized content are never logged; only request IDs,
selected backend, timings and error categories are.
