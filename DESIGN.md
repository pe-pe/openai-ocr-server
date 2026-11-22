# openai-ocr-server design

## Purpose

`openai-ocr-server` is a self-hosted OCR inference service with an
OpenAI-compatible chat-completions API. It lets clients submit images or PDFs
and use CPU-oriented OCR engines such as Indra OCR or PaddleOCR without
implementing a separate client for each engine.

The initial integration target is `librechat-ocr-adapter`. The OCR server
replaces the vLLM service that the adapter sends inference requests to; the
adapter's LibreChat-facing Mistral OCR API does not need to change.

## System boundary

```text
LibreChat
    | Mistral OCR API
    v
librechat-ocr-adapter
    | OpenAI-compatible chat completion containing one page image
    v
openai-ocr-server
    | backend interface
    +-- Indra OCR
    +-- PaddleOCR
    +-- other...
```

The adapter owns its LibreChat-facing file storage, PDF parsing, page
selection, and PDF page rendering. It uses a PDF text layer when suitable and
sends only pages needing OCR to the inference server. Each adapter request
contains one rendered page as an image data URL.

The OCR server is independently useful: its same chat-completions endpoint
accepts a single image, multiple image parts, or a PDF file part. For a PDF,
the server renders and OCRs the pages itself, then returns their results in
document order. The adapter continues to use the single-image form.

## Compatibility contract

### Request

Implement:

```http
POST /v1/chat/completions
Content-Type: application/json
```

The endpoint accepts the standard Chat Completions user-message content forms
needed for OCR:

- One or more `image_url` parts. A single image is treated as one page;
  multiple images are processed in their supplied order.
- A `file` part containing a PDF. The server validates and renders the PDF,
  then OCRs its pages in document order. Chat Completions file input is
  supported for PDFs; broader document file types are not part of this
  endpoint's initial scope.

For example, the PDF content part uses the standard Chat Completions `file`
shape:

```json
{
  "type": "file",
  "file": {
    "filename": "document.pdf",
    "file_data": "data:application/pdf;base64,..."
  }
}
```

The adapter sends a request with a `model`, sampling parameters, token limit,
and one user message containing an `image_url` part. Its image URL is a
base64 PNG data URL. For the initial implementation, accept base64 image and
PDF data directly; remote URLs and `file_id` inputs require additional
retrieval or Files API support and are not required for the adapter
integration. This is a deliberately scoped OpenAI-compatible Chat
Completions interface, not an implementation of every OpenAI endpoint or
input modality. A Responses API endpoint can be added later on the same OCR
core if there is a client need for it.

The request's `model` selects an installed model from the server's model
registry. Each public model ID maps to a backend and its local model
configuration, for example `indra-ocr` or a `paddleocr`. The server
must reject unknown model IDs explicitly rather than silently falling back to
a default. Implement `GET /v1/models` to list the model IDs available for
requests.

The adapter currently sends the single model name configured by its
`OCR_MODEL` setting, so that adapter instance uses one selected model. Other
OpenAI-compatible clients can choose among the server's installed models by
setting `model` on each request.

The OCR server may ignore generation-only parameters that do not apply to a
backend, but should validate requests and return explicit errors for malformed
or unsupported inputs.

### Response

Return a standard non-streaming chat completion with OCR output as the
assistant message content. Echo the selected model ID in the response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "indra-ocr",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Recognized text or Markdown"
      },
      "finish_reason": "stop"
    }
  ]
}
```

The adapter reads `choices[0].message.content`; it does not need token usage
fields. For engines such as PaddleOCR that return recognized lines, boxes, and
confidence values, the backend adapter should order the lines and convert them
to plain text or Markdown before returning the completion. For a multi-page
PDF or multiple images, preserve page order and separate page output clearly,
for example with `## Page 1` headings.

### Health

Implement:

```http
GET /health
```

Return HTTP 200 when the service and its model registry are ready to accept
requests. The LibreChat OCR adapter uses this endpoint to check upstream
availability. During server startup, return a non-200 status until the
registry is initialized.

## Backend interface

OCR backends should:

1. Use model files and runtime dependencies installed in the image or made
   available through a local mounted model directory; do not download models
   as part of each request.
2. Accept a decoded image for a single page.
3. Return recognized text, preferably with optional structured lines/blocks
   for formatting.
4. Report readiness and surface inference errors.

The HTTP layer is responsible for request validation, image/PDF data decoding,
backend selection from the model registry, response formatting, and error
mapping. The document layer is responsible for PDF rendering and page
ordering. The backend layer is responsible for model-specific preprocessing,
inference, and conversion of native output into a common OCR result.

Indra accepts an in-memory image; PaddleOCR also accepts image input. The
server can decode the incoming PNG into an image array or a temporary image
file as required by the selected engine. Backend-specific dependencies and
model loading should remain isolated so adding an engine does not change the
OpenAI API contract.

## Runtime and concurrency

- Keep model instances warm instead of loading one for every request.
- Register every configured model ID at startup. Model weights may be loaded
  at startup or lazily on first use, then reused; make the choice explicit
  because keeping all models resident uses more RAM.
- Run CPU-bound inference outside the asynchronous HTTP event loop.
- Make request concurrency configurable and start conservatively. Bound both
  simultaneous page inference and each engine's internal CPU threads to avoid
  oversubscribing the host.
- Apply request and inference deadlines. Return an explicit error on timeout
  or backend failure; do not return an empty, success-shaped OCR result.
- Keep model assets available locally in the image or a mounted volume so
  container recreation and model switching do not require downloads.

## Security

- Keep the service on a private container network by default; do not publish
  its port publicly without authentication and TLS at a trusted proxy.
- Accept base64 image/PDF content rather than fetching arbitrary
  URLs or file references. This avoids turning the inference service into an
  unrestricted URL-fetching proxy.
- Enforce request-body, PDF page-count, decoded-image pixel, and per-page
  inference limits before and during processing.
- Do not log image bytes or recognized document content. Log request IDs,
  selected backend, timings, and error categories instead.
- Review each backend's code and model-weight licenses before distribution or
  commercial use. These licenses can differ by backend.

## Container integration

Run the OCR server and `librechat-ocr-adapter` on the same private Docker
network. Configure the adapter to use the OCR server's service name and port:

```yaml
ocr-adapter:
  environment:
    OCR_VLLM_URL: http://openai-ocr-server:8000
    OCR_MODEL: indra-ocr
```

The `OCR_VLLM_URL` name is retained for compatibility with the current
adapter; it points to the OpenAI-compatible OCR server, not necessarily vLLM.
Do not expose the adapter's unauthenticated API publicly. Put its LibreChat
endpoint behind the authentication/reverse-proxy arrangement documented by
the adapter.

## Initial scope

### In scope

- OpenAI-compatible, non-streaming `POST /v1/chat/completions`.
- Single-image, multiple-image, and PDF file-part requests.
- Ordered, page-separated text/Markdown for multi-page inputs.
- CPU backends behind a common interface, initially Indra OCR and PaddleOCR.
- Multiple locally installed model choices, selected by the request's `model`
  field, with `GET /v1/models` discovery.
- Configurable CPU concurrency.
- `GET /health` for adapter readiness checks.
- Docker deployment with local model assets.

### Out of scope

- Replacing or modifying the adapter's Mistral-compatible API.
- Accepting LibreChat's file-upload lifecycle or non-PDF document formats.
- Streaming completions, chat history, or general-purpose LLM chat.
- The OpenAI Files API and `file_id` inputs in the initial implementation.
- Exposing backend-specific structured OCR output through LibreChat's current
  adapter response. The initial result is text or Markdown in the completion.

## Validation

Before calling a backend compatible, test the actual adapter request and
response path:

1. Submit one base64 PNG image using the same body shape as the adapter.
2. Submit multiple image parts and verify their order in the response.
3. Submit a multi-page PDF file part and verify ordered, page-separated output.
4. Verify the selected model name routes to the intended backend.
5. Verify recognized output appears in `choices[0].message.content`.
6. Verify `/health` returns 200 only when ready.
7. Exercise invalid image/PDF data, unsupported model names, size/page limits,
   backend failures, and inference timeouts.
8. Run end-to-end OCR through LibreChat's adapter for an image and a scanned
   PDF page.
