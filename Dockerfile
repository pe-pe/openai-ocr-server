FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

RUN mkdir -p /models

ENV OPENAI_OCR_SERVER_CONFIG="" \
    PORT=8000

VOLUME ["/models"]

EXPOSE ${PORT}

CMD ["sh", "-c", "exec uvicorn ocr_server.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
