FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

ENV OPENAI_OCR_SERVER_CONFIG="" \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ocr_server.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
