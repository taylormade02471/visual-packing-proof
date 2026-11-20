FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && \
    python -c "from visual_packing.search import MobileNetV3Encoder; MobileNetV3Encoder()"

COPY artifacts ./artifacts
COPY data ./data
COPY docs ./docs

EXPOSE 8080
CMD ["sh", "-c", "uvicorn visual_packing.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
