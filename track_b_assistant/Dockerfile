FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# curl is used by the docker compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# install CPU-only torch first (the default torch wheel includes CUDA and is ~2GB bigger)
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download the embedding model at build time so the container starts fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests

# convert the embedding model to ONNX + INT8 inside the image
RUN python scripts/export_onnx.py

ENV USE_ONNX_EMBEDDINGS=true \
    ONNX_MODEL_FILE=model_quantized.onnx

# run as a normal user instead of root
RUN useradd --create-home appuser && mkdir -p /app/chroma_db && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

# default command = API. docker compose overrides it for the UI container.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
