FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    MODEL_RELEASE_TAG=v1.2.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install matching CPU-only PyTorch wheels to avoid CUDA image overhead.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       torch==2.5.1+cpu torchaudio==2.5.1+cpu \
       --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<2.0" \
    "pandas>=2.1" \
    "soundfile>=0.12" \
    "fastapi>=0.115" \
    "uvicorn>=0.30" \
    "python-multipart>=0.0.9" \
    "requests>=2.31"

COPY api/ ./api/
COPY src/ ./src/
COPY models/final_test_metrics.json ./models/final_test_metrics.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
