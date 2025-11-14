FROM python:3.10-bullseye

LABEL maintainer="RTVT-LipSync Team"
LABEL description="Real-Time Voice Translation + Lip-Synced Video Pipeline"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ffmpeg -version

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/storage/uploads \
             /app/storage/processing \
             /app/storage/outputs \
             /app/storage/jobs \
             /app/storage/logs \
             /app/app/models

COPY app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STORAGE_PATH=/app/storage \
    MODELS_PATH=/app/app/models \
    DEVICE=cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
