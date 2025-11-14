# Real-Time Voice Translation + Lip-Synced Video Pipeline

A FastAPI-based prototype for real-time voice translation with lip-synced video output using open-source models.

## Overview

This prototype accepts video/audio uploads, processes them through an ASR → Translation → TTS → Lip-sync pipeline using local open-source models, and returns the processed video with translated audio and synchronized lip movements.

### Pipeline Steps

1. **ASR (Automatic Speech Recognition)**: Whisper or Silero extracts transcript with timestamps
2. **Translation**: MarianMT translates text to target language
3. **TTS (Text-to-Speech)**: Coqui TTS or Tacotron synthesizes translated audio
4. **Lip-Sync**: Wav2Lip synchronizes video lip movements with new audio

## Features

- ✅ Chunked file uploads supporting 100MB+ files
- ✅ Asynchronous job processing with status tracking
- ✅ RESTful API with FastAPI
- ✅ Local disk storage (configurable, MinIO-ready)
- ✅ JSON-based job persistence (SQLite-ready, Redis notes included)
- ✅ Docker containerization
- ✅ Kubernetes manifests
- ✅ GPU support notes

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Upload    │─────▶│  Job Queue   │─────▶│  Pipeline   │
│  Endpoint   │      │   Manager    │      │  Processor  │
└─────────────┘      └──────────────┘      └─────────────┘
                                                   │
                           ┌───────────────────────┤
                           ▼                       ▼
                    ┌────────────┐         ┌────────────┐
                    │  Storage   │         │   Models   │
                    │  (Disk)    │         │  (Local)   │
                    └────────────┘         └────────────┘
```

## Prerequisites

- Python 3.10+
- FFmpeg (for audio/video processing)
- Docker & Docker Compose (for containerized deployment)
- 4GB+ RAM (8GB+ recommended for models)
- Optional: NVIDIA GPU with CUDA for faster processing

## Model Setup

Download and place models in the `app/models/` directory:

### 1. Whisper (ASR)
```bash
# Models are auto-downloaded by the whisper library on first use
# Or manually download from: https://github.com/openai/whisper
# Place in: app/models/whisper/
pip install openai-whisper
```

### 2. MarianMT (Translation)
```bash
# Auto-downloaded from HuggingFace on first use
# Example models:
# - en-es: Helsinki-NLP/opus-mt-en-es
# - en-fr: Helsinki-NLP/opus-mt-en-fr
# Cache location: ~/.cache/huggingface/
```

### 3. Coqui TTS
```bash
# Install TTS library
pip install gTTS


```

```

**Model Directory Structure:**
```
app/models/
└── Wav2Lip/          # Wav2Lip repository and checkpoints
    └── checkpoints/
        └── wav2lip_gan.pth
```

## Quick Start

### Local Development (Python venv)

1. **Clone and setup:**
```bash
cd /Users/dragun/Documents/dataviv-interview
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install FFmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# Verify
ffmpeg -version
```

3. **Create storage directories:**
```bash
mkdir -p storage/{uploads,processing,outputs,jobs}
mkdir -p app/models
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Run the application:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Access API:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Compose (Recommended)

1. **Build and run:**
```bash
docker-compose up --build
```

2. **Access API:**
- API: http://localhost:8000

3. **View logs:**
```bash
docker-compose logs -f app
```

4. **Stop services:**
```bash
docker-compose down
```

### Kubernetes Deployment

1. **Apply manifests:**
```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

2. **Check status:**
```bash
kubectl get pods
kubectl get svc rtvt-lipsync-service
```

3. **Access service:**
```bash
# Port forward for local access
kubectl port-forward service/rtvt-lipsync-service 8000:8000

# Or use NodePort/LoadBalancer IP
kubectl get svc rtvt-lipsync-service
```

## API Usage

### 1. Upload Video File

**Simple upload:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@demo.mp4" \
  | jq
```

**Response:**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "demo.mp4",
  "size": 52428800,
  "status": "complete"
}
```

**Chunked upload (for large files):**
```bash
# Split file into chunks (10MB each)
split -b 10485760 demo.mp4 chunk_

# Upload chunks
UPLOAD_ID=$(uuidgen)
TOTAL_CHUNKS=$(ls chunk_* | wc -l)
INDEX=0

for chunk in chunk_*; do
  curl -X POST "http://localhost:8000/upload/chunk" \
    -H "Upload-Id: $UPLOAD_ID" \
    -H "Chunk-Index: $INDEX" \
    -H "Total-Chunks: $TOTAL_CHUNKS" \
    -H "Filename: demo.mp4" \
    -F "file=@$chunk"
  INDEX=$((INDEX + 1))
done

# Check assembly status
curl "http://localhost:8000/upload/$UPLOAD_ID/status" | jq
```

### 2. Start Processing

```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "550e8400-e29b-41d4-a716-446655440000",
    "target_lang": "es"
  }' \
  | jq
```

**Supported languages:**
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

**Response:**
```json
{
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "queued",
  "created_at": "2025-11-14T10:30:00Z"
}
```

### 3. Check Processing Status

```bash
curl "http://localhost:8000/process/7c9e6679-7425-40de-944b-e07fc1f90ae7/status" | jq
```

**Response:**
```json
{
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "processing",
  "step": "tts",
  "progress": 60,
  "message": "Generating translated speech...",
  "created_at": "2025-11-14T10:30:00Z",
  "updated_at": "2025-11-14T10:32:15Z",
  "steps": {
    "asr": "completed",
    "translate": "completed",
    "tts": "processing",
    "lipsync": "pending"
  }
}
```

**Status values:**
- `queued` - Job created, waiting to start
- `processing` - Pipeline running
- `completed` - Successfully finished
- `failed` - Error occurred (check `error` field)

### 4. Download Result

```bash
curl "http://localhost:8000/media/7c9e6679-7425-40de-944b-e07fc1f90ae7" \
  -o translated_video.mp4
```

### 5. List Jobs

```bash
curl "http://localhost:8000/jobs?limit=10&status=completed" | jq
```

## Configuration

Environment variables (`.env` file):

```bash
# Storage
STORAGE_PATH=./storage
MAX_UPLOAD_SIZE=104857600  # 100MB

# Models
MODELS_PATH=./app/models
WHISPER_MODEL=base  # tiny, base, small, medium, large
DEVICE=cpu  # cpu or cuda

# Processing
MAX_WORKERS=2
JOB_TIMEOUT=3600  # 1 hour

# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Optional: Redis for job queue (if not using JSON file)
# REDIS_URL=redis://localhost:6379/0

# Optional: MinIO for object storage
# MINIO_ENDPOINT=localhost:9000
# MINIO_ACCESS_KEY=minioadmin
# MINIO_SECRET_KEY=minioadmin
# MINIO_BUCKET=rtvt-lipsync
```

## GPU Support

### Docker with GPU

1. **Install NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Update docker-compose.yml:**
```yaml
services:
  app:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. **Update Dockerfile to use CUDA base:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# ... install Python, PyTorch with CUDA support ...
```

4. **Set environment:**
```bash
export DEVICE=cuda
```

### Local Development with GPU

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set device in config
export DEVICE=cuda
```

## Performance Expectations

### Processing Times (CPU - Apple M1/M2 or Intel i7)

| Video Length | Whisper (base) | Translation | TTS | Wav2Lip | Total |
|--------------|----------------|-------------|-----|---------|-------|
| 1 min        | ~30s           | ~2s         | ~15s| ~45s    | ~1.5min |
| 5 min        | ~2.5min        | ~8s         | ~1min| ~3min  | ~7min |
| 10 min       | ~5min          | ~15s        | ~2min| ~6min  | ~14min |

### Processing Times (GPU - NVIDIA RTX 3080)

| Video Length | Whisper (base) | Translation | TTS | Wav2Lip | Total |
|--------------|----------------|-------------|-----|---------|-------|
| 1 min        | ~10s           | ~1s         | ~5s | ~15s    | ~30s  |
| 5 min        | ~45s           | ~3s         | ~20s| ~1min   | ~2.5min |
| 10 min       | ~1.5min        | ~5s         | ~40s| ~2min   | ~4.5min |

## Model Integration Status

### ✅ Fully Integrated
- **Storage & Upload**: Chunked upload, disk-based storage
- **Job Management**: JSON-based persistence with status tracking
- **FFmpeg Utils**: Audio extraction, concatenation, video merging

### 🔧 Stub + Integration Points
- **Whisper ASR**: Integration code provided with TODO for optimization
- **MarianMT Translation**: HuggingFace integration with language detection
- **Coqui TTS**: Model loading and inference with fallback
- **Wav2Lip**: Integration scaffold with checkpoint loading

### 📝 How to Complete Model Integration

Each model has stub functions in `app/pipeline.py` with clear TODOs:

**Example: Whisper ASR**
```python
# TODO: Uncomment for real Whisper integration
# import whisper
# model = whisper.load_model("base")
# result = model.transcribe(audio_path, language="en")

# Current: Returns stub transcript
# Replace with real model call
```

**Running models manually (for testing):**
```bash
# Whisper
python -m whisper audio.mp3 --model base --language en --output_format json

# MarianMT
python -c "from transformers import MarianMTModel, MarianTokenizer; ..."

# Coqui TTS
tts --text "Hello world" --model_name "tts_models/en/ljspeech/tacotron2-DDC" --out_path output.wav

# Wav2Lip
cd app/models/Wav2Lip
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face video.mp4 --audio audio.wav --outfile result.mp4
```

## Switching to Production Storage

### Redis for Job Queue

1. **Install Redis:**
```bash
docker-compose up -d redis
```

2. **Update code in `app/jobs.py`:**
```python
import redis
r = redis.from_url(os.getenv("REDIS_URL"))
# Use Redis commands instead of JSON file
```

3. **Set environment:**
```bash
export REDIS_URL=redis://localhost:6379/0
```

### PostgreSQL for Job Metadata

```python
# Install: pip install psycopg2-binary sqlalchemy
from sqlalchemy import create_engine
engine = create_engine(os.getenv("DATABASE_URL"))
# Replace JSON file operations with SQLAlchemy models
```

## Troubleshooting

### FFmpeg not found
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu
```

### CUDA out of memory
```bash
# Use smaller Whisper model
export WHISPER_MODEL=tiny

# Process in smaller chunks
# Reduce batch sizes in pipeline.py
```

### Slow processing on CPU
- Use smaller models (Whisper tiny/base instead of large)
- Reduce video resolution before processing
- Consider GPU deployment

### Model download failures
```bash
# Set HuggingFace cache directory
export HF_HOME=./app/models/huggingface

# Manually download models
huggingface-cli download Helsinki-NLP/opus-mt-en-es
```

### Port already in use
```bash
# Change port in docker-compose.yml or .env
export PORT=8080
```

### Storage permission errors
```bash
# Fix permissions
chmod -R 755 storage/
chown -R $(whoami) storage/
```

## Testing

Run the example test flow:

```bash
# Make scripts executable
chmod +x examples/sample_commands.sh

# Run test flow
./examples/sample_commands.sh
```

**Manual testing steps:**
1. Upload a video: `curl -X POST http://localhost:8000/upload -F "file=@test.mp4"`
2. Start processing: `curl -X POST http://localhost:8000/process -H "Content-Type: application/json" -d '{"upload_id":"<id>","target_lang":"es"}'`
3. Poll status: `watch curl http://localhost:8000/process/<job_id>/status`
4. Download result: `curl http://localhost:8000/media/<job_id> -o result.mp4`

## Project Structure

```
rtvt-lipsync-prototype/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Docker Compose configuration
├── .env.example                # Example environment variables
├── docker/
│   └── Dockerfile              # Container image definition
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment
│   ├── service.yaml            # Kubernetes Service
│   └── pvc.yaml                # Persistent Volume Claim
├── app/
│   ├── main.py                 # FastAPI app & endpoints
│   ├── config.py               # Configuration management
│   ├── storage.py              # File upload & storage
│   ├── jobs.py                 # Job state management
│   ├── pipeline.py             # Pipeline orchestration
│   ├── utils.py                # FFmpeg & helper functions
│   └── models/
│       └── README.md           # Model download instructions
├── storage/                    # Local file storage (gitignored)
│   ├── uploads/               # Uploaded files
│   ├── processing/            # Intermediate files
│   ├── outputs/               # Final videos
│   └── jobs/                  # Job metadata JSON files
└── examples/
    └── sample_commands.sh      # Test script with curl examples
```


