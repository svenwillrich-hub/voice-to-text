# Voice-to-Text

Self-hosted audio transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No data leaves your server. No API keys. Fully offline.

## Quick Start

### Option A: Docker Compose (Development)

```bash
cp .env.example .env
docker compose up --build
```

Open **http://localhost**. The Whisper model downloads automatically on first start (~1.5 GB for `large-v3-turbo`) and is cached in a Docker volume.

### Option B: Single Image (Production / VPS)

Build and run the all-in-one image:

```bash
docker build -t voice-to-text .
docker run -d -p 80:80 -v whisper-cache:/root/.cache/huggingface --name voice-to-text voice-to-text
```

### GPU Support

```bash
# Docker Compose
docker compose --profile gpu up --build

# Single image
docker run -d --gpus all \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -p 80:80 -v whisper-cache:/root/.cache/huggingface \
  voice-to-text
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Publish to GitHub Container Registry

```bash
docker build -t ghcr.io/YOUR_USER/voice-to-text:latest .
docker push ghcr.io/YOUR_USER/voice-to-text:latest
```

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU), `float32` |
| `WHISPER_BEAM_SIZE` | `5` | Higher = more accurate, slower |
| `MAX_UPLOAD_MB` | `500` | Max upload size in MB |

## Architecture

```
Browser ──▶ nginx (:80)
              ├── /       → static frontend (index.html)
              └── /api/*  → FastAPI + faster-whisper (:8000)
```

**Docker Compose** runs these as two separate containers. The **all-in-one Dockerfile** bundles both into a single image using supervisord.

## API

`GET /health` — Service status and model info.

`POST /transcribe` — Upload audio file (`multipart/form-data`), returns JSON with full transcript, segments, detected language, and timing.

## License

MIT — Provided by Dr. Sven-Erik Willrich · mail@svenwillrich.de
