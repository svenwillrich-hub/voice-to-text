# ─────────────────────────────────────────────────────────────────────────────
# Whisper API — Docker Image
# Base: Python 3.11 slim + ffmpeg for audio decoding
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies: ffmpeg is required by faster-whisper for audio decoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer order)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Pre-download model at build time (optional — speeds up first start)
# Uncomment the line below to bake the model into the image:
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"

EXPOSE 8000

# Run with uvicorn — single worker is fine since the model is CPU/GPU-bound
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
