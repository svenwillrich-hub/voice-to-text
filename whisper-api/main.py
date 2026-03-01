"""
Voice-to-Text — Whisper Transcription API
==========================================
FastAPI service wrapping faster-whisper for high-performance local transcription.
Supports all common audio formats including WhatsApp voice messages (.opus, .ogg).
"""

import os
import time
import uuid
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ── Configuration ────────────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Supported audio MIME types / extensions
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".m4a", ".opus", ".aac",
    ".flac", ".wma", ".webm", ".mp4", ".mpeg", ".mpga",
}

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whisper-api")

# ── Model Lifecycle ──────────────────────────────────────────────────────────

model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Whisper model once at startup, release on shutdown."""
    global model
    logger.info(
        "Loading model '%s' on %s (compute: %s) …",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    )
    t0 = time.perf_counter()
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1f s", elapsed)
    yield
    logger.info("Shutting down – releasing model")
    model = None


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice-to-Text Whisper API",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the frontend (running on a different port / container)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Returns service status and loaded model info."""
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
    }


# ── Transcription Endpoint ───────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(
        default=None,
        description="ISO 639-1 language code (e.g. 'de', 'en'). Auto-detect if omitted.",
    ),
    beam_size: int = Query(default=None, description="Beam size override"),
):
    """
    Transcribe an uploaded audio file.

    Accepts multipart/form-data with a single audio file.
    Returns JSON with the full transcript, per-segment details, and metadata.
    """
    if model is None:
        raise HTTPException(503, detail="Model is still loading. Please retry shortly.")

    # ── Validate file extension ──────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415,
            detail=(
                f"Unsupported format '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # ── Read & validate size ─────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            detail=f"File too large ({len(content) / 1e6:.1f} MB). Max: {MAX_UPLOAD_MB} MB.",
        )

    # ── Write to temp file (faster-whisper needs a file path) ────────────
    job_id = uuid.uuid4().hex[:8]
    suffix = ext or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"vtt_{job_id}_")
    tmp.write(content)
    tmp.close()
    tmp_path = tmp.name

    logger.info(
        "[%s] Transcribing '%s' (%.1f MB, ext=%s) …",
        job_id, file.filename, len(content) / 1e6, ext,
    )

    try:
        t0 = time.perf_counter()

        # ── Run transcription ────────────────────────────────────────────
        segments_gen, info = model.transcribe(
            tmp_path,
            beam_size=beam_size or WHISPER_BEAM_SIZE,
            language=language,
            vad_filter=True,          # skip silence for speed
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Materialise segments
        segments = []
        full_text_parts = []
        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        elapsed = time.perf_counter() - t0
        full_text = " ".join(full_text_parts)

        logger.info(
            "[%s] Done in %.1f s — language=%s (p=%.0f%%), %d segments, %d chars",
            job_id, elapsed, info.language, info.language_probability * 100,
            len(segments), len(full_text),
        )

        return JSONResponse({
            "job_id": job_id,
            "filename": file.filename,
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_audio": round(info.duration, 2),
            "duration_processing": round(elapsed, 2),
        })

    except Exception as exc:
        logger.exception("[%s] Transcription failed", job_id)
        raise HTTPException(500, detail=f"Transcription failed: {exc}") from exc

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
