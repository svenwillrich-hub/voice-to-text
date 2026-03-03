"""
Text API — Translation & Text Processing Service
==================================================
FastAPI service using Ollama for local LLM-powered text translation
and processing (summarization, optimization, custom prompts).

Models:
  - TranslateGemma 12B  → dedicated translation
  - Gemma 3 12B         → summarization, text optimization, custom prompts
"""

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langdetect import detect, LangDetectException

# ── Configuration ────────────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "translategemma:12b")
PROCESSING_MODEL = os.getenv("PROCESSING_MODEL", "gemma3:12b")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("text-api")

# ── Supported Languages (TranslateGemma 12B) ────────────────────────────────

LANGUAGES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}

# ── State ────────────────────────────────────────────────────────────────────

translation_model_ready = False
processing_model_ready = False
http_client: httpx.AsyncClient | None = None

# ── Ollama helpers ───────────────────────────────────────────────────────────


async def wait_for_ollama():
    """Wait until Ollama is reachable (up to ~120 s)."""
    for attempt in range(60):
        try:
            r = await http_client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            r.raise_for_status()
            logger.info("Ollama is ready at %s", OLLAMA_HOST)
            return True
        except Exception:
            if attempt % 10 == 0:
                logger.info(
                    "Waiting for Ollama at %s (attempt %d/60) …",
                    OLLAMA_HOST,
                    attempt + 1,
                )
            await asyncio.sleep(2)
    logger.error("Ollama not reachable after 120 s")
    return False


async def model_available(model_name: str) -> bool:
    """Check whether a model is already pulled in Ollama."""
    try:
        r = await http_client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        base = model_name.split(":")[0]
        return any(model_name == m or m.startswith(base + ":") for m in models)
    except Exception:
        return False


async def pull_model(model_name: str) -> bool:
    """Pull a model from the Ollama registry. Streams progress to logs."""
    logger.info("Pulling model '%s' — this may take a while …", model_name)
    try:
        async with http_client.stream(
            "POST",
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            last_log = 0
            async for line in resp.aiter_lines():
                now = time.monotonic()
                if now - last_log > 5:          # log every 5 s max
                    logger.info("[pull %s] %s", model_name, line.strip()[:120])
                    last_log = now
        logger.info("Model '%s' pulled successfully", model_name)
        return True
    except Exception as exc:
        logger.error("Failed to pull model '%s': %s", model_name, exc)
        return False


async def ensure_model(model_name: str) -> bool:
    """Return True once the model is available in Ollama, pulling if needed."""
    if await model_available(model_name):
        logger.info("Model '%s' is already available", model_name)
        return True
    return await pull_model(model_name)


async def init_models():
    """Background task: wait for Ollama, then pull both models."""
    global translation_model_ready, processing_model_ready

    if not await wait_for_ollama():
        return

    # Pull translation model
    translation_model_ready = await ensure_model(TRANSLATION_MODEL)

    # Pull processing model (skip if same as translation model)
    if PROCESSING_MODEL == TRANSLATION_MODEL:
        processing_model_ready = translation_model_ready
    else:
        processing_model_ready = await ensure_model(PROCESSING_MODEL)

    logger.info(
        "Init complete — translation=%s (%s), processing=%s (%s)",
        TRANSLATION_MODEL,
        "ready" if translation_model_ready else "FAILED",
        PROCESSING_MODEL,
        "ready" if processing_model_ready else "FAILED",
    )


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    asyncio.create_task(init_models())
    yield
    await http_client.aclose()


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Text API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    source_language: str | None = Field(
        None, description="ISO 639-1 code, or null for auto-detect"
    )
    target_language: str = Field(..., description="ISO 639-1 language code")


class ProcessAction(str, Enum):
    summarize = "summarize"
    optimize = "optimize"
    custom = "custom"


class ProcessRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    action: ProcessAction
    custom_prompt: str | None = None


class DetectRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        r = await http_client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if (ollama_ok and translation_model_ready and processing_model_ready) else "loading",
        "ollama_connected": ollama_ok,
        "translation_model": TRANSLATION_MODEL,
        "translation_model_ready": translation_model_ready,
        "processing_model": PROCESSING_MODEL,
        "processing_model_ready": processing_model_ready,
    }


@app.get("/languages")
async def get_languages():
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in sorted(LANGUAGES.items(), key=lambda x: x[1])
        ]
    }


@app.post("/detect")
async def detect_language(req: DetectRequest):
    if not req.text.strip():
        return {"language_code": "unknown", "language_name": "Unknown"}
    try:
        code = detect(req.text)
        name = LANGUAGES.get(code, code)
        return {"language_code": code, "language_name": name}
    except LangDetectException:
        return {"language_code": "unknown", "language_name": "Unknown"}


@app.post("/translate")
async def translate(req: TranslateRequest):
    if not translation_model_ready:
        raise HTTPException(
            503,
            detail=f"Translation model '{TRANSLATION_MODEL}' is still loading. Please wait.",
        )

    if not req.text.strip():
        raise HTTPException(400, detail="Text must not be empty.")

    if req.target_language not in LANGUAGES:
        raise HTTPException(
            400, detail=f"Unsupported target language: {req.target_language}"
        )

    # Auto-detect source language
    source_code = req.source_language
    if not source_code:
        try:
            source_code = detect(req.text)
        except LangDetectException:
            source_code = "en"

    source_name = LANGUAGES.get(source_code, source_code)
    target_name = LANGUAGES.get(req.target_language, req.target_language)

    prompt = (
        f"Translate the following text from {source_name} to {target_name}:\n"
        f"{req.text}"
    )

    logger.info(
        "Translating %d chars: %s → %s",
        len(req.text),
        source_name,
        target_name,
    )
    t0 = time.perf_counter()

    try:
        r = await http_client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": TRANSLATION_MODEL, "prompt": prompt, "stream": False},
            timeout=300.0,
        )
        r.raise_for_status()
        translated = r.json().get("response", "").strip()
        elapsed = time.perf_counter() - t0

        logger.info("Translation done in %.1f s (%d chars out)", elapsed, len(translated))

        return {
            "translated_text": translated,
            "source_language": source_code,
            "source_language_name": source_name,
            "target_language": req.target_language,
            "target_language_name": target_name,
            "duration_processing": round(elapsed, 2),
        }
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama error: %s", exc.response.text[:300])
        raise HTTPException(502, detail="Translation model error") from exc
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        raise HTTPException(500, detail=f"Translation failed: {exc}") from exc


@app.post("/process")
async def process_text(req: ProcessRequest):
    if not processing_model_ready:
        raise HTTPException(
            503,
            detail=f"Processing model '{PROCESSING_MODEL}' is still loading. Please wait.",
        )

    if not req.text.strip():
        raise HTTPException(400, detail="Text must not be empty.")

    # Build prompt based on action
    if req.action == ProcessAction.summarize:
        system_msg = (
            "You are a concise text summarizer. "
            "Output only the summary, nothing else. "
            "Keep the same language as the input text."
        )
        user_msg = f"Summarize the following text concisely:\n\n{req.text}"

    elif req.action == ProcessAction.optimize:
        system_msg = (
            "You are a professional text editor. "
            "Clean up the given text: remove contradictions, excessive repetitions, "
            "filler words, and false starts. Improve clarity and readability while "
            "preserving the original meaning and language. "
            "Output only the improved text, nothing else."
        )
        user_msg = f"Optimize and clean up this transcribed text:\n\n{req.text}"

    elif req.action == ProcessAction.custom:
        if not req.custom_prompt:
            raise HTTPException(400, detail="custom_prompt is required for action 'custom'.")
        system_msg = (
            "You are a helpful text processing assistant. "
            "Follow the user's instructions precisely. "
            "Output only the result, nothing else."
        )
        user_msg = f"{req.custom_prompt}\n\n{req.text}"

    else:
        raise HTTPException(400, detail=f"Unknown action: {req.action}")

    logger.info("Processing %d chars (action=%s)", len(req.text), req.action.value)
    t0 = time.perf_counter()

    try:
        r = await http_client.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": PROCESSING_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
            timeout=300.0,
        )
        r.raise_for_status()
        result = r.json().get("message", {}).get("content", "").strip()
        elapsed = time.perf_counter() - t0

        logger.info("Processing done in %.1f s (%d chars out)", elapsed, len(result))

        return {
            "result_text": result,
            "action": req.action.value,
            "duration_processing": round(elapsed, 2),
        }
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama error: %s", exc.response.text[:300])
        raise HTTPException(502, detail="Processing model error") from exc
    except Exception as exc:
        logger.error("Processing failed: %s", exc)
        raise HTTPException(500, detail=f"Processing failed: {exc}") from exc
