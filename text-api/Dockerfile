# ─────────────────────────────────────────────────────────────────────────────
# Text API — Docker Image
# FastAPI service for translation (TranslateGemma) and text processing (Gemma 3)
# Communicates with Ollama for LLM inference.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]
