# Voice-to-Text + Text Translator

Self-hosted audio transcription and text translation powered by local AI models. No data leaves your server. No API keys. Fully offline.

## Features

| Feature | Description | Model |
|---------|-------------|-------|
| **Voice-to-Text** | Record audio or upload files, get instant transcriptions | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3-turbo`) |
| **Post-Processing** | Summarize, optimize, or transform transcribed text | [Gemma 3 12B](https://ai.google.dev/gemma) via Ollama |
| **Text Translation** | Translate text between 55 languages | [TranslateGemma 12B](https://blog.google/technology/developers/translategemma/) via Ollama |

## Architecture

```
                                ┌──────────────────────┐
 :1234  Voice-to-Text UI ─────▶│  nginx (frontend)    │
                                │  /api/*  → whisper   │
                                │  /text-api/* → text  │
                                └──────────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
          │  whisper-api     │  │  text-api         │  │  Ollama      │
          │  FastAPI +       │  │  FastAPI           │  │  LLM Runtime │
          │  faster-whisper  │  │  Translation &     │◀─│              │
          │  :8000           │  │  Processing :5000  │  │  :11434      │
          └──────────────────┘  └──────────────────┘  └──────────────┘

                                ┌──────────────────────┐
 :1235  Text Translator UI ───▶│  nginx (translator)  │
                                │  /api/* → text-api   │
                                └──────────────────────┘
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| `web-frontend` | 1234 | Voice-to-Text web UI |
| `web-translator` | 1235 | Text Translator web UI |
| `whisper-api` | 8000 | Audio transcription API |
| `text-api` | 5000 | Translation & text processing API |
| `ollama` | 11434 | Local LLM runtime |

## Models & Why They Were Chosen

### faster-whisper `large-v3-turbo` (Voice-to-Text)

- **What**: OpenAI Whisper model optimized with CTranslate2 for 4x faster inference
- **Why**: Best speed/quality trade-off for transcription. The `large-v3-turbo` variant (~1.5B parameters) delivers near `large-v3` quality at significantly faster speeds. Supports 99 languages with automatic language detection.
- **Size**: ~1.5 GB download, runs on CPU (int8) or GPU (float16)

### TranslateGemma 12B (Text Translation)

- **What**: Google's purpose-built translation model based on Gemma 3, released January 2026
- **Why**: Specifically fine-tuned for translation rather than being a general-purpose LLM. The 12B model outperforms the base Gemma 3 27B on translation benchmarks (MetricX 3.60 vs 4.04 on WMT24++) while using half the parameters. Supports 55 languages with near-DeepL quality. Runs on consumer GPUs with 8+ GB VRAM using int4 quantization.
- **Size**: ~8 GB download (quantized), served via Ollama
- **Alternatives**: `translategemma:4b` for less VRAM (~4 GB), `gemma3:12b` as a general-purpose fallback

### Gemma 3 12B (Text Processing)

- **What**: Google's instruction-tuned general-purpose LLM
- **Why**: Excels at instruction following, making it ideal for summarization, text cleanup, and following custom prompts. Supports 140+ languages with a 128K context window, so it can process very long transcriptions. The QAT (Quantization-Aware Trained) variant preserves near-BF16 quality at reduced memory.
- **Size**: ~8 GB download (quantized), served via Ollama
- **Use cases**: Summarize transcriptions, remove contradictions/repetitions, follow custom text processing instructions

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- At least 16 GB RAM recommended (models load into memory)
- GPU optional but recommended for faster inference

### 1. Clone and configure

```bash
git clone <repo-url> && cd voice-to-text
cp .env.example .env
# Edit .env if needed (model sizes, ports, etc.)
```

### 2. Start all services

```bash
docker compose up --build
```

On first start, Ollama will automatically download TranslateGemma 12B (~8 GB) and Gemma 3 12B (~8 GB). The Whisper model (~1.5 GB) also downloads on first run. This initial download may take 10-30 minutes depending on your connection.

### 3. Access the UIs

- **Voice-to-Text**: http://localhost:1234
- **Text Translator**: http://localhost:1235

### GPU Support

```bash
docker compose --profile gpu up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Configuration

All settings via environment variables (see `.env.example`):

### Whisper (Voice-to-Text)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU), `float32` |
| `WHISPER_BEAM_SIZE` | `5` | Higher = more accurate, slower |
| `MAX_UPLOAD_MB` | `500` | Max upload size in MB |

### Translation & Text Processing

| Variable | Default | Description |
|---|---|---|
| `TRANSLATION_MODEL` | `translategemma:12b` | Ollama model for translation |
| `PROCESSING_MODEL` | `gemma3:12b` | Ollama model for summarization/optimization |
| `MAX_TEXT_LENGTH` | `10000` | Max input text length (characters) |

### Ports

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | Whisper API |
| `TEXT_API_PORT` | `5000` | Text API |
| `OLLAMA_PORT` | `11434` | Ollama LLM runtime |
| `WEB_PORT` | `1234` | Voice-to-Text frontend |
| `TRANSLATOR_PORT` | `1235` | Text Translator frontend |

## Using Lower-Resource Models

If you have limited hardware (< 12 GB VRAM or CPU-only), you can use smaller models:

```env
# Smaller translation model (~3.3 GB, 4 GB VRAM)
TRANSLATION_MODEL=translategemma:4b

# Smaller processing model (~2 GB, 4 GB VRAM)
PROCESSING_MODEL=gemma3:4b
```

## API Reference

### Whisper API (`:8000`)

- `GET /health` — Service status, loaded model info
- `POST /transcribe` — Upload audio (`multipart/form-data`), returns transcript + segments + language

### Text API (`:5000`)

- `GET /health` — Service status, Ollama connection, model readiness
- `GET /languages` — List of supported translation languages
- `POST /detect` — Detect language of input text
- `POST /translate` — Translate text between languages
- `POST /process` — Summarize, optimize, or apply custom prompt to text

## Voice-to-Text Post-Processing

After transcribing audio, the Voice-to-Text UI offers four post-processing actions:

1. **Summarize** — Get a concise summary of the transcription
2. **Translate** — Translate the transcribed text to another language
3. **Optimize** — Clean up the text by removing contradictions, repetitions, filler words, and false starts
4. **Custom Prompt** — Provide your own instructions for how to transform the text

## License

MIT -- Provided by Dr. Sven-Erik Willrich / mail@svenwillrich.de
