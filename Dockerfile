# ─────────────────────────────────────────────────────────────────────────────
# Voice-to-Text — All-in-One Docker Image
# Combines FastAPI + faster-whisper backend with nginx frontend in one image.
# Ideal for publishing to GitHub Container Registry (ghcr.io) or Docker Hub.
#
# Build:  docker build -t voice-to-text .
# Run:    docker run -p 80:80 -v whisper-cache:/root/.cache/huggingface voice-to-text
# GPU:    docker run --gpus all -e WHISPER_DEVICE=cuda -e WHISPER_COMPUTE_TYPE=float16 \
#           -p 80:80 -v whisper-cache:/root/.cache/huggingface voice-to-text
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies: ffmpeg (audio decoding), nginx (frontend), supervisor (process mgr)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg nginx supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY whisper-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend
COPY whisper-api/main.py .

# Frontend — static files served by nginx
COPY frontend/index.html /var/www/html/index.html

# nginx config — serves frontend + reverse-proxies /api/ to uvicorn
RUN rm -f /etc/nginx/sites-enabled/default
COPY <<'NGINX' /etc/nginx/sites-enabled/voice-to-text.conf
server {
    listen 80;
    server_name _;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 500m;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
NGINX

# Supervisor config — runs both nginx and uvicorn
COPY <<'SUPERVISOR' /etc/supervisor/conf.d/voice-to-text.conf
[supervisord]
nodaemon=true
logfile=/dev/stdout
logfile_maxbytes=0

[program:uvicorn]
command=uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:nginx]
command=nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
SUPERVISOR

EXPOSE 80

ENV WHISPER_MODEL=large-v3-turbo \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_BEAM_SIZE=5 \
    MAX_UPLOAD_MB=500

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
