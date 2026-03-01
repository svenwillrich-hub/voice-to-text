# ─────────────────────────────────────────────────────────────────────────────
# Web Frontend — nginx serving static files + reverse proxy to API
# ─────────────────────────────────────────────────────────────────────────────

FROM nginx:1.27-alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Custom nginx config with API reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static files
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80
