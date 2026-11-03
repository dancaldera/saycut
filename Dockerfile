# SayCut container image (linux/amd64) for Cloudflare Containers or any Docker host.

# Stage 1: fast SAM mask decoder deps (Meta's @meta-sam/parser needs Node >= 20.17).
FROM node:22-bookworm-slim AS decoder
WORKDIR /decoder
COPY decoder/package.json decoder/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

# Stage 2: app.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SAYCUT_DATA_DIR=/data SAYCUT_PUBLIC=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 fonts-dejavu-core fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*
COPY --from=decoder /usr/local/bin/node /usr/local/bin/node
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=decoder /decoder/node_modules decoder/node_modules
COPY decoder/decode.mjs decoder/package.json decoder/
COPY app app
COPY web web
RUN mkdir -p /data
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
