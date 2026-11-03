#!/usr/bin/env bash
# Start SayCut on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (brew install ffmpeg)"; exit 1; }
if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
  else python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; fi
fi
if command -v node >/dev/null && [ ! -d decoder/node_modules ]; then (cd decoder && npm install --silent); fi
[ -f .env ] || { echo "Create .env with MODEL_API_KEY=... (see .env.example)"; exit 1; }
exec .venv/bin/uvicorn app.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
