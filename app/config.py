"""Runtime configuration: paths, model IDs, and limits."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SAYCUT_DATA_DIR", ROOT / "data"))
PROJECTS_DIR = DATA_DIR / "projects"
WEB_DIR = ROOT / "web"


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

API_BASE = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
API_KEY = os.environ.get("MODEL_API_KEY", "")

# Meta models used by the pipeline.
PLANNER_MODEL = os.environ.get("SAYCUT_PLANNER_MODEL", "muse-spark-1.3")
VISION_MODEL = os.environ.get("SAYCUT_VISION_MODEL", "muse-spark-1.3")
SAM_MODEL = "sam-3.1"
IMAGE_MODEL = "muse-image-1.0"
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"

# Input normalisation. Keeps rendering fast and stays well inside SAM's limits
# (15,000 frames, MP4 + H.264).
MAX_SECONDS = float(os.environ.get("SAYCUT_MAX_SECONDS", "60"))
MAX_LONG_SIDE = int(os.environ.get("SAYCUT_MAX_LONG_SIDE", "1280"))

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
