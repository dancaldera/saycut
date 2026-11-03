"""SayCut web server: upload a clip, talk to it, watch it change."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, media, meta_api
from .pipeline import Project, project_lock

app = FastAPI(title="SayCut")

# --------------------------------------------------------------------------- public-deployment guardrails
# The hosted demo spends one Model API key for everyone, so cap usage per visitor and
# globally. Disabled for local runs unless SAYCUT_PUBLIC=1.

PUBLIC = os.environ.get("SAYCUT_PUBLIC") == "1"
LIMITS = {
    "upload": (int(os.environ.get("SAYCUT_UPLOADS_PER_HOUR", "6")), 3600),
    "edit": (int(os.environ.get("SAYCUT_EDITS_PER_HOUR", "60")), 3600),
    "voice": (int(os.environ.get("SAYCUT_VOICE_PER_HOUR", "80")), 3600),
    "global_upload": (int(os.environ.get("SAYCUT_UPLOADS_PER_DAY", "200")), 86400),
}
MAX_UPLOAD_BYTES = int(os.environ.get("SAYCUT_MAX_UPLOAD_MB", "150")) * 1024 * 1024
PROJECT_TTL = float(os.environ.get("SAYCUT_PROJECT_TTL_HOURS", "24")) * 3600
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def rate_limit(kind: str, request: Request) -> None:
    if not PUBLIC:
        return
    now = time.time()
    keys = [(kind, client_ip(request))]
    if kind == "upload":
        keys.append(("global_upload", "*"))
    with _hits_lock:
        for key in keys:
            limit, window = LIMITS[key[0]]
            q = _hits[key]
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= limit:
                wait = int(q[0] + window - now) // 60 + 1
                who = "The public demo" if key[1] == "*" else "You"
                raise HTTPException(429, f"{who} hit the {key[0].replace('_', ' ')} limit. Try again in ~{wait} min, or run SayCut locally (it's open source).")
        for key in keys:
            _hits[key].append(now)


def _janitor() -> None:
    """Delete old projects and finished jobs so the container disk stays small."""
    while True:
        cutoff = time.time() - PROJECT_TTL
        for pdir in config.PROJECTS_DIR.iterdir():
            try:
                if pdir.is_dir() and pdir.stat().st_mtime < cutoff:
                    shutil.rmtree(pdir, ignore_errors=True)
            except OSError:
                pass
        for job_id, job in list(JOBS.items()):
            if job["status"] != "running" and job["started"] < time.time() - 3600:
                JOBS.pop(job_id, None)
        time.sleep(900)


if PUBLIC:
    threading.Thread(target=_janitor, daemon=True).start()

# --------------------------------------------------------------------------- jobs

JOBS: dict[str, dict[str, Any]] = {}


def start_job(pid: str | None, work: Callable[[Callable[[str, float | None], None]], Any]) -> str:
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "project": pid, "status": "running", "stage": "Starting", "progress": None,
           "steps": [], "result": None, "error": None, "started": time.time()}
    JOBS[job_id] = job

    def progress(stage: str, frac: float | None) -> None:
        if not job["steps"] or job["steps"][-1]["stage"].split(" · ")[0] != stage.split(" · ")[0]:
            job["steps"].append({"stage": stage, "at": round(time.time() - job["started"], 1)})
        else:
            job["steps"][-1]["stage"] = stage
        job["stage"], job["progress"] = stage, frac

    def run() -> None:
        lock = project_lock(pid) if pid else threading.Lock()
        try:
            with lock:
                job["result"] = work(progress)
            job["status"] = "done"
        except meta_api.MetaAPIError as exc:
            job["status"], job["error"] = "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI
            traceback.print_exc()
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
        job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=run, daemon=True).start()
    return job_id


def load(pid: str) -> Project:
    try:
        return Project(pid)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")


# --------------------------------------------------------------------------- API

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "api_key": bool(config.API_KEY), "public": PUBLIC, "models": {
        "planner": config.PLANNER_MODEL, "vision": config.VISION_MODEL, "segmentation": config.SAM_MODEL,
        "image": config.IMAGE_MODEL, "voice": config.TRANSCRIBE_MODEL}}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/projects")
async def create_project(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    rate_limit("upload", request)
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"Clip is too large (max {MAX_UPLOAD_BYTES // 2**20} MB).")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    try:
        project = Project.create(tmp_path, file.filename or "clip.mp4")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read that video: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    def prepare(progress):
        project.ensure_analysis(progress)
        project.prefetch(project.likely_targets())
        return {"project": project.public()}

    job_id = start_job(project.id, prepare)
    return {"project": project.public(), "job": job_id}


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict[str, Any]:
    return load(pid).public()


class Command(BaseModel):
    text: str


@app.post("/api/projects/{pid}/command")
def command(pid: str, body: Command, request: Request) -> dict[str, Any]:
    text = body.text.strip()[:500]
    if not text:
        raise HTTPException(400, "Say or type an edit first")
    rate_limit("edit", request)
    load(pid)
    return {"job": start_job(pid, lambda progress: Project(pid).run_command(text, progress))}


@app.post("/api/projects/{pid}/undo")
def undo(pid: str) -> dict[str, Any]:
    load(pid)
    return {"job": start_job(pid, lambda progress: Project(pid).undo(progress))}


@app.delete("/api/projects/{pid}/ops/{op_id}")
def remove_op(pid: str, op_id: str) -> dict[str, Any]:
    load(pid)
    return {"job": start_job(pid, lambda progress: Project(pid).remove_op(op_id, progress))}


class CoverRequest(BaseModel):
    time: float | None = None


@app.post("/api/projects/{pid}/cover")
def cover(pid: str, body: CoverRequest, request: Request) -> dict[str, Any]:
    load(pid)
    rate_limit("edit", request)
    return {"job": start_job(pid, lambda progress: Project(pid).make_cover(body.time, progress))}


@app.post("/api/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)) -> dict[str, Any]:
    """Voice command: browser recording (webm/ogg/mp4) -> 24 kHz mono WAV -> Muse Voice Transcribe."""
    rate_limit("voice", request)
    data = await audio.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Voice clip too long")
    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / f"voice{Path(audio.filename or 'v.webm').suffix or '.webm'}"
        raw.write_bytes(data)
        wav = Path(tmpdir) / "voice.wav"
        if not media.extract_wav(raw, wav):
            raise HTTPException(400, "Couldn't read that recording")
        try:
            result = meta_api.transcribe(wav, mode="PUSH_TO_TALK")
        except meta_api.MetaAPIError as exc:
            raise HTTPException(502, str(exc))
    return {"text": (result.get("transcript") or "").strip()}


# --------------------------------------------------------------------------- static

@app.get("/media/{pid}/{path:path}")
def media_file(pid: str, path: str) -> FileResponse:
    project = load(pid)
    target = (project.dir / path).resolve()
    if project.dir.resolve() not in target.parents or not target.is_file() or target.name == "meta.json":
        raise HTTPException(404)
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
