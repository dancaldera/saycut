"""Thin client for the Meta Model API endpoints SayCut uses.

* Files API          - upload the clip once, reuse the file id.
* Responses API      - Muse Spark (planning, video + image understanding) and SAM 3.1.
* Images API         - Muse Image (backgrounds, Reels covers).
* ASR transcribe API - Muse Voice Transcribe (voice commands, captions).
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from . import config


class MetaAPIError(RuntimeError):
    pass


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    if not config.API_KEY:
        raise MetaAPIError("MODEL_API_KEY is not set. Add it to .env (see README).")
    headers = {"Authorization": f"Bearer {config.API_KEY}"}
    if extra:
        headers.update(extra)
    return headers


def _request(method: str, path: str, *, retries: int = 4, **kwargs: Any) -> httpx.Response:
    """HTTP call with exponential backoff on 429 / 5xx, per the rate-limit docs."""
    delay = 0.5
    timeout = kwargs.pop("timeout", 300)
    for attempt in range(retries + 1):
        try:
            response = httpx.request(method, f"{config.API_BASE}{path}", timeout=timeout, **kwargs)
        except httpx.TransportError as exc:
            if attempt == retries:
                raise MetaAPIError(f"{path}: network error {exc}") from exc
            time.sleep(delay + random.random() * 0.3)
            delay *= 2
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            retry_after = response.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay + random.random() * 0.3)
            delay *= 2
            continue
        if response.status_code >= 400:
            raise MetaAPIError(f"{path} -> HTTP {response.status_code}: {response.text[:500]}")
        return response
    raise MetaAPIError(f"{path}: retries exhausted")


# --------------------------------------------------------------------------- files

def upload_file(path: Path, mime: str = "video/mp4") -> str:
    with path.open("rb") as fh:
        response = _request(
            "POST",
            "/files",
            headers=_headers(),
            data={"purpose": "user_data"},
            files={"file": (path.name, fh, mime)},
        )
    return response.json()["id"]


# --------------------------------------------------------------------------- responses

def _output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def respond(
    content: list[dict[str, Any]],
    *,
    model: str = config.PLANNER_MODEL,
    instructions: str | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "result",
    effort: str | None = "low",
) -> str:
    """Single-turn Responses API call. Returns the output text (JSON when a schema is given)."""
    body: dict[str, Any] = {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": content}],
        "store": False,
    }
    if instructions:
        body["instructions"] = instructions
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}}
    if effort:
        body["reasoning"] = {"effort": effort}
    response = _request("POST", "/responses", headers=_headers({"Content-Type": "application/json"}), json=body)
    return _output_text(response.json())


def respond_json(content: list[dict[str, Any]], schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    text = respond(content, schema=schema, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetaAPIError(f"Model returned invalid JSON: {text[:300]}") from exc


def image_part(png_or_jpg: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    return {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(png_or_jpg).decode()}"}


# --------------------------------------------------------------------------- SAM 3.1

def sam_segment_video(
    file_id: str,
    concept: str,
    on_frame: Callable[[int], None] | None = None,
) -> str:
    """Stream SAM 3.1 concept segmentation for a whole clip. Returns the raw output text.

    The raw text is cached by callers and parsed with meta_sam_parser, so re-renders
    never pay for segmentation twice.
    """
    body = {
        "model": config.SAM_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_video", "file_id": file_id},
                ],
            }
        ],
        "stream": True,
        "metadata": {"mask_encoding": "one_bit"},
    }
    delay = 1.0
    for attempt in range(4):
        text = ""
        frames = 0
        try:
            with httpx.stream(
                "POST",
                f"{config.API_BASE}/responses",
                headers=_headers({"Accept": "text/event-stream", "Content-Type": "application/json"}),
                json=body,
                timeout=httpx.Timeout(600, connect=30),
            ) as stream:
                if stream.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    stream.read()
                    time.sleep(delay)
                    delay *= 2
                    continue
                if stream.status_code >= 400:
                    raise MetaAPIError(f"SAM 3.1 -> HTTP {stream.status_code}: {stream.read()[:500]!r}")
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    kind = event.get("type")
                    if kind == "response.output_text.delta":
                        text += event.get("delta", "")
                        frames += 1
                        if on_frame:
                            on_frame(frames)
                    elif kind == "response.output_text.done" and not text:
                        text = event.get("text", "")
                    elif kind in ("response.failed", "error"):
                        raise MetaAPIError(f"SAM 3.1 failed: {json.dumps(event)[:500]}")
            return text
        except httpx.TransportError as exc:
            if attempt == 3:
                raise MetaAPIError(f"SAM 3.1 network error: {exc}") from exc
            time.sleep(delay)
            delay *= 2
    raise MetaAPIError("SAM 3.1: retries exhausted")


# --------------------------------------------------------------------------- Muse Image

def _image_bytes(payload: dict[str, Any]) -> bytes:
    data = payload.get("data") or []
    if not data:
        raise MetaAPIError(f"Muse Image returned no image: {json.dumps(payload)[:300]}")
    item = data[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return httpx.get(item["url"], timeout=120).content
    raise MetaAPIError("Muse Image response had neither b64_json nor url")


def generate_image(prompt: str, size: str = "auto") -> bytes:
    body = {"model": config.IMAGE_MODEL, "prompt": prompt, "n": 1, "output_format": "png", "size": size}
    response = _request(
        "POST", "/images/generations", headers=_headers({"Content-Type": "application/json"}), json=body, timeout=300
    )
    return _image_bytes(response.json())


def edit_image(prompt: str, image: bytes, mime: str = "image/png", size: str = "auto") -> bytes:
    body = {
        "model": config.IMAGE_MODEL,
        "prompt": prompt,
        "images": [{"image_url": f"data:{mime};base64,{base64.b64encode(image).decode()}"}],
        "n": 1,
        "output_format": "png",
        "size": size,
    }
    response = _request(
        "POST", "/images/edits", headers=_headers({"Content-Type": "application/json"}), json=body, timeout=300
    )
    return _image_bytes(response.json())


# --------------------------------------------------------------------------- Voice Transcribe

def transcribe(wav_path: Path, mode: str = "PUSH_TO_TALK") -> dict[str, Any]:
    """Transcribe a mono 16-bit PCM WAV (16 or 24 kHz). mode: PUSH_TO_TALK | ENDPOINTING | DIARIZATION."""
    settings = {"model": config.TRANSCRIBE_MODEL, "audioEncoding": "WAV", "mode": mode}
    with wav_path.open("rb") as audio:
        response = _request(
            "POST",
            "/asr/transcribe",
            headers=_headers({"Accept": "application/json"}),
            files={
                "request": (None, json.dumps(settings), "application/json"),
                "audio": (wav_path.name, audio, "audio/wav"),
            },
            timeout=300,
        )
    return response.json()
