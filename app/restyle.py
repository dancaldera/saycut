"""Whole-video restyle: Muse Image keyframes + optical-flow propagation.

There is no video-to-video model in the Model API, so SayCut builds one:

1. Pick keyframes (~every 0.6 s, capped) from the clip.
2. Muse Image transforms the first keyframe from the creator's prompt ("anime girl on a
   beach", "claymation", "cyberpunk city"). That frame becomes the style anchor.
3. Every other keyframe is transformed in parallel with the anchor passed as a second
   reference image, so characters, palette and setting stay consistent across the clip.
4. Between keyframes, each stylised keyframe is carried along the *real* motion of the
   source video (dense DIS optical flow, coordinate maps composed frame to frame so the
   image is only resampled once), forward from the previous keyframe and backward from
   the next, and the two are cross-faded. The result moves exactly like the original
   clip, frame for frame, in the new style.
"""

from __future__ import annotations

import base64
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config, media, meta_api

Progress = Callable[[str, float | None], None]

KEY_SPACING_SEC = 0.6
MAX_KEYFRAMES = 28
FLOW_LONG_SIDE = 480
PARALLEL = 8


# Cost guard for the public demo: each restyle is up to MAX_KEYFRAMES Muse Image calls.
DAILY_LIMIT = int(os.environ.get("SAYCUT_RESTYLES_PER_DAY", "120"))
_budget = {"day": "", "count": 0}
_budget_lock = threading.Lock()


def check_budget() -> None:
    with _budget_lock:
        today = time.strftime("%Y-%m-%d")
        if _budget["day"] != today:
            _budget.update(day=today, count=0)
        if _budget["count"] >= DAILY_LIMIT:
            raise RuntimeError("Today's restyle budget for the public demo is used up. Try again tomorrow, or run SayCut locally.")
        _budget["count"] += 1


def keyframe_indices(frames: int, fps: float) -> list[int]:
    if frames <= 1:
        return [0]
    count = min(MAX_KEYFRAMES, max(2, math.ceil(frames / (fps * KEY_SPACING_SEC)) + 1))
    idx = sorted({round(i * (frames - 1) / (count - 1)) for i in range(count)})
    return idx


def _png(frame: np.ndarray) -> bytes:
    return cv2.imencode(".png", frame)[1].tobytes()


def _data_url(png: bytes) -> dict:
    return {"image_url": f"data:image/png;base64,{base64.b64encode(png).decode()}"}


def _edit(prompt: str, images: list[bytes], size: str) -> bytes:
    body = {
        "model": config.IMAGE_MODEL,
        "prompt": prompt,
        "images": [_data_url(i) for i in images],
        "n": 1,
        "output_format": "png",
        "size": size,
    }
    response = meta_api._request(
        "POST", "/images/edits", headers=meta_api._headers({"Content-Type": "application/json"}), json=body, timeout=300
    )
    return meta_api._image_bytes(response.json())


def anchor_prompt(style: str) -> str:
    return (
        f"Transform this video frame: {style}. "
        "Keep the exact same composition, camera framing, subject pose, position and scale so it lines up with "
        "the original frame. Keep each person recognizable (hair, clothing, accessories), restyled to fit. "
        "Fill the whole frame. No text, no captions, no borders."
    )


def follow_prompt(style: str) -> str:
    return (
        f"Transform image 1, a frame from a video: {style}. "
        "Image 2 is another frame of the SAME video that has already been transformed: the result must look like "
        "the next frame of that same animation, with the same art style, the same character designs, the same "
        "background and setting (replace image 1's original background the same way image 2 did), the same colors "
        "and lighting. Take only the pose, framing, camera angle and positions from image 1; do NOT copy image 2's "
        "pose. Fill the whole frame. No text, no borders."
    )


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Cover-resize an image to exactly w x h."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, int(iw * scale + 0.5)), max(h, int(ih * scale + 0.5))), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def stylize_keyframes(
    source: Path, info: media.VideoInfo, style: str, out_dir: Path, progress: Progress
) -> dict[int, Path]:
    """Return {frame_index: stylised png path}. Cached per style in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = keyframe_indices(info.frames, info.fps)
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    for i, frame in enumerate(media.read_frames(source)):
        if i in wanted:
            frames[i] = frame
        if len(frames) == len(wanted):
            break
    indices = [i for i in indices if i in frames]
    size = f"{info.width}x{info.height}"
    paths = {i: out_dir / f"k{i:05d}.png" for i in indices}

    anchor_idx = indices[0]
    if not paths[anchor_idx].exists():
        progress(f"Muse Image is restyling keyframe 1/{len(indices)}", 0.02)
        paths[anchor_idx].write_bytes(_edit(anchor_prompt(style), [_png(frames[anchor_idx])], size))
    anchor = paths[anchor_idx].read_bytes()

    todo = [i for i in indices if not paths[i].exists()]
    done = len(indices) - len(todo)
    failed: list[int] = []

    def work(i: int) -> int:
        last: Exception | None = None
        for _ in range(2):
            try:
                paths[i].write_bytes(_edit(follow_prompt(style), [_png(frames[i]), anchor], size))
                return i
            except Exception as exc:  # noqa: BLE001 - one retry, then skip the keyframe
                last = exc
        raise RuntimeError(f"keyframe {i}: {last}")

    with ThreadPoolExecutor(PARALLEL) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"restyle: {exc}")
                failed.append(futures[fut])
            done += 1
            progress(f"Muse Image is restyling keyframe {done}/{len(indices)}", 0.02 + 0.78 * done / len(indices))

    return {i: p for i, p in paths.items() if i not in failed and p.exists()}


class _Flow:
    def __init__(self, w: int, h: int):
        scale = min(1.0, FLOW_LONG_SIDE / max(w, h))
        self.sw, self.sh = max(16, int(w * scale)), max(16, int(h * scale))
        self.w, self.h = w, h
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        gx, gy = np.meshgrid(np.arange(self.sw, dtype=np.float32), np.arange(self.sh, dtype=np.float32))
        self.grid = np.dstack([gx, gy])

    def gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def step(self, coords: np.ndarray, g_to: np.ndarray, g_from: np.ndarray) -> np.ndarray:
        """coords: per pixel of `g_from` frame, position in the keyframe. Returns the same for `g_to`."""
        flow = self.dis.calc(g_to, g_from, None)  # g_to(x) ~ g_from(x + flow(x))
        pos = self.grid + flow
        return cv2.remap(coords, pos[..., 0], pos[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def warp(self, key: np.ndarray, coords: np.ndarray) -> np.ndarray:
        sx, sy = self.w / self.sw, self.h / self.sh
        full = cv2.resize(coords, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mx = full[..., 0] * sx + (sx - 1) / 2
        my = full[..., 1] * sy + (sy - 1) / 2
        return cv2.remap(key, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_restyled(
    source: Path, info: media.VideoInfo, keyframes: dict[int, Path], dst: Path, progress: Progress
) -> None:
    """Write the restyled video (no audio) to dst."""
    w, h = info.width, info.height
    keys = sorted(keyframes)
    images = {i: _fit(cv2.imread(str(keyframes[i])), w, h) for i in keys}
    flow = _Flow(w, h)
    total = max(1, info.frames)
    tmp = dst.with_suffix(".tmp.mp4")
    writer = media.VideoWriter(tmp, info, None)
    frames_iter = media.read_frames(source)
    index = 0

    def take(n: int) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            f = next(frames_iter, None)
            if f is None:
                break
            out.append(f)
        return out

    try:
        bounds = [(None, keys[0])] + list(zip(keys, keys[1:])) + [(keys[-1], None)]
        carry: np.ndarray | None = None  # last frame read; it is keyframe `a` of the next segment
        for a, b in bounds:
            end = (total - 1) if b is None else b
            fresh = take(end - index + 1) if end >= index else []
            if a is not None:
                if carry is None:
                    break
                window, offset = [carry, *fresh], a
            else:
                window, offset = fresh, index
            if not fresh:
                continue
            grays = [flow.gray(f) for f in window]
            n = len(window)
            fwd: list[np.ndarray | None] = [None] * n
            bwd: list[np.ndarray | None] = [None] * n
            if a is not None:
                coords = flow.grid.copy()
                fwd[0] = coords
                for j in range(1, n):
                    coords = flow.step(coords, grays[j], grays[j - 1])
                    fwd[j] = coords
            if b is not None and 0 <= b - offset < n:
                ib = b - offset
                coords = flow.grid.copy()
                bwd[ib] = coords
                for j in range(ib - 1, -1, -1):
                    coords = flow.step(coords, grays[j], grays[j + 1])
                    bwd[j] = coords
            for j in range(n):
                absolute = offset + j
                if a is not None and j == 0:
                    continue  # keyframe `a` was written by the previous segment
                if fwd[j] is not None and bwd[j] is not None:
                    # Cross-fade only through the middle of the interval: each keyframe
                    # owns the frames near it, which keeps ghosting short.
                    t = min(1.0, max(0.0, ((absolute - a) / (b - a) - 0.3) / 0.4))
                    t = t * t * (3 - 2 * t)
                    out = cv2.addWeighted(flow.warp(images[a], fwd[j]), 1 - t, flow.warp(images[b], bwd[j]), t, 0)
                elif fwd[j] is not None:
                    out = flow.warp(images[a], fwd[j])
                else:
                    out = flow.warp(images[b], bwd[j])
                writer.write(out)
                index += 1
                if index % 10 == 0:
                    progress(f"Animating the new style · frame {index}/{total}", 0.8 + 0.2 * index / total)
            carry = window[-1]
        for _ in frames_iter:  # frames the probe missed: hold the last keyframe
            writer.write(images[keys[-1]])
    finally:
        writer.close()
    tmp.replace(dst)
