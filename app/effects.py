"""Per-frame visual effects driven by SAM 3.1 masks.

Every effect takes a BGR frame and a full-frame uint8 mask (0/255) and returns a new
frame. Masks are feathered so edges look natural instead of cut out.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def hex_to_bgr(value: str | None, default: tuple[int, int, int] = (255, 229, 0)) -> tuple[int, int, int]:
    if not value:
        return default
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    try:
        r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except (ValueError, IndexError):
        return default
    return (b, g, r)


def _odd(n: float) -> int:
    n = max(1, int(n))
    return n if n % 2 else n + 1


def feather(mask: np.ndarray, radius: float, grow: int = 0) -> np.ndarray:
    """Soft alpha in [0,1] (float32, HxWx1) from a hard 0/255 mask."""
    m = mask
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
    if radius > 0:
        m = cv2.GaussianBlur(m, (_odd(radius * 2), _odd(radius * 2)), 0)
    return (m.astype(np.float32) / 255.0)[..., None]


def blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (base.astype(np.float32) * (1 - alpha) + top.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def _scale(frame: np.ndarray) -> float:
    return max(frame.shape[:2]) / 1280.0


# --------------------------------------------------------------------------- object effects

def blur(frame: np.ndarray, mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    if not mask.any():
        return frame
    s = _scale(frame)
    k = _odd((45 + 90 * intensity) * s)
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    blurred = cv2.GaussianBlur(blurred, (k, k), 0)
    return blend(frame, blurred, feather(mask, 6 * s, grow=int(8 * s)))


def pixelate(frame: np.ndarray, mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    if not mask.any():
        return frame
    h, w = frame.shape[:2]
    block = max(6, int((10 + 30 * intensity) * _scale(frame)))
    small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return blend(frame, pixel, feather(mask, 2, grow=int(6 * _scale(frame))))


def glow(frame: np.ndarray, mask: np.ndarray, color: str | None = None, intensity: float = 0.8) -> np.ndarray:
    """Neon outline + halo + subtle subject lift."""
    if not mask.any():
        return frame
    s = _scale(frame)
    bgr = np.array(hex_to_bgr(color, (255, 229, 0)), dtype=np.float32)
    out = frame.astype(np.float32)

    # Lift the subject a little so it pops.
    subject = feather(mask, 3 * s)
    out = out * (1 - subject * 0.18 * intensity) + np.minimum(255, out * 1.25 + 12) * (subject * 0.18 * intensity)

    # Halo: blurred ring outside the mask.
    ring_w = max(2, int(5 * s))
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_w * 2 + 1,) * 2))
    ring = cv2.subtract(dil, mask)
    halo = cv2.GaussianBlur(ring, (_odd(40 * s), _odd(40 * s)), 0).astype(np.float32) / 255.0
    halo = np.clip(halo * (1.6 + 1.4 * intensity), 0, 1)[..., None]
    out = out + halo * bgr * (0.9 + 0.6 * intensity)

    # Crisp neon core line.
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(3 * s),) * 2))
    edge = cv2.GaussianBlur(edge, (_odd(3 * s), _odd(3 * s)), 0).astype(np.float32)[..., None] / 255.0
    core = bgr * 0.4 + 255 * 0.6
    out = out * (1 - edge) + core * edge
    return out.clip(0, 255).astype(np.uint8)


def color_pop(frame: np.ndarray, mask: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    background = blend(frame, gray, np.full((1, 1, 1), min(1.0, 0.6 + 0.4 * intensity), np.float32))
    return blend(background, frame, feather(mask, 4 * _scale(frame)))


def spotlight(frame: np.ndarray, mask: np.ndarray, intensity: float = 0.7) -> np.ndarray:
    s = _scale(frame)
    dark = (frame.astype(np.float32) * (0.85 - 0.6 * intensity)).astype(np.uint8)
    soft = feather(mask, 30 * s, grow=int(25 * s))
    return blend(dark, frame, soft)


def background_blur(frame: np.ndarray, mask: np.ndarray, intensity: float = 0.7) -> np.ndarray:
    k = _odd((15 + 45 * intensity) * _scale(frame))
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return blend(blurred, frame, feather(mask, 3 * _scale(frame)))


def background_replace(frame: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = feather(mask, 2.5 * _scale(frame))
    # Light colour match: nudge the subject toward the new scene's average tone.
    subject = frame.astype(np.float32)
    shift = (background.reshape(-1, 3).mean(0) - frame.reshape(-1, 3).mean(0)) * 0.12
    subject = np.clip(subject + shift, 0, 255)
    return blend(background, subject, alpha)


def remove(frame: np.ndarray, mask: np.ndarray, plate: np.ndarray) -> np.ndarray:
    s = _scale(frame)
    return blend(frame, plate, feather(mask, 5 * s, grow=int(10 * s)))


def cover_with_sprite(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], sprite_rgba: np.ndarray) -> np.ndarray:
    """Paste an RGBA sprite (e.g. an emoji) over each box, sized to cover it."""
    out = frame.copy()
    H, W = frame.shape[:2]
    for left, top, right, bottom in boxes:
        bw, bh = right - left, bottom - top
        size = int(max(bw, bh) * 1.25)
        if size < 8:
            continue
        cx, cy = (left + right) // 2, (top + bottom) // 2
        sprite = cv2.resize(sprite_rgba, (size, size), interpolation=cv2.INTER_AREA)
        x0, y0 = cx - size // 2, cy - size // 2
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(W, x0 + size), min(H, y0 + size)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = sprite[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
        alpha = crop[..., 3:4].astype(np.float32) / 255.0
        out[y1:y2, x1:x2] = blend(out[y1:y2, x1:x2], crop[..., :3], alpha)
    return out


# --------------------------------------------------------------------------- clean plate for "remove"

def build_clean_plate(frames: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Temporal median of each pixel over the frames where it is NOT covered by the
    object (works great for fixed/tripod shots), then inpaint whatever was never seen."""
    stack = np.stack(frames).astype(np.float32)  # N,H,W,3
    hidden = np.stack([cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0 for m in masks])  # N,H,W
    H = stack.shape[1]
    plate = np.zeros(stack.shape[1:], np.float32)
    never_seen = np.zeros(stack.shape[1:3], bool)
    for y in range(0, H, 48):
        chunk = stack[:, y : y + 48].copy()
        chunk[hidden[:, y : y + 48]] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(chunk, axis=0)
        never_seen[y : y + 48] = np.isnan(med[..., 0])
        plate[y : y + 48] = np.nan_to_num(med)
    plate = plate.clip(0, 255).astype(np.uint8)
    if never_seen.any():
        plate = cv2.inpaint(plate, never_seen.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return plate


# --------------------------------------------------------------------------- text + emoji

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
EMOJI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
]


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@lru_cache(maxsize=32)
def emoji_sprite(emoji: str) -> np.ndarray | None:
    """Render an emoji to a 160x160 RGBA (BGR order) numpy array."""
    for path in EMOJI_FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        for size in (160, 137, 109, 96, 64):
            try:
                font = ImageFont.truetype(path, size)
                canvas = Image.new("RGBA", (size + 40, size + 40), (0, 0, 0, 0))
                ImageDraw.Draw(canvas).text((20, 20), emoji, font=font, embedded_color=True)
                bbox = canvas.getbbox()
                if not bbox:
                    continue
                img = np.array(canvas.crop(bbox))
                side = max(img.shape[:2])
                square = np.zeros((side, side, 4), np.uint8)
                oy, ox = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
                square[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
                return cv2.cvtColor(square, cv2.COLOR_RGBA2BGRA)
            except OSError:
                continue
    return None


def caption_overlay(text: str, width: int, height: int, highlight: str | None = None) -> np.ndarray:
    """Reels-style caption: big bold white text, black stroke, lower third. RGBA (BGR order)."""
    size = max(24, int(max(width, height) * 0.048))
    font = _font(size)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    max_w = int(width * 0.86)

    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    lines = lines[-2:]

    stroke = max(2, size // 9)
    line_h = int(size * 1.2)
    y = int(height * 0.72) - line_h * len(lines) // 2
    accent = tuple(reversed(hex_to_bgr(highlight, (0, 229, 255)))) + (255,)
    for i, ln in enumerate(lines):
        tw = draw.textlength(ln, font=font)
        x = (width - tw) / 2
        fill = accent if (i == len(lines) - 1 and highlight) else (255, 255, 255, 255)
        draw.text((x, y), ln, font=font, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def overlay_rgba(frame: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    if not alpha.any():
        return frame
    return blend(frame, rgba[..., :3], alpha)
