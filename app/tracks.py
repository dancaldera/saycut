"""Turn SAM 3.1 video output into per-frame, per-object masks.

SAM streams one line per frame: ``<Nf>id<|box..|><|mask..|>,id...``. We keep the raw
text on disk (cheap to cache, exact to replay), parse it with Meta's
``meta_sam_parser``, and decode each object's mask into a numpy array positioned at its
box. Object ids are stable across frames, which is what lets a user say
"everyone except me" and have it stay correct for the whole clip.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster, parse_responses_stream, video_segmentation_format

DECODER_DIR = Path(__file__).resolve().parent.parent / "decoder"


@dataclass
class ObjectStats:
    object_id: str
    frames: int = 0
    total_area: float = 0.0
    best_frame: int = -1
    best_area: float = 0.0
    best_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    first_frame: int = 10**9
    last_frame: int = -1

    @property
    def mean_area(self) -> float:
        return self.total_area / self.frames if self.frames else 0.0


@dataclass
class Tracks:
    concept: str
    width: int
    height: int
    # frame -> list of (object_id, (left, top, right, bottom), mask uint8 0/1 of box size)
    frames: dict[int, list[tuple[str, tuple[int, int, int, int], np.ndarray]]] = field(default_factory=dict)
    objects: dict[str, ObjectStats] = field(default_factory=dict)

    @property
    def object_ids(self) -> list[str]:
        return sorted(self.objects, key=lambda k: int(k) if k.isdigit() else k)

    def main_subject(self) -> str | None:
        """Most prominent object: screen presence weighted by size."""
        if not self.objects:
            return None
        return max(self.objects.values(), key=lambda s: s.frames * s.mean_area).object_id

    def mask(self, frame_index: int, ids: set[str] | None = None) -> np.ndarray:
        """Full-frame uint8 mask (0/255) for the selected object ids (None = all)."""
        full = np.zeros((self.height, self.width), dtype=np.uint8)
        for object_id, (left, top, right, bottom), raster in self.frames.get(frame_index, ()):
            if ids is not None and object_id not in ids:
                continue
            left, top = max(0, left), max(0, top)
            right, bottom = min(self.width, right), min(self.height, bottom)
            if right <= left or bottom <= top:
                continue
            region = raster
            if region.shape != (bottom - top, right - left):
                region = cv2.resize(region, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
            np.maximum(full[top:bottom, left:right], region * 255, out=full[top:bottom, left:right])
        return full


async def _parse(text: str):
    async def events():
        yield {"type": "response.output_text.done", "item_id": "saycut", "output_index": 0, "content_index": 0, "text": text}
        yield {"type": "response.completed"}

    return await parse_responses_stream(events(), video_segmentation_format()).final_result()


def _add(tracks: Tracks, frame_index: int, object_id: str, box: tuple[int, int, int, int], raster: np.ndarray) -> None:
    tracks.frames.setdefault(frame_index, []).append((object_id, box, raster))
    area = float(raster.sum())
    stats = tracks.objects.setdefault(object_id, ObjectStats(object_id))
    stats.frames += 1
    stats.total_area += area
    stats.first_frame = min(stats.first_frame, frame_index)
    stats.last_frame = max(stats.last_frame, frame_index)
    if area > stats.best_area:
        stats.best_area, stats.best_frame, stats.best_box = area, frame_index, box


def _node_decode(raw_path: Path, tracks: Tracks) -> bool:
    """Fast path: Meta's TypeScript parser in Node (~60x faster than the Python port)."""
    node = shutil.which("node")
    if not node or not (DECODER_DIR / "node_modules" / "@meta-sam" / "parser").exists():
        return False
    bin_path = raw_path.with_suffix(".bin")
    if not bin_path.exists() or bin_path.stat().st_mtime < raw_path.stat().st_mtime:
        proc = subprocess.run([node, str(DECODER_DIR / "decode.mjs"), str(raw_path), str(bin_path)],
                              capture_output=True, text=True, cwd=DECODER_DIR)
        if proc.returncode != 0:
            return False
    data = bin_path.read_bytes()
    if data[:4] != b"SCM1":
        return False
    (count,) = struct.unpack_from("<I", data, 4)
    offset = 8
    for _ in range(count):
        frame, obj, left, top, right, bottom, h, w = struct.unpack_from("<IIiiiiII", data, offset)
        offset += 32
        nbytes = (h * w + 7) // 8
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=nbytes, offset=offset))[: h * w]
        offset += nbytes
        _add(tracks, frame, str(obj), (left, top, right, bottom), bits.reshape(h, w))
    return True


def parse_tracks(text: str, concept: str, width: int, height: int) -> Tracks:
    """Slow but dependency-free path using meta_sam_parser (Python)."""
    tracks = Tracks(concept=concept, width=width, height=height)
    if not text.strip():
        return tracks
    result = asyncio.run(_parse(text))

    # Keep the highest revision per (frame, object).
    latest = {}
    for record in result.records:
        if record.kind != "mask" or record.frame is None:
            continue
        key = (record.frame.frame_index, record.object_id)
        if key not in latest or record.revision > latest[key].revision:
            latest[key] = record

    for (frame_index, object_id), record in latest.items():
        b = record.bounds
        box = (int(round(b.left)), int(round(b.top)), int(round(b.right)), int(round(b.bottom)))
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        _add(tracks, frame_index, object_id, box, raster.reshape(record.mask.height, record.mask.width))
    return tracks


def load_tracks(raw_path: Path, concept: str, width: int, height: int) -> Tracks:
    tracks = Tracks(concept=concept, width=width, height=height)
    if raw_path.read_text().strip() == "" or _node_decode(raw_path, tracks):
        return tracks
    return parse_tracks(raw_path.read_text(), concept, width, height)
