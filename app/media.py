"""ffmpeg / OpenCV helpers: normalise uploads, read frames, write renders."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import config


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    has_audio: bool


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr[-800:]}")
    return proc.stdout


def probe(path: Path) -> VideoInfo:
    out = json.loads(
        _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-count_frames", "-of", "json", str(path)])
    )
    video = next(s for s in out["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in out["streams"])
    num, den = video.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den or 1)
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    duration = float(out["format"].get("duration", frames / fps if fps else 0))
    return VideoInfo(int(video["width"]), int(video["height"]), fps, frames, duration, has_audio)


def normalize(src: Path, dst: Path) -> VideoInfo:
    """Constant-frame-rate H.264 MP4 with faststart (what SAM 3.1 wants), even dims,
    capped length and resolution so renders stay quick. Frame indices of this file are
    the frame indices SAM reports, so everything downstream reads from it."""
    info = probe(src)
    fps = min(round(info.fps) or 30, 30)
    scale = min(1.0, config.MAX_LONG_SIDE / max(info.width, info.height))
    w = int(info.width * scale) // 2 * 2
    h = int(info.height * scale) // 2 * 2
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(config.MAX_SECONDS),
        "-vf", f"scale={w}:{h}:flags=lanczos,fps={fps}", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    ]
    cmd += ["-c:a", "aac", "-b:a", "160k", "-ac", "2"] if info.has_audio else ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    _run(cmd)
    return probe(dst)


def extract_wav(src: Path, dst: Path, rate: int = 24000) -> bool:
    """Mono 16-bit PCM WAV for Muse Voice Transcribe. Returns False if there is no audio."""
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(rate),
              "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)])
    except RuntimeError:
        return False
    return dst.exists() and dst.stat().st_size > 1000


def read_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class VideoWriter:
    """Pipe BGR frames into ffmpeg, muxing the source audio track."""

    def __init__(self, dst: Path, info: VideoInfo, audio_src: Path | None):
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}",
            "-i", "-",
        ]
        if audio_src is not None and info.has_audio:
            cmd += ["-i", str(audio_src), "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode() if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {stderr[-800:]}")


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
