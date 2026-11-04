"""Project state + the edit pipeline.

command -> Muse Spark plan -> SAM 3.1 tracks per target -> object selection
(heuristic or Muse Spark vision) -> assets (Muse Image backgrounds, clean plates,
Voice Transcribe captions) -> frame-accurate render.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import agent, config, effects, media, meta_api, restyle
from .tracks import Tracks, load_tracks

Progress = Callable[[str, float | None], None]

# Order in which effects are composited, regardless of the order they were asked for.
STAGE = {
    "restyle": -1, "remove": 0, "background_replace": 1, "background_blur": 2, "color_pop": 3, "spotlight": 4,
    "blur": 5, "pixelate": 6, "glow": 7, "emoji": 8, "captions": 9,
}
OBJECT_EFFECTS = set(STAGE) - {"captions", "restyle"}

_project_locks: dict[str, threading.Lock] = {}


_concept_locks: dict[tuple[str, str], threading.Lock] = {}
_locks_guard = threading.Lock()


def project_lock(pid: str) -> threading.Lock:
    with _locks_guard:
        return _project_locks.setdefault(pid, threading.Lock())


def concept_lock(pid: str, concept: str) -> threading.Lock:
    with _locks_guard:
        return _concept_locks.setdefault((pid, concept), threading.Lock())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "x"


class Project:
    def __init__(self, pid: str):
        if not re.fullmatch(r"[a-f0-9]{12}", pid):
            raise FileNotFoundError(pid)
        self.id = pid
        self.dir = config.PROJECTS_DIR / pid
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(pid)
        self.meta: dict[str, Any] = json.loads((self.dir / "meta.json").read_text())
        self._tracks: dict[tuple[str, str], Tracks] = {}

    # ------------------------------------------------------------------ storage
    @classmethod
    def create(cls, upload_path: Path, filename: str) -> "Project":
        pid = uuid.uuid4().hex[:12]
        pdir = config.PROJECTS_DIR / pid
        (pdir / "sam").mkdir(parents=True)
        (pdir / "assets").mkdir()
        (pdir / "renders").mkdir()
        info = media.normalize(upload_path, pdir / "source.mp4")
        meta = {
            "id": pid,
            "filename": filename,
            "created": time.time(),
            "info": asdict(info),
            "file_id": None,
            "analysis": None,
            "ops": [],
            "history": [],
            "render": None,
            "render_version": 0,
            "cover": None,
            "concepts": {},
        }
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2))
        frame = media.read_frame(pdir / "source.mp4", min(info.frames - 1, int(info.fps))) if info.frames else None
        if frame is not None:
            small = cv2.resize(frame, (160, max(1, int(160 * info.height / info.width))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(pdir / "assets" / "poster.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return cls(pid)

    def save(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.dir / "meta.json")

    @property
    def source(self) -> Path:
        return self.dir / "source.mp4"

    @property
    def info(self) -> media.VideoInfo:
        return media.VideoInfo(**self.meta["info"])

    def public(self) -> dict[str, Any]:
        m = self.meta
        return {
            "id": self.id,
            "filename": m["filename"],
            "info": m["info"],
            "analysis": m["analysis"],
            "ops": m["ops"],
            "history": m["history"][-30:],
            "source_url": f"/media/{self.id}/source.mp4",
            "render_url": f"/media/{self.id}/renders/{m['render']}" if m.get("render") else None,
            "cover_url": f"/media/{self.id}/assets/{m['cover']}" if m.get("cover") else None,
            "poster_url": f"/media/{self.id}/assets/poster.jpg" if (self.dir / "assets" / "poster.jpg").exists() else None,
            "ready": bool(m.get("file_id")),
        }

    # ------------------------------------------------------------------ Meta services
    def ensure_uploaded(self, progress: Progress) -> str:
        if not self.meta.get("file_id"):
            progress("Uploading clip to Meta Model API", None)
            self.meta["file_id"] = meta_api.upload_file(self.source)
            self.save()
        return self.meta["file_id"]

    def ensure_analysis(self, progress: Progress) -> dict[str, Any]:
        if not self.meta.get("analysis"):
            file_id = self.ensure_uploaded(progress)
            progress("Muse Spark is watching your clip", None)
            self.meta["analysis"] = agent.analyze_video(file_id)
            self.save()
        return self.meta["analysis"]

    def base(self) -> str:
        """The video object effects are drawn on: "" = the source, else the active restyle video."""
        op = next((o for o in self.meta["ops"] if o["effect"] == "restyle"), None)
        return op["video"] if op else ""

    def _base_file_id(self, base: str, progress: Progress) -> str:
        if not base:
            return self.ensure_uploaded(progress)
        files = self.meta.setdefault("base_files", {})
        if base not in files:
            progress("Uploading the restyled video so SAM 3.1 can track it", None)
            files[base] = meta_api.upload_file(self.dir / "assets" / base)
            self.save()
        return files[base]

    def tracks(self, concept: str, progress: Progress, base: str = "") -> Tracks:
        concept = concept.strip().lower()
        if (base, concept) in self._tracks:
            return self._tracks[(base, concept)]
        prefix = f"{Path(base).stem}__" if base else ""
        raw = self.dir / "sam" / f"{prefix}{_slug(concept)}.txt"
        info = self.info
        lock = concept_lock(self.id, prefix + concept)
        if not raw.exists() and lock.locked():
            progress(f"SAM 3.1 is already tracking “{concept}” in the background", None)
        with lock:  # one SAM call per concept, even with background prefetch running
            if not raw.exists():
                file_id = self._base_file_id(base, progress)
                total = max(1, info.frames)

                def on_frame(n: int) -> None:
                    if n % 10 == 0:
                        progress(f"SAM 3.1 tracking “{concept}” · frame {n}/{total}", min(0.99, n / total))

                progress(f"SAM 3.1 tracking “{concept}”", 0.0)
                text = meta_api.sam_segment_video(file_id, concept, on_frame)
                tmp = raw.with_suffix(".tmp")
                tmp.write_text(text)
                tmp.replace(raw)
        tracks = load_tracks(raw, concept, info.width, info.height)
        self._tracks[(base, concept)] = tracks
        return tracks

    def prefetch(self, concepts: list[str]) -> None:
        """Speculatively segment likely targets in the background so edits feel instant."""

        def run(concept: str) -> None:
            try:
                Project(self.id).tracks(concept, lambda *_: None)
            except Exception as exc:  # noqa: BLE001 - best effort
                print(f"prefetch {concept!r} failed: {exc}")

        for concept in concepts:
            threading.Thread(target=run, args=(concept,), daemon=True).start()

    def likely_targets(self) -> list[str]:
        subjects = [s.strip().lower() for s in (self.meta.get("analysis") or {}).get("subjects", [])]
        picks: list[str] = []
        for concept in ["person", "face", *subjects[:2]]:
            if concept and concept not in picks and len(concept.split()) <= 3:
                picks.append(concept)
        return picks[:4]

    # ------------------------------------------------------------------ commands
    def context(self) -> dict[str, Any]:
        a = self.meta.get("analysis") or {}
        return {
            "summary": a.get("summary", ""),
            "subjects": a.get("subjects", []),
            "duration_sec": round(self.info.duration, 2),
            "current_edits": [op["label"] for op in self.meta["ops"]],
        }

    def run_command(self, command: str, progress: Progress) -> dict[str, Any]:
        self.ensure_analysis(progress)
        progress("Muse Spark is planning your edit", None)
        plan = agent.plan_edit(command, self.context())
        trace: list[str] = [f"Muse Spark plan: {self._plan_summary(plan)}"]
        added: list[dict[str, Any]] = []
        changed = False

        for op in plan.get("operations", []):
            effect = op["effect"]
            if effect == "none":
                continue
            if effect == "undo":
                if self.meta["ops"]:
                    self.meta["ops"].pop()
                    changed = True
                continue
            if effect == "reset":
                changed = bool(self.meta["ops"]) or changed
                self.meta["ops"] = []
                continue
            resolved = self._resolve(op, progress, trace)
            if resolved is None:
                continue
            if effect == "restyle":  # one look at a time: a new restyle replaces the old one
                self.meta["ops"] = [o for o in self.meta["ops"] if o["effect"] != "restyle"]
            self.meta["ops"].append(resolved)
            added.append(resolved)
            changed = True

        if changed:
            self.render(progress)
        reply = plan.get("reply") or "Done!"
        if not added and not changed:
            reply = trace[-1] if len(trace) > 1 else (plan.get("reply") or "I couldn't find anything to change.")
        self.meta["history"].append({"command": command, "reply": reply, "trace": trace, "at": time.time()})
        self.save()
        return {"reply": reply, "trace": trace, "project": self.public()}

    @staticmethod
    def _plan_summary(plan: dict[str, Any]) -> str:
        parts = []
        for op in plan.get("operations", []):
            bits = [op["effect"]]
            if op.get("target"):
                bits.append(f"“{op['target']}”")
            if op.get("selection") == "main_subject":
                bits.append("(main subject)")
            if op.get("selection") == "described":
                bits.append(f"({op.get('description')})")
            if op.get("invert"):
                bits.append("inverted")
            parts.append(" ".join(bits))
        return "; ".join(parts) or "no changes"

    def _resolve(self, op: dict[str, Any], progress: Progress, trace: list[str]) -> dict[str, Any] | None:
        effect = op["effect"]
        resolved: dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "effect": effect,
            "target": op.get("target", ""),
            "ids": None,
            "color": op.get("color") or "",
            "intensity": op.get("intensity", 0.8),
            "emoji": op.get("emoji") or "",
            "start": op.get("start_sec"),
            "end": op.get("end_sec"),
        }

        if effect == "restyle":
            style = (op.get("style_prompt") or "").strip() or "anime illustration with clean cel shading"
            resolved.update(target="", start=None, end=None, style=style)
            resolved["video"], keys = self._restyle(style, progress)
            resolved["label"] = f"Restyle · {style[:60]}"
            trace.append(f"Muse Image restyled {keys} keyframes: “{style}”")
            trace.append("Optical flow carried each keyframe along the clip's real motion")
            return resolved

        if effect == "captions":
            captions = self._captions(progress)
            if not captions:
                trace.append("Voice Transcribe found no speech in this clip.")
                return None
            resolved["captions"] = captions
            resolved["color"] = resolved["color"] or "#00E5FF"
            resolved["label"] = "Captions"
            trace.append(f"Muse Voice Transcribe: {len(captions)} caption lines")
            return resolved

        target = resolved["target"] or ("person" if effect in ("background_replace", "background_blur") else "")
        if not target:
            trace.append(f"No target for {effect}.")
            return None
        resolved["target"] = target
        base = self.base()
        resolved["base"] = base
        resolved["plan"] = {k: v for k, v in op.items()}
        tracks = self.tracks(target, progress, base)
        if not tracks.objects:
            trace.append(f"SAM 3.1 found no “{target}” in the clip.")
            return None
        trace.append(f"SAM 3.1 tracked {len(tracks.objects)} × “{target}” across {len(tracks.frames)} frames")

        # Which tracked objects does the user mean?
        ids: set[str] = set(tracks.object_ids)
        selection = op.get("selection", "all")
        if selection == "main_subject" and len(tracks.objects) == 1:
            pass  # the only tracked object is the subject
        elif selection == "main_subject":
            # Ask Muse Spark (vision) who the video is about; screen-time heuristic as fallback.
            want = "the main human subject the video is about (the creator / person on camera, not an animal or bystander)"
            pick = self._pick(tracks, want, progress)
            ids = set(pick["ids"][:1])
            if ids:
                trace.append(f"Muse Spark: main subject is #{next(iter(ids))}: {pick['reason']}")
            else:
                main = tracks.main_subject()
                ids = {main} if main else set()
                trace.append(f"Main subject = #{main} (most screen time × size)")
        elif selection == "described" and op.get("description"):
            pick = self._pick(tracks, op["description"], progress)
            ids = set(pick["ids"])
            trace.append(f"Muse Spark picked {', '.join('#' + i for i in sorted(ids)) or 'nothing'}: {pick['reason']}")
        if op.get("invert"):
            ids = set(tracks.object_ids) - ids
        if not ids:
            trace.append("No matching objects after selection.")
            return None
        resolved["ids"] = sorted(ids) if ids != set(tracks.object_ids) else None

        if effect == "background_replace":
            prompt = op.get("background_prompt") or "a beautiful sunset beach, photorealistic"
            resolved["background"] = self._background(prompt, progress)
            resolved["background_prompt"] = prompt
            trace.append(f"Muse Image painted: “{prompt}”")
        elif effect == "remove":
            resolved["plate"] = self._clean_plate(tracks, resolved["ids"], progress)
        elif effect == "emoji" and not resolved["emoji"]:
            resolved["emoji"] = "😎"

        resolved["label"] = self._label(resolved, op)
        return resolved

    def _pick(self, tracks: Tracks, description: str, progress: Progress) -> dict[str, Any]:
        progress(f"Muse Spark is looking for {description.split('(')[0].strip()}", None)
        base = self.base()
        prefix = f"{Path(base).stem}__" if base else ""
        sheet_path = self.dir / "assets" / f"sheet_{prefix}{_slug(tracks.concept)}.jpg"
        if sheet_path.exists():
            sheet = sheet_path.read_bytes()
        else:
            frames = self._frames_at({s.best_frame for s in tracks.objects.values()}, base)
            sheet = agent.contact_sheet(frames, tracks)
            sheet_path.write_bytes(sheet)
        return agent.pick_objects(description, tracks.concept, sheet, tracks.object_ids)

    @staticmethod
    def _label(resolved: dict[str, Any], op: dict[str, Any]) -> str:
        names = {
            "blur": "Blur", "pixelate": "Pixelate", "glow": "Glow", "color_pop": "Color pop",
            "spotlight": "Spotlight", "background_blur": "Portrait blur", "background_replace": "New background",
            "remove": "Remove", "emoji": f"Emoji {resolved.get('emoji', '')}",
        }
        who = resolved["target"]
        if op.get("selection") == "main_subject":
            who = f"{who} (all except main)" if op.get("invert") else f"{who} (main)"
        elif op.get("selection") == "described":
            who = f"{who}: {'not ' if op.get('invert') else ''}{op.get('description')}"
        label = f"{names.get(resolved['effect'], resolved['effect'])} · {who}"
        if resolved.get("start") is not None or resolved.get("end") is not None:
            label += f" · {resolved.get('start') or 0}s–{resolved.get('end') or 'end'}s"
        return label

    # ------------------------------------------------------------------ assets
    def _frames_at(self, indices: set[int], base: str = "") -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        video = self.dir / "assets" / base if base else self.source
        for i, frame in enumerate(media.read_frames(video)):
            if i in indices:
                out[i] = frame
            if len(out) == len(indices):
                break
        return out

    def _background(self, prompt: str, progress: Progress) -> str:
        info = self.info
        name = f"bg_{hashlib.sha1(prompt.encode()).hexdigest()[:10]}.png"
        path = self.dir / "assets" / name
        if not path.exists():
            progress("Muse Image is painting the new background", None)
            full_prompt = (
                f"{prompt}. Empty scene with no people or animals, photorealistic, natural light, "
                f"eye-level camera, suitable as a video background."
            )
            path.write_bytes(meta_api.generate_image(full_prompt, size=f"{info.width}x{info.height}"))
        return name

    def _restyle(self, style: str, progress: Progress) -> tuple[str, int]:
        info = self.info
        key = hashlib.sha1(style.encode()).hexdigest()[:10]
        name = f"restyle_{key}.mp4"
        path = self.dir / "assets" / name
        keys_dir = self.dir / "assets" / f"restyle_{key}"
        if path.exists():
            return name, len(list(keys_dir.glob("k*.png")))
        restyle.check_budget()
        keyframes = restyle.stylize_keyframes(self.source, info, style, keys_dir, progress)
        if len(keyframes) < 2 and info.frames > 1:
            raise RuntimeError("Muse Image couldn't restyle this clip. Try a different style.")
        restyle.render_restyled(self.source, info, keyframes, path, progress)
        return name, len(keyframes)

    def _load_background(self, name: str) -> np.ndarray:
        info = self.info
        img = cv2.imread(str(self.dir / "assets" / name))
        h, w = img.shape[:2]
        scale = max(info.width / w, info.height / h)
        img = cv2.resize(img, (int(w * scale + 0.5), int(h * scale + 0.5)), interpolation=cv2.INTER_AREA)
        y0, x0 = (img.shape[0] - info.height) // 2, (img.shape[1] - info.width) // 2
        return img[y0 : y0 + info.height, x0 : x0 + info.width]

    def _clean_plate(self, tracks: Tracks, ids: list[str] | None, progress: Progress) -> str:
        base = self.base()
        key = hashlib.sha1(f"{base}|{tracks.concept}|{ids}".encode()).hexdigest()[:10]
        name = f"plate_{key}.png"
        path = self.dir / "assets" / name
        if not path.exists():
            progress("Reconstructing the background behind the object", None)
            info = self.info
            step = max(1, info.frames // 48)
            frames, masks = [], []
            sel = set(ids) if ids else None
            for i, frame in enumerate(media.read_frames(self.dir / "assets" / base if base else self.source)):
                if i % step == 0:
                    frames.append(frame)
                    masks.append(tracks.mask(i, sel))
            cv2.imwrite(str(path), effects.build_clean_plate(frames, masks))
        return name

    def _captions(self, progress: Progress) -> list[dict[str, Any]]:
        cache = self.dir / "assets" / "captions.json"
        if cache.exists():
            return json.loads(cache.read_text())
        wav = self.dir / "assets" / "audio.wav"
        if not self.info.has_audio or not media.extract_wav(self.source, wav):
            return []
        progress("Muse Voice Transcribe is listening", None)
        result = meta_api.transcribe(wav, mode="ENDPOINTING")
        turns = result.get("turns") or []
        if not turns and result.get("transcript"):
            turns = [{"startMs": 0, "endMs": int(self.info.duration * 1000), "transcript": result["transcript"]}]
        captions = []
        for turn in turns:
            words = (turn.get("transcript") or "").split()
            if not words:
                continue
            start, end = turn["startMs"] / 1000, turn["endMs"] / 1000
            # Turn-level timestamps only: split into ~4-word chunks spread across the turn.
            chunks = [words[i : i + 4] for i in range(0, len(words), 4)]
            per_word = (end - start) / max(1, len(words))
            t = start
            for chunk in chunks:
                dur = per_word * len(chunk)
                captions.append({"start": round(t, 3), "end": round(t + dur, 3), "text": " ".join(chunk)})
                t += dur
        cache.write_text(json.dumps(captions))
        return captions

    # ------------------------------------------------------------------ render
    def _rebase(self, progress: Progress) -> None:
        """Object effects must follow what is on screen: re-resolve them when the restyle changes."""
        base = self.base()
        updated = []
        for op in self.meta["ops"]:
            if op["effect"] in OBJECT_EFFECTS and op.get("base", "") != base and op.get("plan"):
                fresh = self._resolve(op["plan"], progress, [])
                if fresh is None:
                    continue  # the object can't be found in the new look: drop the effect
                fresh["id"] = op["id"]
                op = fresh
            updated.append(op)
        self.meta["ops"] = updated

    def render(self, progress: Progress) -> None:
        info = self.info
        self._rebase(progress)
        ops = sorted(self.meta["ops"], key=lambda o: STAGE.get(o["effect"], 50))
        if not ops:
            self.meta["render"] = None
            self.save()
            return

        tracks = {o["target"]: self.tracks(o["target"], progress, o.get("base", "")) for o in ops if o["effect"] in OBJECT_EFFECTS}
        backgrounds = {o["id"]: self._load_background(o["background"]) for o in ops if o.get("background")}
        plates = {o["id"]: cv2.imread(str(self.dir / "assets" / o["plate"])) for o in ops if o.get("plate")}
        sprites = {o["id"]: effects.emoji_sprite(o["emoji"]) for o in ops if o["effect"] == "emoji"}
        caption_cache: dict[tuple[str, int], np.ndarray] = {}

        base = next((o for o in ops if o["effect"] == "restyle"), None)
        base_video = self.dir / "assets" / base["video"] if base else self.source

        version = self.meta.get("render_version", 0) + 1
        name = f"v{version}.mp4"
        out_path = self.dir / "renders" / name
        writer = media.VideoWriter(out_path, info, self.source if info.has_audio else None)
        total = max(1, info.frames)
        try:
            for index, frame in enumerate(media.read_frames(base_video)):
                t = index / info.fps
                for op in ops:
                    if op["effect"] == "restyle":
                        continue
                    if op.get("start") is not None and t < op["start"]:
                        continue
                    if op.get("end") is not None and t > op["end"]:
                        continue
                    frame = self._apply(op, frame, index, t, tracks, backgrounds, plates, sprites, caption_cache)
                writer.write(frame)
                if index % 15 == 0:
                    progress(f"Rendering · frame {index}/{total}", index / total)
        finally:
            writer.close()

        previous = self.meta.get("render")
        self.meta["render"] = name
        self.meta["render_version"] = version
        self.save()
        # Keep disk tidy: only the latest two renders.
        if previous and previous != name:
            for old in sorted((self.dir / "renders").glob("v*.mp4"))[:-2]:
                old.unlink(missing_ok=True)

    def _apply(self, op, frame, index, t, tracks, backgrounds, plates, sprites, caption_cache):
        effect = op["effect"]
        if effect == "captions":
            for i, cap in enumerate(op.get("captions", [])):
                if cap["start"] <= t <= cap["end"] + 0.05:
                    key = (op["id"], i)
                    if key not in caption_cache:
                        caption_cache[key] = effects.caption_overlay(cap["text"], frame.shape[1], frame.shape[0], op.get("color"))
                    return effects.overlay_rgba(frame, caption_cache[key])
            return frame

        tr = tracks[op["target"]]
        ids = set(op["ids"]) if op.get("ids") else None
        if effect == "emoji":
            sprite = sprites.get(op["id"])
            boxes = [box for oid, box, _ in tr.frames.get(index, ()) if ids is None or oid in ids]
            if sprite is None:
                return effects.pixelate(frame, tr.mask(index, ids), 1.0)
            return effects.cover_with_sprite(frame, boxes, sprite)

        mask = tr.mask(index, ids)
        intensity = op.get("intensity", 0.8)
        if effect == "blur":
            return effects.blur(frame, mask, intensity)
        if effect == "pixelate":
            return effects.pixelate(frame, mask, intensity)
        if effect == "glow":
            return effects.glow(frame, mask, op.get("color"), intensity)
        if effect == "color_pop":
            return effects.color_pop(frame, mask, intensity)
        if effect == "spotlight":
            return effects.spotlight(frame, mask, intensity)
        if effect == "background_blur":
            return effects.background_blur(frame, mask, intensity)
        if effect == "background_replace":
            return effects.background_replace(frame, mask, backgrounds[op["id"]])
        if effect == "remove":
            return effects.remove(frame, mask, plates[op["id"]])
        return frame

    # ------------------------------------------------------------------ undo / cover
    def undo(self, progress: Progress) -> dict[str, Any]:
        if self.meta["ops"]:
            removed = self.meta["ops"].pop()
            self.render(progress)
            self.meta["history"].append({"command": "Undo", "reply": f"Removed: {removed['label']}", "trace": [], "at": time.time()})
            self.save()
        return {"reply": "Undone.", "trace": [], "project": self.public()}

    def remove_op(self, op_id: str, progress: Progress) -> dict[str, Any]:
        before = len(self.meta["ops"])
        self.meta["ops"] = [o for o in self.meta["ops"] if o["id"] != op_id]
        if len(self.meta["ops"]) != before:
            self.render(progress)
        return {"reply": "Removed.", "trace": [], "project": self.public()}

    def make_cover(self, frame_time: float | None, progress: Progress) -> dict[str, Any]:
        analysis = self.ensure_analysis(progress)
        video = self.dir / "renders" / self.meta["render"] if self.meta.get("render") else self.source
        info = self.info
        index = int((frame_time if frame_time is not None else info.duration * 0.4) * info.fps)
        frame = media.read_frame(video, min(max(0, index), max(0, info.frames - 1)))
        if frame is None:
            frame = media.read_frame(self.source, 0)
        ok, png = cv2.imencode(".png", frame)
        title = analysis.get("title") or "Watch this"
        prompt = (
            f"Turn this video frame into a scroll-stopping Instagram Reels cover. Keep the people and scene recognizable. "
            f"Add the bold title text \"{title}\" in a clean, modern, high-contrast style with plenty of breathing room. "
            f"Boost color and lighting slightly. Vertical 9:16 composition."
        )
        progress("Muse Image is designing your cover", None)
        image = meta_api.edit_image(prompt, png.tobytes(), "image/png", size="1080x1920")
        name = f"cover_{uuid.uuid4().hex[:6]}.png"
        (self.dir / "assets" / name).write_bytes(image)
        self.meta["cover"] = name
        self.meta["history"].append({"command": "Make a cover", "reply": f"Cover ready: “{title}”", "trace": ["Muse Image edited the frame into a cover"], "at": time.time()})
        self.save()
        return {"reply": f"Cover ready: “{title}”", "trace": [], "project": self.public()}


def delete_project(pid: str) -> None:
    project = Project(pid)
    shutil.rmtree(project.dir, ignore_errors=True)
