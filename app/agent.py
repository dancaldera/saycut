"""Muse Spark as the editor's brain.

1. analyze_video  - watches the uploaded clip (video understanding) and proposes edits.
2. plan_edit      - turns a natural-language/voice command into structured edit ops.
3. pick_objects   - looks at a contact sheet of SAM-tracked objects and picks the ones
                    a description refers to ("the girl in the green shirt").
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from . import config, meta_api
from .media import encode_jpeg
from .tracks import Tracks

EFFECTS = [
    "blur", "pixelate", "glow", "color_pop", "spotlight", "background_blur",
    "background_replace", "remove", "emoji", "captions", "restyle", "undo", "reset", "none",
]

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "title", "subjects", "suggestions"],
    "properties": {
        "summary": {"type": "string", "description": "One sentence describing the clip."},
        "title": {"type": "string", "description": "Scroll-stopping Reels hook, max 6 words."},
        "subjects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short noun phrases for segmentable things visible in the clip, e.g. 'person', 'dog', 'face'.",
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4 short edit commands a creator would say for THIS clip.",
        },
    },
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "operations"],
    "properties": {
        "reply": {"type": "string"},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "effect", "target", "selection", "description", "invert", "color",
                    "intensity", "emoji", "background_prompt", "style_prompt", "start_sec", "end_sec",
                ],
                "properties": {
                    "effect": {"type": "string", "enum": EFFECTS},
                    "target": {"type": "string"},
                    "selection": {"type": "string", "enum": ["all", "main_subject", "described"]},
                    "description": {"type": "string"},
                    "invert": {"type": "boolean"},
                    "color": {"type": "string"},
                    "intensity": {"type": "number"},
                    "emoji": {"type": "string"},
                    "background_prompt": {"type": "string"},
                    "style_prompt": {"type": "string"},
                    "start_sec": {"type": ["number", "null"]},
                    "end_sec": {"type": ["number", "null"]},
                },
            },
        },
    },
}

PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ids", "reason"],
    "properties": {"ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}},
}

PLANNER_INSTRUCTIONS = """You are SayCut, an AI video editor for Instagram Reels. Convert the creator's command into edit operations.

Effects (pick the closest; never invent new ones):
- blur / pixelate: hide the target (faces, license plates, screens, logos).
- glow: neon outline + halo on the target. Use `color` (hex) if a colour is mentioned, else a vivid default like #00E5FF.
- color_pop: keep the target in colour, turn everything else black & white.
- spotlight: darken everything except the target.
- background_blur: portrait-mode blur behind the target (target is usually "person").
- background_replace: put the target (usually "person") in a new scene; write a vivid `background_prompt` describing ONLY the empty scene (no people), photorealistic, matching the clip's camera angle.
- remove: erase the target from the video.
- emoji: cover the target with an emoji (set `emoji` to one emoji character). Great for hiding faces playfully.
- captions: burn in subtitles from the speech in the video (target "").
- restyle: transform the WHOLE video into a new look or world while keeping its motion (target ""). Use it for art styles
  (anime, cartoon, Pixar-style 3D, claymation, LEGO, comic book, watercolor, oil painting, Van Gogh, pixel art, pencil sketch,
  vaporwave), whole-scene changes ("make it snow", "underwater", "cyberpunk city at night", "zombie apocalypse",
  "on the moon") and turning people into characters ("make me an anime girl on a beach", "turn us into robots",
  "make me a superhero"). Write a vivid `style_prompt`: the art style + the setting + any character changes, in one
  sentence, describing what the WHOLE frame should become (e.g. "anime illustration with clean cel shading; she is an
  anime girl on a sunny tropical beach with palm trees"). Prefer restyle over background_replace whenever the person
  should change too, or the request is an art style / "crazy" transformation. Only one restyle is active at a time.
- undo: remove the last edit. reset: remove all edits. none: only for requests that are not video edits at all.
Never refuse a creative transformation: if no other effect fits, use restyle.

Target rules (the target is sent to Segment Anything 3.1):
- `target` is ONE short concrete noun phrase: "face", "person", "dog", "ball", "car", "laptop screen". No verbs, no "and", no lists, no negation, no positions.
- For several different object types, emit one operation per type.
- "me", "myself", "the creator", "the main person" => target "person" (or "face" for face effects), selection "main_subject".
- "everyone/every X except me" => selection "main_subject", invert true.
- A specific instance described by appearance or position ("the girl in green", "the man on the left") => selection "described" with that `description`. Add invert true for "everyone except ...".
- Otherwise selection "all", invert false. `description` is "" unless selection is "described".

Other fields: `intensity` 0..1 (default 0.8). `start_sec`/`end_sec` only if the user gives a time range, else null. Unused strings are "".
`reply`: one short, upbeat sentence telling the creator what you're doing (no markdown)."""


def analyze_video(file_id: str) -> dict[str, Any]:
    prompt = (
        "You are helping a creator edit this clip into an Instagram Reel. "
        "Return a one-sentence summary, a short hook title, the segmentable subjects you see, and 4 edit commands "
        "that would make this specific clip more engaging. Commands must use these tools: blur/pixelate/emoji over "
        "faces or objects, neon glow on a subject, colour pop, spotlight, portrait background blur, replace the "
        "background with a new scene, remove an object, add captions, or restyle the whole video (anime, claymation, cyberpunk, a new world). Make the LAST suggestion a bold whole-video transformation. Keep each command under 9 words, "
        "phrased like a person talking, e.g. 'Make the dog glow pink'."
    )
    return meta_api.respond_json(
        [{"type": "input_text", "text": prompt}, {"type": "input_file", "file_id": file_id}],
        ANALYSIS_SCHEMA,
        model=config.VISION_MODEL,
        schema_name="clip_analysis",
        effort="low",
    )


def plan_edit(command: str, context: dict[str, Any]) -> dict[str, Any]:
    context_text = json.dumps(context, ensure_ascii=False)
    plan = meta_api.respond_json(
        [{"type": "input_text", "text": f"Clip context: {context_text}\n\nCreator command: {command}"}],
        PLAN_SCHEMA,
        model=config.PLANNER_MODEL,
        instructions=PLANNER_INSTRUCTIONS,
        schema_name="edit_plan",
        effort="low",
    )
    for op in plan.get("operations", []):
        op["target"] = op.get("target", "").strip().lower().rstrip(".")
        op["intensity"] = float(min(1.0, max(0.0, op.get("intensity") or 0.8)))
    return plan


def contact_sheet(source_frames: dict[int, np.ndarray], tracks: Tracks, max_objects: int = 16) -> bytes:
    """Grid of crops, one per tracked object, each labelled with its object id."""
    tiles = []
    for object_id in tracks.object_ids[:max_objects]:
        stats = tracks.objects[object_id]
        frame = source_frames.get(stats.best_frame)
        if frame is None:
            continue
        left, top, right, bottom = stats.best_box
        pad_x, pad_y = int((right - left) * 0.6) + 10, int((bottom - top) * 0.6) + 10
        h, w = frame.shape[:2]
        crop = frame[max(0, top - pad_y) : min(h, bottom + pad_y), max(0, left - pad_x) : min(w, right + pad_x)].copy()
        # Outline the object inside its context crop.
        ox, oy = max(0, left - pad_x), max(0, top - pad_y)
        cv2.rectangle(crop, (left - ox, top - oy), (right - ox, bottom - oy), (0, 229, 255), 3)
        tile = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA) if crop.size else np.zeros((240, 240, 3), np.uint8)
        cv2.rectangle(tile, (0, 0), (70, 46), (0, 0, 0), -1)
        cv2.putText(tile, f"#{object_id}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(tile)
    if not tiles:
        return b""
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * 248, cols * 248, 3), 30, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 248 + 4 : r * 248 + 244, c * 248 + 4 : c * 248 + 244] = tile
    return encode_jpeg(sheet, 88)


def pick_objects(description: str, concept: str, sheet_jpeg: bytes, valid_ids: list[str]) -> dict[str, Any]:
    prompt = (
        f"Each tile shows one tracked '{concept}' from a video, labelled #id, outlined in yellow. "
        f"Return the ids of every tile that matches: \"{description}\". Valid ids: {valid_ids}. "
        "If none match, return an empty list."
    )
    result = meta_api.respond_json(
        [{"type": "input_text", "text": prompt}, meta_api.image_part(sheet_jpeg)],
        PICK_SCHEMA,
        model=config.VISION_MODEL,
        schema_name="object_pick",
        effort="low",
    )
    result["ids"] = [i.lstrip("#") for i in result.get("ids", []) if i.lstrip("#") in valid_ids]
    return result
