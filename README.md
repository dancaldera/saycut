# ✂️ SayCut: edit Reels by saying what you want

> “Blur every face except mine.” “Make the dog glow pink.” “Put me on a beach at sunset.”
> SayCut turns a sentence (typed or spoken) into a frame-accurate video edit.

**Live demo: [saycut.softmelody.dev](https://saycut.softmelody.dev)**

Built for the **Meta Global AI Developer Hackathon** with four Meta models working together:

| Model | What it does in SayCut |
|---|---|
| **Muse Spark 1.3** | Watches the uploaded clip (video understanding) and suggests edits for that clip. Turns each command into a structured edit plan (JSON schema). Looks at tracked objects to work out *which* person you mean (“the girl in the green shirt”, “everyone except me”). |
| **SAM 3.1** | Segments and **tracks** the target through the whole video from a text prompt (“face”, “dog”, “person”), giving pixel masks with stable object IDs across frames. |
| **Muse Image** | Paints new backgrounds (“put me on the moon”) and designs a scroll-stopping Reels cover from any frame. |
| **Muse Voice Transcribe** | Takes push-to-talk voice commands and burns in captions from the clip's own speech. |

## How it works

```
 "make the girl in green glow pink"  (voice → Muse Voice Transcribe, or typed)
        │
        ▼
 Muse Spark ── plan ──►  { effect: glow, target: "girl", selection: described,
        │                  description: "girl in green shirt", color: "#FF69B4" }
        ▼
 SAM 3.1  ── tracks every "girl" across all frames (stable ids #0, #1, #2)
        │
        ▼
 Muse Spark (vision) ── looks at a labelled contact sheet of the tracks ──► picks #2
        │
        ▼
 Renderer ── feathered masks + effect per frame → H.264 + original audio
```

**Edits** (they stack, and each can be removed or undone):
- `blur` / `pixelate` / `emoji`: hide faces, plates, screens; or cover faces with 😎
- `glow`: neon outline + halo that follows the subject
- `color_pop`: subject in colour, world in black and white
- `spotlight`: darken everything but the subject
- `background_blur`: portrait mode for video
- `background_replace`: Muse Image scene with the subject composited in
- `remove`: erase an object (temporal clean plate + inpainting; best on tripod shots)
- `captions`: Voice Transcribe turns, burned in Reels-style
- **cover**: Muse Image redesigns the current frame into a 9:16 cover with a hook title

**Selection** is what makes this more than a filter app. SAM 3.1 gives every object a stable ID, so:
- “everyone **except me**” → the main subject (picked by Muse Spark vision) is excluded from the effect
- “**the man on the left**” / “**the girl in green**” → Muse Spark picks the matching IDs from a contact sheet
- time ranges: “pixelate the logo **from 2 to 5 seconds**”

**Speed tricks**
- Right after upload, Muse Spark lists the likely subjects and SAM 3.1 starts tracking them **in the background**. By the time you've finished typing, the masks are usually ready.
- SAM output is cached as raw text, so re-renders, undo, and stacked edits never segment the same thing twice.
- Masks are decoded with Meta's official `@meta-sam/parser` in Node (about 60× faster than the pure-Python port). The Python port is used as a fallback when Node is missing.

## Run it

Requirements: Python 3.10+, `ffmpeg`, Node 20+ (optional but recommended), and a Meta Model API key from [dev.meta.ai](https://dev.meta.ai).

```bash
git clone <this repo> && cd saycut
cp .env.example .env          # then paste your MODEL_API_KEY
./run.sh                      # creates .venv, installs deps, starts http://127.0.0.1:8000
```

Open http://127.0.0.1:8000, drop in a clip (≤60 s), and talk to it. Hold 🎙 (or the space bar) to speak.

Configuration (environment variables):

| Variable | Default | |
|---|---|---|
| `MODEL_API_KEY` | (required) | Meta Model API key |
| `SAYCUT_PLANNER_MODEL` | `muse-spark-1.3` | Model for planning |
| `SAYCUT_VISION_MODEL` | `muse-spark-1.3` | Model for clip analysis and object picking |
| `SAYCUT_MAX_SECONDS` | `60` | Input is trimmed to this length |
| `SAYCUT_MAX_LONG_SIDE` | `1280` | Input is downscaled to this size for fast renders |

## Deploy (Cloudflare Containers)

The hosted demo runs on **Cloudflare Containers**. A Worker on the custom domain routes every request to one container instance running this repo's `Dockerfile` (Python, ffmpeg, OpenCV, Node).

```bash
cd cloudflare && npm install
npx wrangler deploy                    # builds the linux/amd64 image with Docker and pushes it
npx wrangler secret put MODEL_API_KEY  # handed to the container as an env var
```

The container image also runs anywhere Docker does:

```bash
docker build -t saycut . && docker run -p 8000:8000 -e MODEL_API_KEY=... saycut
```

When `SAYCUT_PUBLIC=1` (set in the image), usage guardrails turn on because the demo spends a single API key:
- per-IP limits on uploads, edits, and voice commands
- a global daily upload cap
- a 150 MB upload limit
- projects deleted after 24 hours

Tune these with `SAYCUT_UPLOADS_PER_HOUR`, `SAYCUT_EDITS_PER_HOUR`, `SAYCUT_UPLOADS_PER_DAY`, and `SAYCUT_PROJECT_TTL_HOURS`.

## Project layout

```
app/
  server.py     FastAPI routes + background jobs with live progress
  pipeline.py   project state, command → plan → tracks → selection → render
  agent.py      Muse Spark: clip analysis, edit planning, object picking (JSON schema)
  meta_api.py   Meta Model API client: Files, Responses (Spark + SAM), Images, ASR
  tracks.py     SAM 3.1 output → per-frame, per-object masks
  effects.py    blur, glow, colour pop, spotlight, remove, background swap, captions…
  media.py      ffmpeg normalisation + frame I/O
decoder/
  decode.mjs    fast mask decoding with @meta-sam/parser
web/            single-page UI (no build step)
cloudflare/     Worker + wrangler config for Cloudflare Containers
Dockerfile      container image (linux/amd64)
```

## Cost

On a 30-second, 30 fps clip, with prices from Meta's public pricing page:
- SAM 3.1 at $0.20 per 1,000 frames ≈ **$0.18 per tracked concept**
- Muse Image at $0.01 per image
- Muse Spark calls are a few thousand tokens each

A full editing session typically costs well under $1.

## Limitations

- `remove` rebuilds the background from the frames where the object isn't covering it, so it works best with a fixed camera.
- Voice Transcribe gives turn-level timestamps, so caption chunks are timed by spreading the words across each turn.
- SAM 3.1 tracks up to 16 objects per frame.

## License

MIT. See [LICENSE](LICENSE).
