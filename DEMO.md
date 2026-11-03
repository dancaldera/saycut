# SayCut: 3-minute demo video script

**Judging weights** (from the official rules, as reported): impact 30% · creative use of Meta's AI tools 25% · demo and video 25% · technical depth 20%.
Every beat below is aimed at one of them.

## Before recording
- Shoot **your own** 15–25 s vertical clip on a tripod or propped-up phone. Real footage of you makes the strongest demo.
  - You talking to camera (for captions), with 1–2 friends walking behind you (for "blur everyone except me"), and a pet or ball if you can (for glow).
  - Keep the camera still. That makes the `remove` shot look best.
- Upload it once before recording so SAM masks are cached. Then do a **fresh upload on camera** and keep the cached project open as a backup.
- Screen record at 1080p or higher. Zoom the browser to 110% so text is readable.

## Script

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:12 | Hook: the finished, edited Reel playing full-screen. Faces blurred, subject glowing, beach background, captions. | “Everything you're seeing was edited by talking. No timeline. No masking. No keyframes.” |
| 0:12–0:30 | A quick montage of a pro editor masking frame by frame in a traditional NLE (your own screen recording), then the SayCut upload screen. | “Hiding a bystander's face or making yourself pop takes creators hours of rotoscoping. 200 million people post Reels every day. SayCut makes it a sentence.” |
| 0:30–0:45 | Drop the clip. The suggestion chips appear. | “I drop in a clip. Muse Spark watches it and suggests edits for this specific video.” |
| 0:45–1:15 | **Hold the mic**: “Blur everyone's face except mine.” Show the live agent trace (SPARK → SAM → SPARK → RENDER), then the result. Hold “👁 Hold for original” to compare. | “Muse Voice Transcribe hears me. Muse Spark plans the edit. SAM 3.1 tracks every face across every frame with stable IDs. Then Muse Spark looks at those tracks and works out which one is *me*.” |
| 1:15–1:40 | Type: “Make the dog glow pink.” Then: “Put me on a beach at sunset.” | “Edits stack. SAM tracks the dog. Muse Image paints a brand-new scene, and I'm composited into it.” |
| 1:40–1:55 | Voice: “Add captions.” | “Captions come straight from Voice Transcribe.” |
| 1:55–2:10 | Click **Make cover**. Show the Muse Image cover. | “And Muse Image designs a scroll-stopping cover from any frame.” |
| 2:10–2:40 | Architecture diagram (from the README) with the 4 model logos. Flash the terminal showing tracked IDs. | “Under the hood: four Meta models in one agent loop. Speculative segmentation starts tracking likely subjects the moment you upload. Masks are decoded with Meta's SAM parser and rendered frame-accurately with the original audio.” |
| 2:40–3:00 | Back to the final Reel, then the GitHub URL. | “SayCut. Edit Reels by saying what you want. It's open source. Link below.” |

## Submission form answers (draft)

**One-liner**: SayCut is a voice-first video editor for Reels: say “blur everyone except me” or “put me on the moon” and it tracks, selects, and edits the right pixels in every frame.

**Problem**: Privacy blurs, subject effects, background swaps, and captions are the edits creators want most. Today they need manual masking or rotoscoping in pro tools. Most creators skip them, or post bystanders' faces unblurred.

**How it uses Meta's models**:
- **Muse Spark 1.3** handles video understanding (suggestions), structured planning, and vision-based selection of the right tracked object.
- **SAM 3.1** does text-prompted video segmentation and tracking with stable IDs.
- **Muse Image** generates backgrounds and Reels covers.
- **Muse Voice Transcribe** handles voice commands and burned-in captions.

**Technical highlights**:
- Object-level selection: “except me”, “the girl in green” (SAM IDs + Spark vision over a labelled contact sheet).
- Speculative background segmentation.
- Cached raw SAM output, so re-renders cost nothing.
- About 60× faster mask decoding via `@meta-sam/parser` in Node.
- Temporal clean-plate object removal.
- Edits that stack and can be undone, rendered with the original audio.

**What's next**: realtime preview on the phone, Instagram share sheet, Ray-Ban Meta capture → auto-edit, multi-clip stories.
