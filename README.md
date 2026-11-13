# Automated Faceless YouTube Shorts Factory (US Tech & AI Niche)

An end-to-end automated pipeline for generating high-retention vertical videos (9:16, 1080x1920, 60fps) targeting US Tech & AI audiences, designed to run on a **$0/month cloud architecture**.

---

## 1. Project Overview & Art Direction
* **Target Audience:** US & European Tech/AI professionals, developers, and productivity enthusiasts (High RPM niche).
* **Visual Style (Art Direction):** `Isometric 3D Pixel Art` / `Voxel Art` (vibrant neon lighting, retro-futuristic, 9:16 vertical).
* **Audio Direction:** American English AI voiceover paired with event-driven UI/Retro sound effects (`glitch`, `click`, `whoosh`, `pop`) mapped to scene transitions—no generic background music needed.
* **Subtitle Style:** Word-level dynamic captions (Hormozi style), positioned within YouTube Shorts Safe Zones (bottom margin ≥ 25%, right margin ≥ 15%).

---

## 2. Tech Stack & Zero-Cost Architecture
To stay within a $0 operational budget (leveraging existing free tiers), compute-heavy tasks are separated from control/API orchestration:

| Layer | Tool / Service | Cost / Quota Tier | Purpose |
| :--- | :--- | :--- | :--- |
| **Brain & Planner** | `Gemini Flash API` (`google-genai` SDK) | Free Tier | Trend filtering, multi-part decision making, and structured JSON script generation. |
| **Voiceover (TTS)** | `Gemini 3.8 Flash TTS` / `3.1 Flash TTS` | Free Tier (10 RPD) | Natural English narration generation (Fallback: `edge-tts`). |
| **Word Timestamps** | `Faster-Whisper` / `FFmpeg libwhisper` | Local / Free | Word-level alignment for animated subtitles. |
| **Visual Generation** | `Pollinations.ai API` (Primary) / `Pexels API` | Free / Unlimited | Generating 9:16 `Isometric 3D Pixel Art` images (since Google AI Studio Free Tier has `0/0` quota for Imagen/Veo). |
| **Sound Effects (SFX)** | Local Static Assets (`soundcn` + `100-free-soundeffects`) | Open-Source / Free | Crisp UI/Game SFX stored locally in `assets/sfx/` for zero-latency FFmpeg mixing. |
| **Video Assembly** | `Python` + `FFmpeg` (v9.0.2+) | Local / Free | Ken Burns zoom

4. Data Schema (output/story_plan.json)
The script generator outputs strict JSON validated via Pydantic:

JSON
{
  "topic_slug": "ai-agents-replacing-saas-2026",
  "is_multi_part": false,
  "episodes": [
    {
      "part_number": 1,
      "title": "Why Traditional SaaS Apps Are Dying in 2026",
      "scenes": [
        {
          "scene_number": 1,
          "narration": "Stop paying fifty bucks a month for ten different dashboards. In 2026, traditional SaaS is officially dying.",
          "visual_prompt": "Isometric 3D Pixel Art, vibrant neon lighting, 9:16 vertical, ...",
          "sfx_cue": "glitch"
        }
      ]
    }
  ]
}
5. Google Drive Naming & Organization Rules
When uploading rendered videos to Google Drive:

Date Format: [dd-mm-yy] (using hyphens - instead of slashes / to avoid path conflicts).

Single-Part Story (is_multi_part: false):

Upload directly as: [dd-mm-yy] <Story Title>.mp4

Multi-Part Story (is_multi_part: true):

Create a dedicated parent folder: [dd-mm-yy] <Story Title>/

Upload episodes inside as: Part 1 - <Episode Title>.mp4, Part 2 - <Episode Title>.mp4, etc.

6. Development Roadmap & Current Status
Phase 1: Core Video Engine (In Progress)

[x] Environment setup (Python + FFmpeg 9.0.2 full build verified).

[x] 1_plan_story.py tested and outputting valid story_plan.json.

[ ] Populate assets/sfx/ with glitch, click, whoosh, pop audio files.

[ ] Build 2_render_video.py to generate TTS audio, fetch 3D Pixel Art images, generate word-level .ass subtitles, and composite the final 1080x1920 .mp4 via FFmpeg.

Phase 2: Trend Automation & Cloud Storage

[ ] Build 3_trend_hunter.py to fetch weekly US Tech/AI trends and feed them into 1_plan_story.py.

[ ] Build 4_drive_uploader.py using Google Drive & Sheets APIs (Service Account) with the [dd-mm-yy] naming convention.

Phase 3: Serverless CI/CD Execution

[ ] Configure .github/workflows/daily_factory.yml to run the full pipeline automatically via Cron on GitHub Actions.

Phase 4: Web Control Dashboard

[ ] Build a Next.js + MongoDB dashboard deployed on Vercel to track pipeline runs, preview scripts, and manually queue custom topics.

7. Instructions for CLI Agent
Inspect output/story_plan.json and assets/sfx/ to verify existing assets.

Implement pipeline/2_render_video.py using Python and FFmpeg:

Generate scene narration using Gemini 3.8 Flash TTS (with edge-tts fallback).

Download vertical 1080x1920 images from Pollinations.ai (https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&nologo=true) for each scene's visual_prompt.

Apply smooth zoom/pan (Ken Burns effect) matching each scene's exact audio duration.

Overlay the corresponding SFX file from assets/sfx/{sfx_cue}.mp3 at the exact start timestamp of each scene.

Generate word-level animated subtitles in .ass format within safe zones and burn them into the final video.