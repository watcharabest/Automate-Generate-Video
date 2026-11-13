import asyncio
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

import edge_tts
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from faster_whisper import WhisperModel
except (ImportError, OSError):
    WhisperModel = None

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
STORY_PLAN_PATH = ROOT_DIR / "output" / "story_plan.json"
TEMP_DIR = ROOT_DIR / "output" / "temp"
FINAL_DIR = ROOT_DIR / "output" / "final"
SFX_DIR = ROOT_DIR / "assets" / "sfx"

WIDTH = 1080
HEIGHT = 1920
FPS = 60


def run_cmd(cmd: List[str], desc: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess command and raise a clear error if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({desc}): {' '.join(cmd)}\nSTDERR:\n{result.stderr}"
        )
    return result


def get_audio_duration(audio_path: Path) -> float:
    """Return exact audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    res = run_cmd(cmd, f"ffprobe duration for {audio_path.name}")
    return float(res.stdout.strip())


def generate_tts_gemini(text: str, out_mp3: Path) -> bool:
    """Try generating TTS via gemini-3.8-flash-tts. Returns True on success, False on failure/quota."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Puck"
                        )
                    )
                ),
            ),
        )
        part = response.candidates[0].content.parts[0]
        audio_bytes = part.inline_data.data
        mime_type = getattr(part.inline_data, "mime_type", "") or ""

        raw_path = out_mp3.with_suffix(".raw_tts")
        raw_path.write_bytes(audio_bytes)

        if "pcm" in mime_type.lower() or not audio_bytes.startswith((b"RIFF", b"ID3", b"\xff\xfb")):
            # Gemini TTS typically returns 24kHz 16-bit mono PCM
            run_cmd(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", str(raw_path),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    str(out_mp3),
                ],
                "convert Gemini PCM to mp3",
            )
        else:
            run_cmd(
                [
                    "ffmpeg", "-y",
                    "-i", str(raw_path),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    str(out_mp3),
                ],
                "convert Gemini audio to mp3",
            )
        raw_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"  [TTS] Gemini TTS unavailable/quota exceeded ({exc}). Falling back to edge-tts...")
        return False


async def _edge_tts_save(text: str, out_mp3: Path, voice: str = "en-US-AndrewNeural") -> None:
    communicate = edge_tts.Communicate(text, voice, rate="+5%")
    await communicate.save(str(out_mp3))


def generate_voiceover(text: str, out_mp3: Path) -> Path:
    """Generate voiceover using Gemini 3.8 Flash TTS with automatic edge-tts fallback."""
    if generate_tts_gemini(text, out_mp3):
        print(f"  [TTS] Generated via gemini-3.8-flash-tts -> {out_mp3.name}")
        return out_mp3

    for attempt in range(1, 4):
        try:
            asyncio.run(_edge_tts_save(text, out_mp3))
            print(f"  [TTS] Generated via edge-tts -> {out_mp3.name}")
            return out_mp3
        except Exception as exc:
            print(f"  [TTS] edge-tts attempt {attempt} failed: {exc}")
            if attempt == 3:
                raise
    return out_mp3


def fetch_scene_image(prompt: str, out_img: Path) -> Path:
    """Download 1080x1920 image from Pollinations.ai, with retry and FFmpeg fallback."""
    encoded = urllib.parse.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={WIDTH}&height={HEIGHT}&nologo=true"

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                out_img.write_bytes(resp.content)
                print(f"  [IMG] Downloaded from Pollinations.ai -> {out_img.name}")
                return out_img
            print(f"  [IMG] Attempt {attempt} returned status {resp.status_code}")
        except Exception as exc:
            print(f"  [IMG] Attempt {attempt} failed: {exc}")

    # Fallback: generate a dark cyberpunk gradient image so rendering never breaks
    print(f"  [IMG] Using FFmpeg fallback image -> {out_img.name}")
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=#0d0f1a:s={WIDTH}x{HEIGHT}:d=1",
            "-frames:v", "1",
            str(out_img),
        ],
        "generate fallback image",
    )
    return out_img


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp H:MM:SS.cs."""
    cs_total = max(0, int(round(seconds * 100)))
    h = cs_total // 360000
    m = (cs_total % 360000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_word_level_ass(
    audio_path: Path,
    out_ass: Path,
    whisper_model: Optional[Any],
    scene_timings: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Extract word-level timestamps with faster-whisper (or proportional fallback) and write dynamic .ass subtitles."""
    words: List[Dict[str, Any]] = []

    if whisper_model is not None:
        segments, _ = whisper_model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
        )
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    txt = w.word.strip()
                    if txt:
                        words.append({"word": txt, "start": w.start, "end": w.end})
    elif scene_timings:
        # Proportional word-level alignment per scene using exact scene audio durations
        for st in scene_timings:
            scene_words = [w for w in st["narration"].split() if w.strip()]
            if not scene_words:
                continue
            total_chars = sum(len(w) for w in scene_words) or 1
            t = st["start"]
            dur = st["duration"]
            for w in scene_words:
                w_dur = dur * (len(w) / total_chars)
                words.append({"word": w, "start": t, "end": t + w_dur})
                t += w_dur

    # YouTube Shorts Safe Zones (1080x1920):
    # Bottom margin >= 25% of 1920 (480px) -> MarginV = 500
    # Right margin >= 15% of 1080 (162px) -> MarginR = 180, MarginL = 180
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,78,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,180,180,500,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for item in words:
        start_str = format_ass_time(item["start"])
        end_str = format_ass_time(max(item["end"], item["start"] + 0.08))
        # Pop-in scale animation + bright gold/cyan accent
        word_text = item["word"].upper().replace("{", "").replace("}", "")
        styled = r"{\fscx115\fscy115\t(0,80,\fscx100\fscy100)\c&H00D7FF&}" + word_text
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled}\n")

    out_ass.write_text("".join(lines), encoding="utf-8")
    print(f"  [SUB] Generated {len(words)} word-level cues -> {out_ass.name}")
    return out_ass


def render_ken_burns_clip(
    image_path: Path,
    duration: float,
    out_clip: Path,
    zoom_in: bool = True,
) -> Path:
    """Render a 1080x1920 60fps video clip with smooth Ken Burns zoompan matching exact audio duration."""
    frames = max(1, int(round(duration * FPS)))
    if zoom_in:
        zoom_expr = f"min(zoom+0.0008,1.15)"
    else:
        zoom_expr = f"if(lte(zoom,1.0),1.15,max(1.001,zoom-0.0008))"

    vf = (
        f"scale=2160:-1,"
        f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"format=yuv420p"
    )
    run_cmd(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", vf,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-r", str(FPS),
            str(out_clip),
        ],
        f"Ken Burns clip {out_clip.name}",
    )
    return out_clip


def escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a Windows/POSIX path for FFmpeg's ass= filter."""
    s = path.resolve().as_posix()
    # Escape colon after drive letter (e.g. C:/ -> C\:/)
    if len(s) >= 2 and s[1] == ":":
        s = s[0] + r"\:" + s[2:]
    return s


def render_episode(episode: Dict[str, Any], topic_slug: str, whisper_model: Optional[Any]) -> Path:
    """Render a complete episode into output/final/."""
    part_num = episode["part_number"]
    scenes = episode["scenes"]
    prefix = f"{topic_slug}_part{part_num}"

    print(f"\n=== Rendering Part {part_num}: {episode['title']} ({len(scenes)} scenes) ===")

    scene_clips: List[Path] = []
    scene_audios: List[Path] = []
    sfx_events: List[Dict[str, Any]] = []
    scene_timings: List[Dict[str, Any]] = []
    current_time = 0.0

    for idx, scene in enumerate(scenes):
        s_num = scene["scene_number"]
        print(f"\n[Scene {s_num}/{len(scenes)}]")

        audio_path = TEMP_DIR / f"{prefix}_s{s_num}.mp3"
        img_path = TEMP_DIR / f"{prefix}_s{s_num}.jpg"
        clip_path = TEMP_DIR / f"{prefix}_s{s_num}.mp4"

        generate_voiceover(scene["narration"], audio_path)
        duration = get_audio_duration(audio_path)
        print(f"  [DUR] {duration:.2f}s")

        fetch_scene_image(scene["visual_prompt"], img_path)
        render_ken_burns_clip(img_path, duration, clip_path, zoom_in=(idx % 2 == 0))

        scene_clips.append(clip_path)
        scene_audios.append(audio_path)

        cue = (scene.get("sfx_cue") or "none").strip().lower()
        if cue != "none":
            sfx_file = SFX_DIR / f"{cue}.mp3"
            if sfx_file.exists():
                sfx_events.append({"path": sfx_file, "start": current_time})
            else:
                print(f"  [SFX] '{cue}.mp3' not found in {SFX_DIR}, skipping gracefully.")

        scene_timings.append({
            "narration": scene["narration"],
            "start": current_time,
            "duration": duration,
        })
        current_time += duration

    # 1. Concatenate scene videos
    video_list_txt = TEMP_DIR / f"{prefix}_videos.txt"
    video_list_txt.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in scene_clips),
        encoding="utf-8",
    )
    concat_video = TEMP_DIR / f"{prefix}_concat_video.mp4"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(video_list_txt),
            "-c", "copy",
            str(concat_video),
        ],
        "concat scene videos",
    )

    # 2. Concatenate scene voiceover audios
    audio_list_txt = TEMP_DIR / f"{prefix}_audios.txt"
    audio_list_txt.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in scene_audios),
        encoding="utf-8",
    )
    concat_voice = TEMP_DIR / f"{prefix}_concat_voice.mp3"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(audio_list_txt),
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(concat_voice),
        ],
        "concat scene audios",
    )

    # 3. Generate word-level .ass subtitles from full voiceover
    ass_path = TEMP_DIR / f"{prefix}.ass"
    generate_word_level_ass(concat_voice, ass_path, whisper_model, scene_timings)

    # 4. Mix voiceover + SFX cues (if any)
    mixed_audio = TEMP_DIR / f"{prefix}_mixed_audio.m4a"
    if sfx_events:
        cmd = ["ffmpeg", "-y", "-i", str(concat_voice)]
        filter_parts = ["[0:a]volume=1.0[vo]"]
        mix_inputs = ["[vo]"]
        for i, ev in enumerate(sfx_events, start=1):
            cmd.extend(["-i", str(ev["path"])])
            delay_ms = int(round(ev["start"] * 1000))
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms},volume=0.35[sfx{i}]"
            )
            mix_inputs.append(f"[sfx{i}]")
        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:normalize=0[aout]"
        )
        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k",
            str(mixed_audio),
        ])
        run_cmd(cmd, "mix voiceover and SFX")
    else:
        run_cmd(
            [
                "ffmpeg", "-y",
                "-i", str(concat_voice),
                "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ],
            "encode voiceover to aac",
        )

    # 5. Burn .ass subtitles and mux final 1080x1920 60fps MP4
    final_mp4 = FINAL_DIR / f"{prefix}.mp4"
    escaped_ass = escape_ffmpeg_filter_path(ass_path)
    run_cmd(
        [
            "ffmpeg", "-y",
            "-i", str(concat_video),
            "-i", str(mixed_audio),
            "-vf", f"ass='{escaped_ass}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "copy",
            "-shortest",
            str(final_mp4),
        ],
        "burn subtitles and mux final mp4",
    )

    print(f"\n[DONE] Final video saved to: {final_mp4}")
    return final_mp4


def main() -> None:
    if not STORY_PLAN_PATH.exists():
        print(f"Error: {STORY_PLAN_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    plan = json.loads(STORY_PLAN_PATH.read_text(encoding="utf-8"))
    topic_slug = plan["topic_slug"]

    whisper_model = None
    if WhisperModel is not None:
        print("Loading faster-whisper model (base, cpu, int8)...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    else:
        print("faster-whisper unavailable; using proportional word-level alignment.")

    for episode in plan["episodes"]:
        render_episode(episode, topic_slug, whisper_model)


if __name__ == "__main__":
    main()
