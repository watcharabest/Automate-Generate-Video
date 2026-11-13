import asyncio
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import edge_tts
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageFilter, ImageOps, ImageStat

try:
    from faster_whisper import WhisperModel
except (ImportError, OSError):
    WhisperModel = None

try:
    from rembg import new_session as rembg_new_session, remove as rembg_remove
except (ImportError, OSError):
    rembg_new_session = None
    rembg_remove = None

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
STORY_PLAN_PATH = ROOT_DIR / "output" / "story_plan.json"
STYLE_CONFIG_PATH = ROOT_DIR / "config" / "style.json"
TEMP_DIR = ROOT_DIR / "output" / "temp"
FINAL_DIR = ROOT_DIR / "output" / "final"
SFX_DIR = ROOT_DIR / "assets" / "sfx"

STYLE = json.loads(STYLE_CONFIG_PATH.read_text(encoding="utf-8"))
WIDTH = STYLE["video"]["width"]
HEIGHT = STYLE["video"]["height"]
FPS = STYLE["video"]["fps"]
SFX_VOLUME = STYLE["audio"]["sfx_volume"]
VO_VOLUME = STYLE["audio"]["voiceover_volume"]
SUB_CFG = STYLE["subtitles"]


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
    """Try generating TTS via gemini-3.8-flash-tts Interactions API. Returns True on success."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False

    try:
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        interaction = client.interactions.create(
            model="gemini-3.8-flash-tts",
            input=[{
                "type": "user_input",
                "content": [{
                    "type": "text",
                    "text": text,
                    "annotations": [{
                        "type": "speech_metadata",
                        "style": "energetic, clear, and engaging",
                    }],
                }],
            }],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [{"voice": "Puck"}]
            },
        )
        audio_bytes = base64.b64decode(interaction.output_audio.data)
        raw_path = out_mp3.with_suffix(".wav")
        raw_path.write_bytes(audio_bytes)

        run_cmd(
            [
                "ffmpeg", "-y",
                "-i", str(raw_path),
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(out_mp3),
            ],
            "convert Gemini WAV to mp3",
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
    """Generate voiceover using edge-tts (fast, reliable fallback while Gemini TTS quota is exhausted)."""
    voices = ["en-US-AndrewNeural", "en-US-GuyNeural", "en-US-AriaNeural"]
    for attempt in range(1, 6):
        voice = voices[(attempt - 1) % len(voices)]
        try:
            asyncio.run(_edge_tts_save(text, out_mp3, voice=voice))
            print(f"  [TTS] Generated via edge-tts ({voice}) -> {out_mp3.name}")
            return out_mp3
        except Exception as exc:
            print(f"  [TTS] edge-tts attempt {attempt} ({voice}) failed: {exc}")
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
    return out_mp3


def _clean_visual_prompt(prompt: str) -> str:
    """Remove resolution/aspect strings and append cohesive 3D Pixel Art style."""
    cleaned = re.sub(r"\b(9:16\s*vertical|9:16|1080x1920|vertical)\b", "", prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*,+", ", ", cleaned).strip(" ,")
    return f"{cleaned}, single cohesive scene, full frame, vibrant Isometric 3D Pixel Art"


def _validate_and_save_image(img_bytes: bytes, out_img: Path) -> bool:
    """Validate image bytes with PIL and ensure mean brightness > 15 before saving."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mean_brightness = ImageStat.Stat(img.convert("L")).mean[0]
        if mean_brightness <= 15:
            print(f"  [IMG] Rejected dark/blank image (brightness={mean_brightness:.1f})")
            return False
        img.save(out_img, format="JPEG", quality=95)
        return True
    except Exception as exc:
        print(f"  [IMG] Invalid image bytes: {exc}")
        return False


def fetch_scene_image(prompt: str, out_img: Path) -> Path:
    """Generate image via Cloudflare Workers AI (FLUX.1 Schnell), with HF fallback."""
    clean_prompt = _clean_visual_prompt(prompt)

    # 1. Primary: Cloudflare Workers AI
    cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cf_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if cf_account and cf_token:
        cf_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        headers = {"Authorization": f"Bearer {cf_token}"}
        payload = {"prompt": clean_prompt}
        for attempt in range(1, 3):
            try:
                resp = requests.post(cf_url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    ctype = resp.headers.get("content-type", "")
                    if "application/json" in ctype:
                        b64 = resp.json().get("result", {}).get("image", "")
                        img_bytes = base64.b64decode(b64) if b64 else b""
                    else:
                        img_bytes = resp.content
                    if img_bytes and _validate_and_save_image(img_bytes, out_img):
                        print(f"  [IMG] Generated via Cloudflare FLUX.1 -> {out_img.name}")
                        return out_img
                else:
                    print(f"  [IMG] Cloudflare attempt {attempt} returned {resp.status_code}")
            except Exception as exc:
                print(f"  [IMG] Cloudflare attempt {attempt} failed: {exc}")

    # 2. Fallback: Hugging Face Inference API (router.huggingface.co)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        hf_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"}
        for attempt in range(1, 3):
            try:
                resp = requests.post(hf_url, headers=headers, json={"inputs": clean_prompt}, timeout=60)
                if resp.status_code == 200 and _validate_and_save_image(resp.content, out_img):
                    print(f"  [IMG] Generated via HuggingFace FLUX.1 -> {out_img.name}")
                    return out_img
                print(f"  [IMG] HuggingFace attempt {attempt} returned {resp.status_code}")
            except Exception as exc:
                print(f"  [IMG] HuggingFace attempt {attempt} failed: {exc}")

    # 3. Final fallback: FFmpeg dark background
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
Style: Default,{SUB_CFG['font_name']},{SUB_CFG['font_size']},{SUB_CFG['primary_color']},{SUB_CFG['accent_color']},{SUB_CFG['outline_color']},{SUB_CFG['back_color']},{SUB_CFG['bold']},0,0,0,100,100,0,0,1,{SUB_CFG['outline']},{SUB_CFG['shadow']},{SUB_CFG['alignment']},{SUB_CFG['margin_l']},{SUB_CFG['margin_r']},{SUB_CFG['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        start_str = format_ass_time(chunk[0]["start"])
        end_str = format_ass_time(max(chunk[-1]["end"], chunk[0]["start"] + 0.12))
        phrase = " ".join(w["word"] for w in chunk).upper().replace("{", "").replace("}", "")
        styled = rf"{{\fscx112\fscy112\t(0,80,\fscx100\fscy100)\c{SUB_CFG['accent_color']}}}" + phrase
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled}\n")

    out_ass.write_text("".join(lines), encoding="utf-8")
    print(f"  [SUB] Generated {len(lines) - 1} phrase cues (2-3 words) -> {out_ass.name}")
    return out_ass


def extract_foreground_layer(
    image_path: Path,
    bg_out: Path,
    fg_out: Path,
    rembg_session: Optional[Any],
) -> Optional[Path]:
    """Center-crop image to 1080x1920 and extract foreground cutout if alpha is 8%-75%."""
    img = Image.open(image_path).convert("RGB")
    cropped = ImageOps.fit(img, (WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    cropped.save(bg_out, format="JPEG", quality=95)

    if rembg_remove is None or rembg_session is None:
        return None

    try:
        fg_rgba = rembg_remove(cropped, session=rembg_session).convert("RGBA")
        alpha = fg_rgba.getchannel("A")
        non_zero = sum(1 for p in alpha.getdata() if p > 128)
        coverage = non_zero / (WIDTH * HEIGHT)
        if 0.08 <= coverage <= 0.75:
            # Add subtle drop shadow behind foreground cutout
            shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            shadow_alpha = alpha.point(lambda p: int(p * 0.45))
            shadow.paste((0, 0, 0, 255), (10, 14), shadow_alpha)
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
            combined = Image.alpha_composite(shadow, fg_rgba)
            combined.save(fg_out, format="PNG")
            print(f"  [2.5D] Foreground mask {coverage*100:.1f}% -> {fg_out.name}")
            return fg_out
        print(f"  [2.5D] Mask {coverage*100:.1f}% outside 8%-75%, using Ken Burns fallback.")
    except Exception as exc:
        print(f"  [2.5D] rembg failed ({exc}), using Ken Burns fallback.")
    return None


def render_ken_burns_clip(
    image_path: Path,
    duration: float,
    out_clip: Path,
    zoom_in: bool = True,
    rembg_session: Optional[Any] = None,
) -> Path:
    """Render 1080x1920 60fps clip with 2.5D Parallax (or Ken Burns fallback) + 0.20s impact punch."""
    frames = max(1, int(round(duration * FPS)))
    bg_path = out_clip.with_name(out_clip.stem + "_bg.jpg")
    fg_path = out_clip.with_name(out_clip.stem + "_fg.png")

    extracted_fg = extract_foreground_layer(image_path, bg_path, fg_path, rembg_session)

    if extracted_fg is not None:
        # 2.5D Parallax: blurred/zoom-out background + floating/zoom-in foreground + 0.20s punch
        fc = (
            f"[0:v]scale={WIDTH}:{HEIGHT},gblur=sigma=2,scale=2160:-1,"
            f"zoompan=z='if(lte(zoom,1.0),1.15,max(1.001,zoom-0.0006))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}[bg];"
            f"[1:v]format=rgba,scale=2160:-1,"
            f"zoompan=z='min(zoom+0.0007,1.12)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}[fg];"
            f"[bg][fg]overlay=0:'8*sin(t*2.5)':shortest=1,"
            f"scale=w='if(lt(t,0.20),iw*(1.06-0.30*t),iw)':h=-1:eval=frame,"
            f"crop={WIDTH}:{HEIGHT},format=yuv420p[vout]"
        )
        run_cmd(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(bg_path),
                "-loop", "1", "-i", str(extracted_fg),
                "-filter_complex", fc,
                "-map", "[vout]",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-r", str(FPS),
                str(out_clip),
            ],
            f"2.5D parallax clip {out_clip.name}",
        )
        return out_clip

    # Fallback: Standard Ken Burns + 0.20s impact punch
    zoom_expr = "min(zoom+0.0008,1.15)" if zoom_in else "if(lte(zoom,1.0),1.15,max(1.001,zoom-0.0008))"
    vf = (
        f"scale=2160:-1,"
        f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"scale=w='if(lt(t,0.20),iw*(1.06-0.30*t),iw)':h=-1:eval=frame,"
        f"crop={WIDTH}:{HEIGHT},format=yuv420p"
    )
    run_cmd(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(bg_path),
            "-vf", vf,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-r", str(FPS),
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


def concat_videos_xfade(
    scene_clips: List[Path],
    scene_durations: List[float],
    out_video: Path,
    xfade_dur: float = 0.25,
) -> Path:
    """Concatenate scene clips using smooth 0.25s xfade transitions."""
    if len(scene_clips) == 1:
        run_cmd(["ffmpeg", "-y", "-i", str(scene_clips[0]), "-c", "copy", str(out_video)], "copy single clip")
        return out_video

    transitions = ["pixelize", "smoothleft", "smoothright", "wipeup", "fadefast"]
    cmd = ["ffmpeg", "-y"]
    for clip in scene_clips:
        cmd.extend(["-i", str(clip)])

    filter_parts = []
    prev_label = "[0:v]"
    cumulative = scene_durations[0]

    for i in range(1, len(scene_clips)):
        t_name = transitions[(i - 1) % len(transitions)]
        offset = max(0.0, cumulative - i * xfade_dur)
        out_label = "[vout]" if i == len(scene_clips) - 1 else f"[vx{i}]"
        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition={t_name}:duration={xfade_dur:.2f}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label
        cumulative += scene_durations[i]

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(out_video),
    ])
    run_cmd(cmd, "xfade concat scene videos")
    return out_video


def render_episode(
    episode: Dict[str, Any],
    topic_slug: str,
    whisper_model: Optional[Any],
    rembg_session: Optional[Any] = None,
) -> Path:
    """Render a complete episode into output/final/."""
    part_num = episode["part_number"]
    scenes = episode["scenes"]
    prefix = f"{topic_slug}_part{part_num}"
    xfade_dur = 0.25

    print(f"\n=== Rendering Part {part_num}: {episode['title']} ({len(scenes)} scenes) ===")

    scene_clips: List[Path] = []
    scene_durations: List[float] = []
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
        render_ken_burns_clip(
            img_path,
            duration,
            clip_path,
            zoom_in=(idx % 2 == 0),
            rembg_session=rembg_session,
        )

        scene_clips.append(clip_path)
        scene_durations.append(duration)
        scene_audios.append(audio_path)

        # Adjust start timestamp for xfade overlap
        adjusted_start = max(0.0, current_time - idx * xfade_dur)

        cue = (scene.get("sfx_cue") or "none").strip().lower()
        if cue != "none":
            sfx_file = SFX_DIR / f"{cue}.mp3"
            if sfx_file.exists():
                sfx_events.append({"path": sfx_file, "start": adjusted_start})
            else:
                print(f"  [SFX] '{cue}.mp3' not found in {SFX_DIR}, skipping gracefully.")

        scene_timings.append({
            "narration": scene["narration"],
            "start": adjusted_start,
            "duration": duration,
        })
        current_time += duration

    # 1. Concatenate scene videos with 0.25s xfade transitions
    concat_video = TEMP_DIR / f"{prefix}_concat_video.mp4"
    concat_videos_xfade(scene_clips, scene_durations, concat_video, xfade_dur=xfade_dur)

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
        filter_parts = [f"[0:a]volume={VO_VOLUME}[vo]"]
        mix_inputs = ["[vo]"]
        for i, ev in enumerate(sfx_events, start=1):
            cmd.extend(["-i", str(ev["path"])])
            delay_ms = int(round(ev["start"] * 1000))
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms},volume={SFX_VOLUME}[sfx{i}]"
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

    rembg_session = None
    if rembg_new_session is not None:
        print("Loading rembg model (isnet-general-use)...")
        rembg_session = rembg_new_session("isnet-general-use")

    for episode in plan["episodes"]:
        render_episode(episode, topic_slug, whisper_model, rembg_session=rembg_session)


if __name__ == "__main__":
    main()
