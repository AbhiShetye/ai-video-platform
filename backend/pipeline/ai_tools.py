"""
AI-powered video processing tools:
  - run_auto_captions  — Whisper speech-to-text → SRT + optional burn-in
  - run_bg_remove      — rembg background removal per frame
  - run_stabilize      — FFmpeg vidstab two-pass stabilization
"""

import os
import logging
import shutil
import subprocess
import tempfile
import threading
import uuid

from vision.frames import extract_frames
from pipeline.engine import (
    _video_info, _ffmpeg, _add_audio, _STORAGE, CRF, jobs
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Whisper model cache (loaded once per worker process)
# ─────────────────────────────────────────────────────────────────────────────
_whisper_model      = None
_whisper_model_lock = threading.Lock()


def _load_whisper(size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                log.info("Loading Whisper '%s' on CPU …", size)
                _whisper_model = WhisperModel(
                    size, device="cpu", compute_type="int8"
                )
                log.info("Whisper loaded.")
    return _whisper_model


def _fmt_srt_time(seconds: float) -> str:
    ms  = int((seconds % 1) * 1000)
    s   = int(seconds) % 60
    m   = (int(seconds) // 60) % 60
    h   = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segments, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. AUTO CAPTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_auto_captions(video_path: str, language: str, burn: bool,
                      style: str, job_id: str | None = None):
    """
    Transcribe video with Whisper and produce subtitles.
    language: ISO 639-1 code or "auto"
    burn:     True → bake SRT into output video; False → return .srt file
    style:    "white" | "yellow" | "large"
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== AUTO-CAPTIONS %s  lang=%s burn=%s ===", job_id, language, burn)

    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        # ── 1. Extract mono 16 kHz audio (Whisper requirement) ──────────────
        jobs[job_id]["progress"] = 8
        audio_path = os.path.join(out_dir, "audio.wav")
        _ffmpeg("-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path, label="extract_audio")

        # ── 2. Transcribe ───────────────────────────────────────────────────
        jobs[job_id]["progress"] = 20
        model = _load_whisper("base")
        lang_param = None if language in ("auto", "") else language
        log.info("Transcribing with Whisper (lang=%s) …", lang_param or "auto-detect")
        segments_iter, info = model.transcribe(
            audio_path, beam_size=5, language=lang_param,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}
        )
        # Materialise the lazy iterator so we can write the SRT
        segments = list(segments_iter)
        log.info("Detected language: %s  (%.0f%%)", info.language,
                 info.language_probability * 100)
        jobs[job_id]["progress"] = 75

        # ── 3. Write SRT ────────────────────────────────────────────────────
        srt_path = os.path.join(out_dir, "captions.srt")
        _write_srt(segments, srt_path)
        log.info("Wrote %d subtitle segments → %s", len(segments), srt_path)

        if not burn:
            # Return the .srt file directly for download
            jobs[job_id].update({
                "status": "completed", "output": srt_path,
                "srt": srt_path, "progress": 100,
                "language": info.language,
                "segments": len(segments),
            })
            return job_id

        # ── 4. Burn subtitles into video ─────────────────────────────────────
        jobs[job_id]["progress"] = 80
        style_map = {
            "white":  "Fontsize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Bold=1,MarginV=30",
            "yellow": "Fontsize=20,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,Outline=2,Bold=1,MarginV=30",
            "large":  "Fontsize=30,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Bold=1,MarginV=30",
        }
        srt_style  = style_map.get(style, style_map["white"])
        # SRT path needs forward slashes for FFmpeg on Windows
        srt_safe   = srt_path.replace("\\", "/").replace(":", "\\:")
        final      = os.path.join(out_dir, "final.mp4")
        _ffmpeg(
            "-i", video_path,
            "-vf", (f"subtitles='{srt_safe}'"
                    f":force_style='{srt_style}',"
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2"),
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            final, label="burn_subs"
        )

        jobs[job_id].update({
            "status": "completed", "output": final,
            "srt": srt_path, "progress": 100,
            "language": info.language,
            "segments": len(segments),
        })
        log.info("=== AUTO-CAPTIONS %s DONE ===", job_id)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== AUTO-CAPTIONS %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKGROUND REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

_rembg_session      = None
_rembg_session_lock = threading.Lock()


def _load_rembg():
    global _rembg_session
    if _rembg_session is None:
        with _rembg_session_lock:
            if _rembg_session is None:
                from rembg import new_session
                log.info("Loading rembg u2net model …")
                _rembg_session = new_session("u2net")
                log.info("rembg loaded.")
    return _rembg_session


_BG_COLORS = {
    "black":   (0,   0,   0),
    "white":   (255, 255, 255),
    "green":   (0,   255, 0),
    "blue":    (0,   0,   255),
    "red":     (255, 0,   0),
}


def run_bg_remove(video_path: str, bg_color: str, job_id: str | None = None):
    """
    Remove background from every video frame using rembg (u2net).
    bg_color: "black" | "white" | "green" | "blue" | "red"
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== BG-REMOVE %s  bg=%s ===", job_id, bg_color)

    try:
        from rembg import remove as rembg_remove
        from PIL import Image

        out_dir    = os.path.join(_STORAGE, job_id, "output")
        frames_dir = os.path.join(_STORAGE, job_id, "frames")
        edited_dir = os.path.join(_STORAGE, job_id, "edited")
        os.makedirs(out_dir, exist_ok=True)

        orig_fps, duration, _ = _video_info(video_path)
        # 2 fps extraction: fast path, smooth enough for background effects
        extract_fps = 2
        jobs[job_id]["progress"] = 5
        frame_paths = extract_frames(video_path, frames_dir, fps=extract_fps)
        if not frame_paths:
            raise RuntimeError("No frames extracted")
        log.info("Extracted %d frames at %dfps", len(frame_paths), extract_fps)

        session = _load_rembg()
        bg_rgb  = _BG_COLORS.get(bg_color, (0, 0, 0))
        os.makedirs(edited_dir, exist_ok=True)
        n = len(frame_paths)

        for i, fp in enumerate(frame_paths):
            img = Image.open(fp).convert("RGB")
            # Downscale to max 640px for speed, then upscale result
            max_dim = 640
            scale   = min(max_dim / img.width, max_dim / img.height, 1.0)
            w_s     = max(2, int(img.width  * scale) & ~1)
            h_s     = max(2, int(img.height * scale) & ~1)
            small   = img.resize((w_s, h_s), Image.LANCZOS)

            output = rembg_remove(small, session=session)          # RGBA

            bg     = Image.new("RGBA", output.size, (*bg_rgb, 255))
            result = Image.alpha_composite(bg, output).convert("RGB")

            if scale < 1.0:
                result = result.resize((img.width, img.height), Image.LANCZOS)

            out_fp = os.path.join(edited_dir, f"edited_{i:04d}.jpg")
            result.save(out_fp, quality=95)

            jobs[job_id]["progress"] = 5 + int((i + 1) / n * 73)
            log.info("  BG-REMOVE frame %d/%d", i + 1, n)

        # Encode frames → video segment
        jobs[job_id]["progress"] = 80
        tmp_dir = tempfile.mkdtemp(prefix="frameai_bg_")
        seg_b   = os.path.join(tmp_dir, "seg.mp4")
        _ffmpeg(
            "-framerate", str(extract_fps),
            "-i", os.path.join(edited_dir, "edited_%04d.jpg"),
            "-vf", f"fps={orig_fps:.3f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            seg_b, label="bg_encode"
        )

        # Mux original audio
        jobs[job_id]["progress"] = 93
        final = os.path.join(out_dir, "final.mp4")
        _add_audio(seg_b, video_path, final)

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== BG-REMOVE %s DONE ===", job_id)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== BG-REMOVE %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. VIDEO STABILIZATION (FFmpeg vidstab — two-pass)
# ─────────────────────────────────────────────────────────────────────────────

def run_stabilize(video_path: str, shakiness: int, smoothing: int,
                  job_id: str | None = None):
    """
    Stabilize shaky video using FFmpeg vidstab (two-pass).
    shakiness: 1–10 (how shaky the input is)
    smoothing: 5–100 (how aggressively to smooth motion)
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== STABILIZE %s  shakiness=%d smoothing=%d ===",
             job_id, shakiness, smoothing)

    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="frameai_stab_")
        trf     = os.path.join(tmp_dir, "transforms.trf")
        final   = os.path.join(out_dir, "final.mp4")

        # Use a relative filename for the TRF so FFmpeg's filter parser never
        # sees a Windows drive-letter colon (which it mistakes for an option separator).
        # We run both passes with cwd=tmp_dir so "transforms.trf" resolves correctly.
        trf_rel = "transforms.trf"

        # Pass 1: detect transforms
        jobs[job_id]["progress"] = 10
        log.info("Stabilize pass 1: detecting transforms …")
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", video_path,
             "-vf", (f"vidstabdetect=stepsize=6:"
                     f"shakiness={shakiness}:accuracy=9:"
                     f"result={trf_rel}"),
             "-f", "null", "-"],
            capture_output=True, text=True, cwd=tmp_dir
        )
        if r.returncode != 0:
            raise RuntimeError(f"vidstabdetect failed:\n{r.stderr[-1000:]}")
        jobs[job_id]["progress"] = 55

        # Pass 2: apply transforms (also run with cwd=tmp_dir so trf_rel resolves)
        log.info("Stabilize pass 2: applying transforms …")
        r2 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", video_path,
             "-vf", (f"vidstabtransform=smoothing={smoothing}:"
                     f"input={trf_rel},"
                     "unsharp=5:5:0.8:3:3:0.4,"
                     "scale=trunc(iw/2)*2:trunc(ih/2)*2"),
             "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
             "-c:a", "copy", final],
            capture_output=True, text=True, cwd=tmp_dir
        )
        if r2.returncode != 0:
            raise RuntimeError(f"vidstabtransform failed:\n{r2.stderr[-1000:]}")

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== STABILIZE %s DONE ===", job_id)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== STABILIZE %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id
