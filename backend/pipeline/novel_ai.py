"""
Novel AI features — genuinely unique capabilities no mainstream video editor offers:

  1. detect_filler_words     — Whisper word-timestamps → find um/uh/like/basically
  2. run_cut_segments        — Remove specific time-ranges from video (concat approach)
  3. run_silence_remover     — Whisper segments → auto-cut silent gaps
  4. run_speech_enhance      — Multi-stage FFmpeg audio restoration chain
  5. detect_beats            — librosa beat-tracking → rhythm-aligned cut markers
  6. run_auto_speedramp      — Frame-difference analysis → boring fast / action slow
  7. run_smart_thumbnail     — YOLO + Laplacian sharpness → best-frame extractor
  8. run_video_ocr           — EasyOCR every N frames → searchable text index
"""

import os, logging, shutil, subprocess, tempfile, threading, uuid, json
import cv2, numpy as np

from vision.frames import extract_frames
from pipeline.engine import _video_info, _ffmpeg, _add_audio, _STORAGE, CRF, jobs

log = logging.getLogger(__name__)

# ─── shared helpers ───────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, out_dir: str) -> str:
    """Extract 16kHz mono WAV suitable for Whisper."""
    audio = os.path.join(out_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio],
        check=True
    )
    return audio


def _concat_parts(parts: list, out_dir: str, job_id: str) -> str:
    """Concatenate a list of MP4 clip paths into one final.mp4."""
    final = os.path.join(out_dir, "final.mp4")
    tmp   = tempfile.mkdtemp(prefix="frameai_cat_")
    txt   = os.path.join(tmp, "concat.txt")
    with open(txt, "w") as f:
        for p in parts:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", txt,
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            final, label="concat_parts")
    return final


# ─── 1 + 2. FILLER WORD CUTTER ───────────────────────────────────────────────

FILLER_WORDS = {
    "um", "uh", "er", "eh", "hmm", "hm", "ah", "uhh", "umm", "uhm",
    "like", "so", "right", "okay", "ok", "actually", "basically",
    "literally", "you know", "i mean", "kind of", "sort of",
}

_whisper_lock = threading.Lock()
_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                log.info("Loading Whisper 'base' for novel-AI …")
                _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def detect_filler_words(video_path: str) -> list:
    """
    Return list of filler-word occurrences with word-level timestamps.
    [{index, word, start, end}]  — immediate (no job_id).
    """
    tmp   = tempfile.mkdtemp(prefix="frameai_fw_")
    audio = _extract_audio_wav(video_path, tmp)

    model = _load_whisper()
    segments, _ = model.transcribe(audio, word_timestamps=True, beam_size=5,
                                   vad_filter=True)
    fillers = []
    idx = 0
    for seg in segments:
        words = getattr(seg, "words", None) or []
        for w in words:
            clean = w.word.strip().lower().strip(".,!?;:'\"")
            if clean in FILLER_WORDS and (w.end - w.start) < 1.5:
                fillers.append({
                    "index": idx,
                    "word":  w.word.strip(),
                    "start": round(w.start, 3),
                    "end":   round(w.end + 0.05, 3),
                })
                idx += 1
    log.info("detect_filler_words: found %d fillers", len(fillers))
    return fillers


# ─── 2. CUT SEGMENTS (shared by filler-cutter & silence-remover) ─────────────

def run_cut_segments(video_path: str, segments_to_remove: list,
                     job_id: str | None = None):
    """
    Remove specified [{start, end}] segments and concatenate what remains.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== CUT-SEGMENTS %s  remove=%d ===", job_id, len(segments_to_remove))
    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        _, duration, _ = _video_info(video_path)

        # Merge overlapping removal segments
        segs = sorted(segments_to_remove, key=lambda s: float(s["start"]))
        merged = []
        for s in segs:
            s0, s1 = float(s["start"]), float(s["end"])
            if merged and s0 <= merged[-1][1] + 0.02:
                merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
            else:
                merged.append((s0, s1))

        # Build kept ranges (invert)
        kept, pos = [], 0.0
        for s0, s1 in merged:
            if s0 - pos > 0.05:
                kept.append((pos, s0))
            pos = s1
        if duration - pos > 0.05:
            kept.append((pos, duration))

        if not kept:
            raise ValueError("All content would be removed — nothing to keep.")

        jobs[job_id]["progress"] = 10
        tmp_dir = tempfile.mkdtemp(prefix="frameai_cs_")
        parts   = []
        for i, (s, e) in enumerate(kept):
            part = os.path.join(tmp_dir, f"part_{i:04d}.mp4")
            _ffmpeg("-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", video_path,
                    "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", part,
                    label=f"cs_part{i}")
            parts.append(part)
            jobs[job_id]["progress"] = 10 + int((i + 1) / len(kept) * 75)

        final = _concat_parts(parts, out_dir, job_id)
        time_removed = sum(e - s for s, e in merged)
        jobs[job_id].update({
            "status": "completed", "output": final, "progress": 100,
            "segments_removed": len(merged),
            "time_saved_sec": round(time_removed, 1),
        })
        log.info("=== CUT-SEGMENTS %s DONE: %.1fs removed ===", job_id, time_removed)
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ─── 3. SMART SILENCE REMOVER ────────────────────────────────────────────────

def run_silence_remover(video_path: str, min_silence_sec: float = 1.0,
                        job_id: str | None = None):
    """
    Transcribe with Whisper, find gaps > min_silence_sec between speech
    segments, and remove them.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== SILENCE-REMOVER %s  min=%.1fs ===", job_id, min_silence_sec)
    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        tmp   = tempfile.mkdtemp(prefix="frameai_sr_")
        audio = _extract_audio_wav(video_path, tmp)
        jobs[job_id]["progress"] = 12

        model    = _load_whisper()
        segs, _  = model.transcribe(audio, beam_size=5, vad_filter=True)
        speech   = [(s.start, s.end) for s in segs]
        jobs[job_id]["progress"] = 60

        _, duration, _ = _video_info(video_path)
        silences = []
        prev = 0.0
        for s0, s1 in speech:
            gap = s0 - prev
            if gap >= min_silence_sec:
                # Keep a 0.15s buffer each side so speech isn't clipped
                silences.append({"start": prev + 0.15, "end": s0 - 0.15})
            prev = s1
        if duration - prev >= min_silence_sec:
            silences.append({"start": prev + 0.15, "end": duration - 0.05})

        total_removed = sum(float(s["end"]) - float(s["start"])
                            for s in silences if float(s["end"]) > float(s["start"]))

        if not silences or total_removed < 0.1:
            # Nothing meaningful to cut — just copy
            final = os.path.join(out_dir, "final.mp4")
            shutil.copy(video_path, final)
            jobs[job_id].update({"status": "completed", "output": final,
                                  "progress": 100, "silences_removed": 0,
                                  "time_saved_sec": 0.0})
            return job_id

        jobs[job_id]["progress"] = 65
        # Delegate cut to run_cut_segments logic inline
        run_cut_segments.__wrapped__ if hasattr(run_cut_segments, "__wrapped__") \
            else None  # no-op guard
        # Actually call the helper directly via job reuse
        inner_job = run_cut_segments(video_path, silences)
        from pipeline.engine import get_job_status
        import time as _time
        while True:
            s = get_job_status(inner_job)
            if s["status"] in ("completed", "failed"):
                break
            _time.sleep(0.5)
        if get_job_status(inner_job)["status"] == "failed":
            raise RuntimeError(get_job_status(inner_job).get("error", "inner cut failed"))

        inner_out = get_job_status(inner_job)["output"]
        final = os.path.join(out_dir, "final.mp4")
        shutil.move(inner_out, final)

        jobs[job_id].update({
            "status": "completed", "output": final, "progress": 100,
            "silences_removed": len(silences),
            "time_saved_sec": round(total_removed, 1),
        })
        log.info("=== SILENCE-REMOVER %s DONE: %d silences, %.1fs ===",
                 job_id, len(silences), total_removed)
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ─── 4. SPEECH ENHANCER ──────────────────────────────────────────────────────

def run_speech_enhance(video_path: str, job_id: str | None = None):
    """
    Multi-stage FFmpeg audio restoration chain:
      highpass → lowpass → AI denoiser (anlmdn) → EQ presence boost →
      compressor → limiter.
    Video stream is stream-copied (no re-encode).
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== SPEECH-ENHANCE %s ===", job_id)
    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        final = os.path.join(out_dir, "final.mp4")
        jobs[job_id]["progress"] = 20

        af = (
            "highpass=f=80,"                             # remove low rumble
            "lowpass=f=12000,"                           # remove high hiss
            "anlmdn=s=7:p=0.002:r=0.002:m=15,"          # AI noise reduction
            "equalizer=f=250:t=h:width=200:g=-4,"       # cut mud
            "equalizer=f=3500:t=h:width=1500:g=4,"      # boost speech presence
            "equalizer=f=8000:t=h:width=3000:g=2,"      # add air/clarity
            "acompressor=threshold=-18dB:ratio=4:attack=5:release=100:makeup=4,"
            "alimiter=level_in=1:level_out=1:limit=0.95:attack=5:release=50"
        )
        _ffmpeg(
            "-i", video_path,
            "-c:v", "copy",
            "-af", af,
            "-c:a", "aac", "-b:a", "192k",
            final, label="speech_enhance"
        )
        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== SPEECH-ENHANCE %s DONE ===", job_id)
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ─── 5. BEAT SYNC DETECTOR ───────────────────────────────────────────────────

def detect_beats(video_path: str) -> dict:
    """
    librosa beat tracking on video's audio track.
    Returns: {bpm, beats: [seconds...], downbeats: [seconds...]}
    Immediate (no job_id).
    """
    import librosa, soundfile as sf

    tmp   = tempfile.mkdtemp(prefix="frameai_bt_")
    audio = _extract_audio_wav(video_path, tmp)

    y, sr = librosa.load(audio, sr=22050, mono=True)
    onset_env     = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats  = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times    = librosa.frames_to_time(beats, sr=sr).tolist()
    downbeat_times = beat_times[::4]          # every 4 beats = downbeats
    half_times     = beat_times[::2]          # half-notes

    log.info("detect_beats: %.1f BPM, %d beats", float(tempo), len(beat_times))
    return {
        "bpm":        round(float(tempo), 1),
        "beats":      [round(t, 3) for t in beat_times],
        "downbeats":  [round(t, 3) for t in downbeat_times],
        "half_beats": [round(t, 3) for t in half_times],
    }


# ─── 6. AUTO SPEED RAMP ──────────────────────────────────────────────────────

def run_auto_speedramp(video_path: str,
                       slow_factor: float = 0.5,
                       fast_factor: float = 3.0,
                       window_sec:  float = 2.0,
                       job_id: str | None = None):
    """
    Analyse per-second frame-difference scores.
    Top-30% motion windows → slow down (cinematic).
    Bottom-30% motion windows → speed up (boring parts).
    Middle 40% → keep 1×.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== AUTO-SPEEDRAMP %s  slow=%.2f fast=%.2f ===",
             job_id, slow_factor, fast_factor)
    try:
        import tempfile as _tf
        from concurrent.futures import ThreadPoolExecutor
        from pipeline.engine import _build_atempo

        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        orig_fps, duration, _ = _video_info(video_path)
        sample_fps = 4.0   # sample at 4fps for motion analysis

        tmp_frames = os.path.join(_STORAGE, job_id, "sample_frames")
        frames = extract_frames(video_path, tmp_frames, fps=sample_fps)
        jobs[job_id]["progress"] = 15

        # Compute frame-difference scores (motion intensity per window)
        scores = []
        prev_gray = None
        for fp in frames:
            gray = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray - prev_gray))
                scores.append(diff)
            prev_gray = gray

        if not scores:
            raise RuntimeError("Could not compute motion scores")

        # Group into windows of window_sec seconds
        frames_per_window = max(1, int(window_sec * sample_fps))
        windows = []
        for i in range(0, len(scores), frames_per_window):
            chunk = scores[i:i + frames_per_window]
            windows.append(float(np.mean(chunk)))

        p30 = float(np.percentile(windows, 30))
        p70 = float(np.percentile(windows, 70))
        jobs[job_id]["progress"] = 30

        # Build segments with assigned speed
        segments = []
        for i, w_score in enumerate(windows):
            t_start = i * window_sec
            t_end   = min((i + 1) * window_sec, duration)
            if t_end <= t_start:
                break
            factor = (slow_factor if w_score >= p70
                      else fast_factor if w_score <= p30
                      else 1.0)
            segments.append((t_start, t_end, factor))

        log.info("  %d windows: %d slow, %d fast, %d normal",
                 len(segments),
                 sum(1 for s in segments if s[2] == slow_factor),
                 sum(1 for s in segments if s[2] == fast_factor),
                 sum(1 for s in segments if s[2] == 1.0))

        # Encode each window at its assigned speed
        tmp_dir = _tf.mkdtemp(prefix="frameai_sr_")
        parts   = []
        for i, (s, e, factor) in enumerate(segments):
            part = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            pts  = 1.0 / factor
            vf   = f"setpts={pts:.6f}*PTS,scale=trunc(iw/2)*2:trunc(ih/2)*2"
            af   = _build_atempo(factor)
            _ffmpeg("-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", video_path,
                    "-vf", vf, "-af", af,
                    "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    part, label=f"ramp_seg{i}")
            parts.append(part)
            jobs[job_id]["progress"] = 30 + int((i + 1) / len(segments) * 55)

        final = _concat_parts(parts, out_dir, job_id)
        jobs[job_id].update({"status": "completed", "output": final, "progress": 100,
                              "windows": len(segments)})
        log.info("=== AUTO-SPEEDRAMP %s DONE ===", job_id)
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ─── 7. SMART THUMBNAIL ──────────────────────────────────────────────────────

def run_smart_thumbnail(video_path: str, job_id: str | None = None):
    """
    Score every second of video:
      sharpness (Laplacian variance) × brightness quality × person-detection bonus.
    Returns the highest-scoring frame as a JPEG.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== SMART-THUMBNAIL %s ===", job_id)
    try:
        from pipeline.engine import _load_yolo
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        tmp_frames = os.path.join(_STORAGE, job_id, "thumb_frames")
        frames = extract_frames(video_path, tmp_frames, fps=1)
        if not frames:
            raise RuntimeError("No frames extracted")
        jobs[job_id]["progress"] = 20

        yolo = _load_yolo()
        best_score, best_path = -1.0, frames[0]

        for i, fp in enumerate(frames):
            img  = cv2.imread(fp)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sharpness: high Laplacian variance = in-focus
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Brightness quality: ideal around 110-160
            bright = float(gray.mean())
            bright_q = max(0.0, 1.0 - abs(bright - 135) / 135)

            # Contrast: std dev of brightness
            contrast = float(gray.std()) / 128.0

            # Person detection bonus
            dets = yolo(fp, verbose=False)
            person_bonus = sum(
                2.0 for d in dets[0].boxes
                if dets[0].names[int(d.cls)] in ("person", "face") and float(d.conf) > 0.5
            )

            # Rule of thirds: prefer subjects off-center
            h, w = gray.shape
            thirds_score = 0.0
            for d in dets[0].boxes:
                cx = float(d.xyxy[0][0] + d.xyxy[0][2]) / 2 / w
                cy = float(d.xyxy[0][1] + d.xyxy[0][3]) / 2 / h
                thirds_score += 1.0 - min(abs(cx - 0.333), abs(cx - 0.667)) * 2

            score = (sharp ** 0.3) * (bright_q * 3) * (1 + contrast) \
                    * (1 + person_bonus) * (1 + thirds_score * 0.2)

            if score > best_score:
                best_score = score
                best_path  = fp
            jobs[job_id]["progress"] = 20 + int((i + 1) / len(frames) * 70)

        # Copy best frame as the output JPEG
        thumbnail = os.path.join(out_dir, "thumbnail.jpg")
        shutil.copy(best_path, thumbnail)

        # Get frame timestamp
        frame_idx = frames.index(best_path)
        timestamp = frame_idx  # 1fps → frame index ≈ seconds

        jobs[job_id].update({
            "status": "completed", "output": thumbnail,
            "progress": 100, "score": round(best_score, 2),
            "timestamp_sec": timestamp,
            "is_image": True,
        })
        log.info("=== SMART-THUMBNAIL %s DONE: frame@%ds score=%.1f ===",
                 job_id, timestamp, best_score)
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ─── 8. IN-VIDEO OCR (Text Search Index) ─────────────────────────────────────

_ocr_reader      = None
_ocr_reader_lock = threading.Lock()

def _load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_reader_lock:
            if _ocr_reader is None:
                import easyocr
                log.info("Loading EasyOCR (English) …")
                _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                log.info("EasyOCR loaded.")
    return _ocr_reader


def run_video_ocr(video_path: str, job_id: str | None = None):
    """
    Sample 1 frame every 3 seconds, run EasyOCR, build a searchable text index.
    Result stored as JSON in output dir; also available via /ocr-results/{job_id}.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== VIDEO-OCR %s ===", job_id)
    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        tmp_frames = os.path.join(_STORAGE, job_id, "ocr_frames")
        # 1 frame every 3 seconds (0.33 fps)
        frames = extract_frames(video_path, tmp_frames, fps=0.33)
        if not frames:
            raise RuntimeError("No frames extracted")
        jobs[job_id]["progress"] = 10

        reader  = _load_ocr()
        index   = []   # [{timestamp, text, confidence, bbox}]
        seen_texts = set()

        for i, fp in enumerate(frames):
            timestamp = i * 3.0  # 1 frame / 3s
            results   = reader.readtext(fp, detail=1, paragraph=False)
            for (bbox, text, conf) in results:
                text_clean = text.strip()
                if conf >= 0.6 and len(text_clean) >= 2:
                    key = (text_clean.lower(), int(timestamp / 5) * 5)  # dedupe within 5s
                    if key not in seen_texts:
                        seen_texts.add(key)
                        index.append({
                            "timestamp": round(timestamp, 1),
                            "text":      text_clean,
                            "conf":      round(conf, 2),
                        })
            jobs[job_id]["progress"] = 10 + int((i + 1) / len(frames) * 85)
            log.info("  OCR frame %d/%d: %d entries so far", i+1, len(frames), len(index))

        # Sort by timestamp
        index.sort(key=lambda x: x["timestamp"])

        idx_path = os.path.join(out_dir, "ocr_index.json")
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        jobs[job_id].update({
            "status":   "completed",
            "output":   idx_path,
            "progress": 100,
            "is_json":  True,
            "total_entries": len(index),
            "ocr_index": index,   # also embed in status for immediate access
        })
        log.info("=== VIDEO-OCR %s DONE: %d text entries ===", job_id, len(index))
    except Exception as exc:
        import traceback
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ── FACE BLUR & ANONYMIZER ────────────────────────────────────────────────────

def run_face_blur(video_path: str, blur_strength: int = 45, job_id: str = None):
    """
    Detect faces in every frame using OpenCV Haar cascade and apply Gaussian blur.
    blur_strength: odd integer, higher = more blurred (default 45).
    """
    import tempfile, subprocess
    if not job_id:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== FACE-BLUR %s ===", job_id)

    try:
        import cv2 as _cv2
        # Use OpenCV's built-in frontal face cascade (no extra download needed)
        cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        face_cascade = _cv2.CascadeClassifier(cascade_path)

        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = os.path.join(_STORAGE, job_id, "frames")
        edited_dir = os.path.join(_STORAGE, job_id, "edited")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(edited_dir, exist_ok=True)

        # Get video info
        cap = _cv2.VideoCapture(video_path)
        fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Extract all frames at original FPS
        jobs[job_id]["progress"] = 5
        frame_paths = extract_frames(video_path, frames_dir, fps=fps)
        if not frame_paths:
            raise RuntimeError("No frames extracted")

        n = len(frame_paths)
        ks = blur_strength | 1  # ensure odd
        log.info("Face-blur: %d frames, kernel=%d", n, ks)

        for i, fp in enumerate(frame_paths):
            frame = _cv2.imread(fp)
            if frame is None:
                continue
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                pad = int(max(w, h) * 0.15)
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                roi = frame[y1:y2, x1:x2]
                blurred = _cv2.GaussianBlur(roi, (ks, ks), 0)
                frame[y1:y2, x1:x2] = blurred

            out_fp = os.path.join(edited_dir, f"f_{i:05d}.jpg")
            _cv2.imwrite(out_fp, frame, [_cv2.IMWRITE_JPEG_QUALITY, 92])
            jobs[job_id]["progress"] = 5 + int((i + 1) / n * 75)

        # Encode edited frames → video
        jobs[job_id]["progress"] = 82
        seg = os.path.join(tempfile.mkdtemp(prefix="frameai_fblur_"), "seg.mp4")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(edited_dir, "f_%05d.jpg"),
            "-vf", f"fps={fps:.3f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", seg
        ], check=True)

        # Mux original audio
        from pipeline.engine import _add_audio
        final = os.path.join(out_dir, "final.mp4")
        _add_audio(seg, video_path, final)
        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== FACE-BLUR %s DONE ===", job_id)
    except Exception as exc:
        import traceback
        log.error("FACE-BLUR failed: %s", exc)
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id


# ── SCENE DETECTION ───────────────────────────────────────────────────────────

def detect_scenes(video_path: str, threshold: float = 27.0) -> dict:
    """
    Detect scene changes using PySceneDetect ContentDetector.
    Returns list of {scene, start, end, duration} dicts.
    """
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(video_path, ContentDetector(threshold=threshold))
        result = []
        for i, (start, end) in enumerate(scenes):
            result.append({
                "scene": i + 1,
                "start": round(start.get_seconds(), 3),
                "end":   round(end.get_seconds(), 3),
                "duration": round((end - start).get_seconds(), 3),
            })
        return {"scenes": result, "count": len(result)}
    except Exception as exc:
        return {"scenes": [], "count": 0, "error": str(exc)}


# ── AI AUDIO DENOISE ─────────────────────────────────────────────────────────

def run_ai_denoise(video_path: str, strength: float = 0.75, job_id: str = None):
    """
    Remove background noise from video audio using noisereduce.
    strength: 0.0–1.0 (how aggressively to suppress noise).
    """
    import subprocess, tempfile
    if not job_id:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== AI-DENOISE %s strength=%.2f ===", job_id, strength)

    try:
        import soundfile as sf
        import noisereduce as nr
        import numpy as _np

        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="frameai_dnoise_")

        # Extract audio as WAV
        jobs[job_id]["progress"] = 10
        wav_path = os.path.join(tmp_dir, "audio.wav")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            wav_path
        ], check=True)

        # Load + denoise
        jobs[job_id]["progress"] = 25
        data, rate = sf.read(wav_path)
        jobs[job_id]["progress"] = 35
        # Use first 0.5s as noise profile if available
        noise_sample = data[:int(rate * 0.5)] if len(data) > rate * 0.5 else data[:1000]
        denoised = nr.reduce_noise(
            y=data, sr=rate,
            y_noise=noise_sample,
            prop_decrease=float(strength),
            stationary=False,
        )
        jobs[job_id]["progress"] = 70
        clean_wav = os.path.join(tmp_dir, "clean.wav")
        sf.write(clean_wav, denoised, rate)

        # Mux clean audio with original video
        jobs[job_id]["progress"] = 80
        final = os.path.join(out_dir, "final.mp4")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-i", clean_wav,
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", final
        ], check=True)

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== AI-DENOISE %s DONE ===", job_id)
    except Exception as exc:
        import traceback
        log.error("AI-DENOISE failed: %s", exc)
        jobs[job_id].update({"status": "failed", "error": str(exc),
                              "trace": traceback.format_exc()})
    return job_id
