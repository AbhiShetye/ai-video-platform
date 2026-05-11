from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, os, uuid, threading
from dotenv import load_dotenv

load_dotenv()

from pipeline.engine import (run_pipeline, run_magic_erase, run_mute_audio,
                             run_quick_edit, get_job_status, jobs)
from pipeline.ai_tools import run_auto_captions, run_bg_remove, run_stabilize
from pipeline.novel_ai import (
    detect_filler_words, run_cut_segments, run_silence_remover,
    run_speech_enhance, detect_beats, run_auto_speedramp,
    run_smart_thumbnail, run_video_ocr,
    run_face_blur, detect_scenes, run_ai_denoise,
)
from pipeline.image_tools import (
    run_image_bg_remove, run_image_filter, run_image_crop,
    run_image_upscale, run_image_text, run_image_object_remove,
    detect_image_objects, IMG_UPLOAD_DIR,
)
from routes.ai_studio import router as ai_studio_router

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND   = os.path.join(BASE_DIR, "..", "frontend", "index.html")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


import time as _time

_model_status: dict = {}   # tracks which models loaded OK / failed

@app.on_event("startup")
async def _warmup():
    """Pre-load heavy AI models in background + start periodic job-cleanup."""
    import threading

    def _load():
        steps = [
            ("yolo",    "pipeline.engine",   "_load_yolo",    []),
            ("whisper", "pipeline.ai_tools",  "_load_whisper", ["small"]),
            ("rembg",   "pipeline.ai_tools",  "_load_rembg",   []),
        ]
        for name, mod, fn, args in steps:
            try:
                import importlib
                m = importlib.import_module(mod)
                getattr(m, fn)(*args)
                _model_status[name] = "ready"
            except Exception as exc:
                _model_status[name] = f"error: {exc}"
    threading.Thread(target=_load, daemon=True, name="model-warmup").start()

    def _cleanup_jobs():
        """Remove jobs older than 2 hours to prevent memory leaks."""
        while True:
            _time.sleep(3600)
            cutoff = _time.time() - 7200
            with _jobs_lock:
                stale = [k for k, v in jobs.items()
                         if isinstance(v, dict) and v.get("_ts", 0) < cutoff]
                for k in stale:
                    del jobs[k]
    from pipeline.engine import _jobs_lock
    threading.Thread(target=_cleanup_jobs, daemon=True, name="job-cleanup").start()

    def _cleanup_files():
        """
        Auto-delete uploaded videos and processed outputs older than 24 hours.
        Runs every hour. Uses file mtime so active/recently-edited files are kept.
        """
        import shutil, logging
        _log = logging.getLogger("cleanup")
        while True:
            _time.sleep(3600)   # check every hour
            cutoff = _time.time() - 86400  # 24 hours ago
            deleted_files = 0
            deleted_jobs  = 0

            # ── uploads directory ─────────────────────────────────────
            for fname in os.listdir(UPLOAD_DIR):
                fpath = os.path.join(UPLOAD_DIR, fname)
                try:
                    if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        deleted_files += 1
                except Exception:
                    pass

            # ── img_uploads directory ──────────────────────────────────
            from pipeline.image_tools import IMG_UPLOAD_DIR
            if os.path.isdir(IMG_UPLOAD_DIR):
                for fname in os.listdir(IMG_UPLOAD_DIR):
                    fpath = os.path.join(IMG_UPLOAD_DIR, fname)
                    try:
                        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            deleted_files += 1
                    except Exception:
                        pass

            # ── storage directory (job outputs) ───────────────────────
            from pipeline.engine import _STORAGE
            if os.path.isdir(_STORAGE):
                for job_dir in os.listdir(_STORAGE):
                    job_path = os.path.join(_STORAGE, job_dir)
                    try:
                        if os.path.isdir(job_path) and os.path.getmtime(job_path) < cutoff:
                            shutil.rmtree(job_path, ignore_errors=True)
                            deleted_jobs += 1
                    except Exception:
                        pass

            if deleted_files or deleted_jobs:
                _log.info("Auto-cleanup: deleted %d upload file(s), %d job dir(s)",
                          deleted_files, deleted_jobs)

    threading.Thread(target=_cleanup_files, daemon=True, name="file-cleanup").start()


@app.get("/health")
def health():
    """Health + model-status check for frontend connection verification."""
    return {
        "status": "ok",
        "models": _model_status,
        "jobs": len(jobs),
    }


@app.get("/storage-stats")
def storage_stats():
    """Return per-file expiry info and total disk usage for the cleanup banner."""
    import shutil as _shutil
    now = _time.time()
    TTL = 86400  # 24 hours

    files = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        mtime = os.path.getmtime(fpath)
        expires_in = max(0, int(TTL - (now - mtime)))
        files.append({
            "name": fname,
            "size_mb": round(os.path.getsize(fpath) / 1048576, 1),
            "expires_in_sec": expires_in,
            "expires_in_hr": round(expires_in / 3600, 1),
        })

    # Total storage used (uploads + job outputs)
    from pipeline.engine import _STORAGE
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(UPLOAD_DIR)
        for f in filenames
    ) + sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(_STORAGE)
        for f in filenames
    )
    return {
        "files": sorted(files, key=lambda x: x["expires_in_sec"]),
        "total_mb": round(total_bytes / 1048576, 1),
        "ttl_hours": 24,
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_studio_router)

_ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@app.get("/")
def root():
    return FileResponse(FRONTEND, media_type="text/html")


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _ALLOWED_VIDEO_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")
    dest = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"success": True, "filename": safe_name}


class DetectRequest(BaseModel):
    filename: str
    full_scan: bool = False

@app.post("/detect")
def detect(req: DetectRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"objects": [], "error": "File not found"}
    from vision.detection import detect_first_frame
    return {"objects": detect_first_frame(path, full_scan=req.full_scan)}


class EditRequest(BaseModel):
    filename: str
    start_time: float = 0
    end_time: float = None
    bbox: list = None

@app.post("/process-edit")
def process_edit(req: EditRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}

    cmd = {
        "start_time": req.start_time,
        "end_time": req.end_time,
        "bbox": req.bbox or [0, 0, 100, 100],
    }

    threading.Thread(target=run_pipeline, args=(path, cmd, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}



class MagicEraseRequest(BaseModel):
    filename: str
    bboxes: list          # [[x1,y1,x2,y2], ...] — one per selected object
    start_time: float = 0
    end_time: float = None

@app.post("/magic-erase")
def magic_erase(req: MagicEraseRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    if not req.bboxes:
        return {"error": "No objects selected"}

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    cmd = {"bboxes": req.bboxes, "start_time": req.start_time, "end_time": req.end_time}
    threading.Thread(target=run_magic_erase, args=(path, cmd, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


class MuteRequest(BaseModel):
    filename: str
    segments: list        # [{"start": float, "end": float}, ...]

@app.post("/mute-audio")
def mute_audio(req: MuteRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    if not req.segments:
        return {"error": "No segments provided"}

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_mute_audio, args=(path, req.segments, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


class QuickEditRequest(BaseModel):
    filename: str
    operation: dict   # {"type": "trim"|"speed"|"rotate"|"filter", ...params}

@app.post("/quick-edit")
def quick_edit(req: QuickEditRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    op_type = req.operation.get("type", "")
    _ALLOWED = {"trim","speed","rotate","filter","reverse","fade","aspect","compress",
                "crop","text","volume","sharpen","denoise","vignette","loop","gif"}
    if op_type not in _ALLOWED:
        return {"error": f"Unknown operation '{op_type}'"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_quick_edit,
                     args=(path, req.operation, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def status(job_id: str):
    return get_job_status(job_id)


@app.get("/download/{job_id}")
def download(job_id: str):
    s = get_job_status(job_id)
    if s.get("status") == "completed":
        output = s["output"]
        if output.endswith(".gif"):
            return FileResponse(output, media_type="image/gif", filename="export.gif")
        if output.endswith(".mp3"):
            return FileResponse(output, media_type="audio/mpeg", filename="audio.mp3")
        return FileResponse(output, media_type="video/mp4", filename="edited.mp4")
    return {"error": "Not ready"}


# ── AI TOOLS ──────────────────────────────────────────────────────────────────

class CaptionsRequest(BaseModel):
    filename: str
    language: str  = "auto"   # ISO 639-1 or "auto"
    burn:     bool = True     # burn subtitles into video
    style:    str  = "white"  # white | yellow | large

@app.post("/auto-captions")
def auto_captions(req: CaptionsRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(
        target=run_auto_captions,
        args=(path, req.language, req.burn, req.style, job_id),
        daemon=True
    ).start()
    return {"job_id": job_id, "status": "processing"}


class BgRemoveRequest(BaseModel):
    filename: str
    bg_color: str = "black"   # black | white | green | blue | red

@app.post("/remove-bg")
def remove_bg(req: BgRemoveRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(
        target=run_bg_remove,
        args=(path, req.bg_color, job_id),
        daemon=True
    ).start()
    return {"job_id": job_id, "status": "processing"}


class StabilizeRequest(BaseModel):
    filename:   str
    shakiness:  int = 5    # 1–10
    smoothing:  int = 30   # 5–100

@app.post("/stabilize")
def stabilize(req: StabilizeRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(
        target=run_stabilize,
        args=(path, req.shakiness, req.smoothing, job_id),
        daemon=True
    ).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/download-srt/{job_id}")
def download_srt(job_id: str):
    s = get_job_status(job_id)
    srt = s.get("srt")
    if s.get("status") == "completed" and srt and os.path.exists(srt):
        return FileResponse(srt, media_type="text/plain", filename="captions.srt")
    return {"error": "SRT not available"}


# ── NOVEL AI ENDPOINTS ────────────────────────────────────────────────────────

class FilenameRequest(BaseModel):
    filename: str

@app.post("/detect-fillers")
def detect_fillers_ep(req: FilenameRequest):
    """Immediate: returns list of filler words with timestamps (no job)."""
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    try:
        fillers = detect_filler_words(path)
        return {"fillers": fillers, "count": len(fillers)}
    except Exception as e:
        return {"error": str(e)}


class CutSegmentsRequest(BaseModel):
    filename: str
    segments: list  # [{start, end}, ...]

@app.post("/cut-segments")
def cut_segments_ep(req: CutSegmentsRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    if not req.segments:
        return {"error": "No segments provided"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_cut_segments,
                     args=(path, req.segments, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


class SilenceRequest(BaseModel):
    filename: str
    min_silence_sec: float = 1.0

@app.post("/remove-silences")
def remove_silences_ep(req: SilenceRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_silence_remover,
                     args=(path, req.min_silence_sec, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/enhance-speech")
def enhance_speech_ep(req: FilenameRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_speech_enhance,
                     args=(path, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/detect-beats/{filename}")
def detect_beats_ep(filename: str):
    """Immediate: returns BPM + beat timestamps."""
    safe = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    try:
        return detect_beats(path)
    except Exception as e:
        return {"error": str(e)}


class SpeedRampRequest(BaseModel):
    filename: str
    slow_factor:  float = 0.5
    fast_factor:  float = 3.0
    window_sec:   float = 2.0

@app.post("/auto-speedramp")
def auto_speedramp_ep(req: SpeedRampRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_auto_speedramp,
                     args=(path, req.slow_factor, req.fast_factor,
                           req.window_sec, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/smart-thumbnail")
def smart_thumbnail_ep(req: FilenameRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_smart_thumbnail,
                     args=(path, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/download-thumbnail/{job_id}")
def download_thumbnail(job_id: str):
    s = get_job_status(job_id)
    if s.get("status") == "completed" and s.get("is_image"):
        return FileResponse(s["output"], media_type="image/jpeg",
                            filename="thumbnail.jpg")
    return {"error": "Not ready"}


@app.post("/video-ocr")
def video_ocr_ep(req: FilenameRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_video_ocr,
                     args=(path, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/ocr-results/{job_id}")
def ocr_results(job_id: str):
    s = get_job_status(job_id)
    if s.get("status") == "completed":
        return {"entries": s.get("ocr_index", []),
                "total": s.get("total_entries", 0)}
    return {"status": s.get("status", "unknown")}


@app.get("/extract-audio/{filename}")
def extract_audio(filename: str):
    import tempfile, subprocess as _sp
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    probe = _sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    if "audio" not in probe.stdout:
        raise HTTPException(status_code=422, detail="This video has no audio stream")
    try:
        tmp = tempfile.mktemp(suffix=".mp3", dir=UPLOAD_DIR)
        _sp.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp],
            check=True, capture_output=True
        )
        return FileResponse(tmp, media_type="audio/mpeg", filename="audio.mp3")
    except _sp.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="Audio extraction failed")


# ══════════════════════════════════════════════════════════════════════
# EXCLUSIVE AI VIDEO FEATURES
# ══════════════════════════════════════════════════════════════════════

class FaceBlurRequest(BaseModel):
    filename: str
    blur_strength: int = 45

@app.post("/face-blur")
def face_blur_ep(req: FaceBlurRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_face_blur,
                     args=(path, req.blur_strength, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/detect-scenes/{filename}")
def detect_scenes_ep(filename: str, threshold: float = 27.0):
    safe = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    try:
        return detect_scenes(path, threshold=threshold)
    except Exception as e:
        return {"scenes": [], "count": 0, "error": str(e)}


class DenoiseRequest(BaseModel):
    filename: str
    strength: float = 0.75

@app.post("/ai-denoise")
def ai_denoise_ep(req: DenoiseRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    strength = max(0.0, min(1.0, req.strength))
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    threading.Thread(target=run_ai_denoise,
                     args=(path, strength, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


# ══════════════════════════════════════════════════════════════════════
# PHOTO EDITOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename or "image")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _ALLOWED_IMG_EXT:
        raise HTTPException(400, f"Unsupported image type '{ext}'")
    dest = os.path.join(IMG_UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    from PIL import Image as _PILImage
    try:
        with _PILImage.open(dest) as img:
            w, h = img.size
    except Exception:
        w = h = 0
    return {"success": True, "filename": safe_name, "width": w, "height": h}


@app.get("/detect-image-objects/{filename}")
def detect_img_objects(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    try:
        return {"objects": detect_image_objects(path)}
    except Exception as e:
        return {"objects": [], "error": str(e)}


class ImgBgRemoveRequest(BaseModel):
    filename: str
    bg: str = "transparent"

@app.post("/image-bg-remove")
def img_bg_remove(req: ImgBgRemoveRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    jid = run_image_bg_remove(path, bg=req.bg)
    return {"job_id": jid, "status": "processing"}


class ImgFilterRequest(BaseModel):
    filename: str
    brightness: float = 1.0
    contrast:   float = 1.0
    saturation: float = 1.0
    sharpness:  float = 1.0
    blur:       float = 0.0

@app.post("/image-filter")
def img_filter(req: ImgFilterRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    jid = run_image_filter(path, {
        "brightness": req.brightness, "contrast":   req.contrast,
        "saturation": req.saturation, "sharpness":  req.sharpness,
        "blur":       req.blur,
    })
    return {"job_id": jid, "status": "processing"}


class ImgCropRequest(BaseModel):
    filename: str
    x: int = 0; y: int = 0
    w: int; h: int
    out_w: int = 0; out_h: int = 0

@app.post("/image-crop")
def img_crop(req: ImgCropRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    jid = run_image_crop(path, req.x, req.y, req.w, req.h, req.out_w, req.out_h)
    return {"job_id": jid, "status": "processing"}


class ImgUpscaleRequest(BaseModel):
    filename: str
    scale: int = 2

@app.post("/image-upscale")
def img_upscale(req: ImgUpscaleRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    if req.scale not in (2, 3, 4):
        return {"error": "Scale must be 2, 3, or 4"}
    jid = run_image_upscale(path, scale=req.scale)
    return {"job_id": jid, "status": "processing"}


class ImgTextRequest(BaseModel):
    filename: str
    text: str
    position: str = "bc"
    size: int = 48
    color: str = "white"

@app.post("/image-text")
def img_text(req: ImgTextRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    if not req.text.strip():
        return {"error": "Text cannot be empty"}
    jid = run_image_text(path, req.text, req.position, req.size, req.color)
    return {"job_id": jid, "status": "processing"}


class ImgObjectRemoveRequest(BaseModel):
    filename: str
    bbox: list

@app.post("/image-object-remove")
def img_object_remove(req: ImgObjectRemoveRequest):
    safe = os.path.basename(req.filename)
    path = os.path.join(IMG_UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return {"error": "Image not found"}
    if not req.bbox or len(req.bbox) != 4:
        return {"error": "bbox must be [x1,y1,x2,y2]"}
    jid = run_image_object_remove(path, req.bbox)
    return {"job_id": jid, "status": "processing"}


@app.get("/image-result/{job_id}")
def image_result(job_id: str):
    s = get_job_status(job_id)
    if s.get("status") == "completed" and s.get("is_image"):
        out = s["output"]
        ext = os.path.splitext(out)[1].lower()
        media = "image/png" if ext == ".png" else "image/jpeg"
        fname = "result.png" if ext == ".png" else "result.jpg"
        return FileResponse(out, media_type=media, filename=fname)
    if s.get("status") == "failed":
        raise HTTPException(500, s.get("error", "Processing failed"))
    raise HTTPException(425, "Not ready yet")


@app.post("/promote-result/{job_id}")
def promote_result(job_id: str):
    """Copy a completed job's output into uploads/ so it can be used as a new source."""
    s = get_job_status(job_id)
    if s.get("status") != "completed":
        raise HTTPException(400, "Job not completed")
    out = s.get("output", "")
    if not out or not os.path.exists(out):
        raise HTTPException(404, "Output file not found")
    ext = os.path.splitext(out)[1].lower() or ".mp4"
    new_name = "edited_" + job_id[:8] + ext
    dest = os.path.join(UPLOAD_DIR, new_name)
    shutil.copy2(out, dest)
    return {"success": True, "filename": new_name}
