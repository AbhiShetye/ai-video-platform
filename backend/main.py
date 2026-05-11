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
)
from routes.ai_studio import router as ai_studio_router

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND   = os.path.join(BASE_DIR, "..", "frontend", "index.html")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


@app.on_event("startup")
async def _warmup():
    """Pre-load heavy AI models in background so first user request is instant."""
    import asyncio, threading
    def _load():
        try:
            from pipeline.engine import _load_yolo
            _load_yolo()
            from pipeline.ai_tools import _load_whisper, _load_rembg
            _load_whisper()
            _load_rembg()
        except Exception:
            pass  # non-fatal — models load on demand if this fails
    threading.Thread(target=_load, daemon=True, name="model-warmup").start()


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
    import tempfile
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    tmp = tempfile.mktemp(suffix=".mp3", dir=UPLOAD_DIR)
    import subprocess
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp],
        check=True
    )
    return FileResponse(tmp, media_type="audio/mpeg", filename="audio.mp3")
