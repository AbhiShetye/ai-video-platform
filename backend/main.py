from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, os, uuid, threading
from dotenv import load_dotenv

load_dotenv()

from pipeline.engine import run_pipeline, get_job_status, jobs
from routes.ai_studio import router as ai_studio_router

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND   = os.path.join(BASE_DIR, "..", "frontend", "index.html")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

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

@app.post("/detect")
def detect(req: DetectRequest):
    safe_name = os.path.basename(req.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        return {"objects": [], "error": "File not found"}
    from vision.detection import detect_first_frame
    return {"objects": detect_first_frame(path)}


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



@app.get("/status/{job_id}")
def status(job_id: str):
    return get_job_status(job_id)


@app.get("/download/{job_id}")
def download(job_id: str):
    s = get_job_status(job_id)
    if s.get("status") == "completed":
        return FileResponse(s["output"], media_type="video/mp4", filename="edited.mp4")
    return {"error": "Not ready"}
