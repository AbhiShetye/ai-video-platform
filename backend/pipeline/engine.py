import os
import uuid
import logging
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np

from vision.frames import extract_frames

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

jobs: dict = {}
_jobs_lock  = threading.Lock()

EXTRACT_FPS = 1      # 1 fps — LaMa takes ~4s/frame; 1fps = manageable
CRF         = 18

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND    = os.path.dirname(_BASE_DIR)
_STORAGE    = os.path.join(_BACKEND, "storage")
_YOLO_PT    = os.path.join(_BACKEND, "yolov8n.pt")


# ── helpers ───────────────────────────────────────────────────────────────────

def _video_info(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = n / fps if fps else 0.0
    rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    cap.release()
    return float(fps), float(dur), rot


def _ffmpeg(*args, label="ffmpeg"):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.info("[%s] %s", label, " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg [{label}] failed:\n{r.stderr[-2000:]}")
    return r


def _iou(a, b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _detect_in_frame(frame_path: str, hint_bbox: list, model) -> list | None:
    """
    Run YOLO on one frame. Return the bbox of the detection that best overlaps
    hint_bbox. Returns None if nothing close enough is found.
    """
    results = model(frame_path, verbose=False)
    best_bbox, best_score = None, 0.0
    for b in results[0].boxes:
        coords = [int(v) for v in b.xyxy[0].tolist()]
        score = _iou(coords, hint_bbox)
        if score > best_score:
            best_score = score
            best_bbox = coords
    # Accept if there's any overlap (IoU > 0.05) — objects shift with camera
    if best_bbox and best_score > 0.05:
        return best_bbox
    # Also accept if no IoU match but there's a detection very close to hint centre
    if results[0].boxes:
        hcx = (hint_bbox[0]+hint_bbox[2])/2
        hcy = (hint_bbox[1]+hint_bbox[3])/2
        for b in results[0].boxes:
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if abs(cx-hcx) < 300 and abs(cy-hcy) < 300:
                return [x1,y1,x2,y2]
    return None


def _bbox_mask(frame_path: str, bbox: list) -> np.ndarray:
    """Solid rectangular mask covering the full bbox. Best input for LaMa."""
    frame = cv2.imread(frame_path)
    h, w  = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Add 8px padding so LaMa has no leftover edge pixels to confuse it
    pad = 8
    x1=max(0,x1-pad); y1=max(0,y1-pad)
    x2=min(w,x2+pad); y2=min(h,y2+pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


_lama_lock = threading.Lock()


def _lama_inpaint(frame_path: str, mask: np.ndarray, output_path: str):
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    global _lama_model
    if _lama_model is None:
        with _lama_lock:
            if _lama_model is None:
                _lama_model = SimpleLama()

    frame = cv2.imread(frame_path)
    h, w  = frame.shape[:2]

    # Full resolution — portrait phone videos are 576px wide, LaMa handles it fine
    scale = min(1.0, 1024 / max(w, h))
    nw, nh = int(w*scale)&~1, int(h*scale)&~1
    fr_s = cv2.resize(frame, (nw,nh), interpolation=cv2.INTER_AREA)
    mk_s = cv2.resize(mask,  (nw,nh), interpolation=cv2.INTER_NEAREST)

    img_pil  = Image.fromarray(cv2.cvtColor(fr_s, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mk_s)
    result_pil = _lama_model(img_pil, mask_pil)

    lama_full = cv2.resize(
        cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR),
        (w, h), interpolation=cv2.INTER_LANCZOS4
    )

    # Composite: paste LaMa fill onto original frame using the mask.
    # Feather only the outer 4px of the mask edge so the boundary blends
    # smoothly without pulling any original-object colors back in.
    mask_f32 = (mask > 127).astype(np.float32)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    alpha    = cv2.GaussianBlur(cv2.erode(mask_f32, kernel, iterations=1),
                                (9, 9), 0)
    result = frame.copy().astype(np.float32)
    lama_f = lama_full.astype(np.float32)
    for c in range(3):
        result[:, :, c] = frame[:, :, c] * (1.0 - alpha) + lama_f[:, :, c] * alpha

    cv2.imwrite(output_path, result.astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 97])
    return output_path


_lama_model = None


# ── main entry point ──────────────────────────────────────────────────────────

def run_pipeline(video_path: str, command: dict, job_id: str | None = None):
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== JOB %s START ===", job_id)

    try:
        base       = _STORAGE
        frames_dir = os.path.join(base, job_id, "frames")
        edited_dir = os.path.join(base, job_id, "edited")
        out_dir    = os.path.join(base, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        orig_fps, duration, rotation = _video_info(video_path)
        log.info("video: %.2f fps  %.2f s  rotation=%d°", orig_fps, duration, rotation)

        start_sec = float(command.get("start_time") or 0)
        end_sec   = float(command.get("end_time")   or duration)
        end_sec   = min(end_sec, duration)
        hint_bbox = command.get("bbox") or [0, 0, 100, 100]

        if start_sec >= end_sec:
            raise ValueError(f"Invalid range: {start_sec}–{end_sec}s")

        # ── STEP 1: Extract frames ────────────────────────────────────────
        jobs[job_id]["progress"] = 8
        log.info("Step 1: extracting frames %.1fs–%.1fs at %d fps",
                 start_sec, end_sec, EXTRACT_FPS)
        frame_paths = extract_frames(video_path, frames_dir,
                                     fps=EXTRACT_FPS,
                                     start_sec=start_sec, end_sec=end_sec)
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {start_sec}s to {end_sec}s.")
        log.info("  extracted %d frames", len(frame_paths))

        # ── STEP 2: Per-frame YOLO detection for accurate bbox ────────────
        jobs[job_id]["progress"] = 15
        log.info("Step 2: per-frame object detection (tracking)")
        from ultralytics import YOLO
        yolo = YOLO(_YOLO_PT)
        per_frame_bboxes = []
        for fp in frame_paths:
            detected = _detect_in_frame(fp, hint_bbox, yolo)
            per_frame_bboxes.append(detected or hint_bbox)
            log.info("  %s → %s", os.path.basename(fp),
                     detected if detected else "fallback to hint")
        del yolo  # free memory before LaMa

        # ── STEP 3: Full bbox masks per frame (LaMa works best with solid masks)
        jobs[job_id]["progress"] = 22
        log.info("Step 3: generating masks")
        masks = [_bbox_mask(fp, bb)
                 for fp, bb in zip(frame_paths, per_frame_bboxes)]

        # ── STEP 4: LaMa inpainting ───────────────────────────────────────
        jobs[job_id]["progress"] = 30
        log.info("Step 4: LaMa inpainting %d frames", len(frame_paths))
        os.makedirs(edited_dir, exist_ok=True)
        edited_paths = []
        for i, (fp, mask) in enumerate(zip(frame_paths, masks)):
            out_path = os.path.join(edited_dir, f"edited_{i:04d}.jpg")
            _lama_inpaint(fp, mask, out_path)
            edited_paths.append(out_path)
            jobs[job_id]["progress"] = 30 + int((i+1)/len(frame_paths)*42)
            log.info("  frame %d/%d done", i+1, len(frame_paths))
        log.info("  inpainting complete")

        # ── STEP 5: Encode edited segment ────────────────────────────────
        jobs[job_id]["progress"] = 73
        tmp_dir = tempfile.mkdtemp(prefix="frameai_")
        seg_b   = os.path.join(tmp_dir, "seg_b.mp4")
        log.info("Step 5: encoding edited segment")
        _ffmpeg(
            "-framerate", str(EXTRACT_FPS),
            "-i", os.path.join(edited_dir, "edited_%04d.jpg"),
            "-vf", f"fps={orig_fps:.3f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            seg_b, label="encode_seg"
        )

        # ── STEP 6: Assemble full video ───────────────────────────────────
        jobs[job_id]["progress"] = 83
        log.info("Step 6: assembling full video")
        no_audio = os.path.join(tmp_dir, "no_audio.mp4")
        _assemble(video_path, seg_b, no_audio, start_sec, end_sec, duration, tmp_dir, rotation)

        # ── STEP 7: Add audio ─────────────────────────────────────────────
        jobs[job_id]["progress"] = 93
        final = os.path.join(out_dir, "final.mp4")
        log.info("Step 7: adding audio → %s", final)
        _add_audio(no_audio, video_path, final)

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== JOB %s DONE → %s ===", job_id, final)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== JOB %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id


# ── assembly / audio ──────────────────────────────────────────────────────────

def _assemble(original, seg_b, output, start_sec, end_sec, duration, tmp_dir, rotation=0):
    # FFmpeg auto-applies rotation metadata when re-encoding, so A/C segments
    # will match the corrected B segment automatically.
    parts = []
    if start_sec > 0.05:
        seg_a = os.path.join(tmp_dir, "seg_a.mp4")
        _ffmpeg("-ss","0","-to",f"{start_sec:.3f}","-i",original,
                "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v","libx264","-crf",str(CRF),"-pix_fmt","yuv420p","-an",
                seg_a, label="cut_a")
        parts.append(seg_a)
    parts.append(seg_b)
    if end_sec < duration - 0.05:
        seg_c = os.path.join(tmp_dir, "seg_c.mp4")
        _ffmpeg("-ss",f"{end_sec:.3f}","-i",original,
                "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v","libx264","-crf",str(CRF),"-pix_fmt","yuv420p","-an",
                seg_c, label="cut_c")
        parts.append(seg_c)
    if len(parts) == 1:
        shutil.copy(parts[0], output); return
    concat = os.path.join(tmp_dir, "concat.txt")
    with open(concat,"w") as f:
        for p in parts: f.write(f"file '{p.replace(chr(92),'/')}'\n")
    _ffmpeg("-f","concat","-safe","0","-i",concat,
            "-c:v","libx264","-crf",str(CRF),"-pix_fmt","yuv420p",
            output, label="concat")


def _add_audio(video_no_audio, original_with_audio, output):
    probe = subprocess.run(
        ["ffprobe","-v","error","-select_streams","a",
         "-show_entries","stream=codec_type",
         "-of","default=noprint_wrappers=1:nokey=1",original_with_audio],
        capture_output=True, text=True)
    if "audio" not in probe.stdout:
        shutil.copy(video_no_audio, output); return
    _ffmpeg("-i",video_no_audio,"-i",original_with_audio,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",
            output, label="mux_audio")


def get_job_status(job_id: str):
    return jobs.get(job_id, {"status": "not found"})
