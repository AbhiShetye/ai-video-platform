import os
import uuid
import logging
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from vision.frames import extract_frames

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

jobs: dict = {}
_jobs_lock  = threading.Lock()

# Adaptive extraction FPS:
# • FAST path (Gaussian fill, ~0.2 s/frame) → 6 fps → smooth motion
# • LAMA path (deep inpaint, ~13 s/frame)   → 1 fps → manageable time
EXTRACT_FPS_FAST = 6
EXTRACT_FPS_LAMA = 1
EXTRACT_FPS      = EXTRACT_FPS_LAMA   # kept for backward compat
CRF              = 18

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND    = os.path.dirname(_BASE_DIR)
_STORAGE    = os.path.join(_BACKEND, "storage")
_YOLO_PT    = os.path.join(_BACKEND, "yolov8n.pt")

# Global YOLO cache — loaded once, reused across all pipeline runs
_yolo_model      = None
_yolo_model_lock = threading.Lock()


def _load_yolo():
    """Return a cached YOLO instance, loading it on first call."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_model_lock:
            if _yolo_model is None:
                from ultralytics import YOLO
                _yolo_model = YOLO(_YOLO_PT)
    return _yolo_model


def _bg_is_neutral(frame_path: str, bbox: list) -> bool:
    """
    Return True when the background surrounding bbox is mostly neutral
    (wall / ceiling / floor).  Used to choose FAST vs LAMA inpaint path
    and the appropriate extraction FPS before frames are extracted.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask_u8 = np.zeros((h, w), np.uint8)
    mask_u8[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    neutral  = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 100)
    sample   = (neutral & (mask_u8 == 0)).astype(np.float32)
    coverage = float(sample.sum()) / max(float((mask_u8 > 0).sum()), 1)
    return coverage >= 0.3


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
    """Solid rectangular mask covering the full bbox with expansion padding."""
    frame = cv2.imread(frame_path)
    h, w  = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Expand by 15% of the shorter bbox dimension (min 20px).
    # YOLO tends to hug the visible content edge and miss object borders such
    # as TV bezels; this expansion ensures full coverage.
    pad = max(20, int(min(x2 - x1, y2 - y1) * 0.17))
    x1=max(0,x1-pad); y1=max(0,y1-pad)
    x2=min(w,x2+pad); y2=min(h,y2+pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


_lama_lock = threading.Lock()


def _lama_inpaint(frame_path: str, mask: np.ndarray, output_path: str):
    frame   = cv2.imread(frame_path)
    h, w    = frame.shape[:2]
    mask_u8 = (mask > 127).astype(np.uint8) * 255

    # Classify background: neutral pixels (wall/ceiling/floor) have low HSV
    # saturation and mid-high value.  Measure how many such pixels surround
    # the object relative to the object area — high ratio = smooth background.
    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    neutral   = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 100)
    sample_px = (neutral & (mask_u8 == 0)).astype(np.float32)
    coverage  = float(sample_px.sum()) / max(float((mask_u8 > 0).sum()), 1)

    if coverage >= 0.3:
        # ── FAST PATH: smooth/neutral background (wall, ceiling, floor) ──────
        # Run the large-sigma Gaussian at 1/4 resolution (identical quality,
        # ~50× faster than full-res because kernel cost scales with image area).
        DS = 4
        sh, sw   = max(1, h // DS), max(1, w // DS)
        frame_s  = cv2.resize(frame,     (sw, sh), interpolation=cv2.INTER_AREA)
        samp_s   = cv2.resize(sample_px, (sw, sh), interpolation=cv2.INTER_AREA)

        gauss_s = np.zeros((sh, sw, 3), np.float32)
        for c in range(3):
            ch      = frame_s[:, :, c].astype(np.float32) * samp_s
            # Near scale (captures local gradient near the mask edge)
            bv_n = cv2.GaussianBlur(ch,     (0, 0), 60 / DS)
            bw_n = cv2.GaussianBlur(samp_s, (0, 0), 60 / DS)
            near = bv_n / (bw_n + 1e-6)
            # Far scale (covers pixels deep inside large masks)
            bv_f = cv2.GaussianBlur(ch,     (0, 0), 150 / DS)
            bw_f = cv2.GaussianBlur(samp_s, (0, 0), 150 / DS)
            far  = bv_f / (bw_f + 1e-6)
            t = np.clip(bw_n * 4, 0, 1)
            gauss_s[:, :, c] = near * t + far * (1.0 - t)

        # Upsample fill back to original resolution
        gauss = cv2.resize(gauss_s, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add camera grain that matches the texture variance of the real wall
        ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        ring   = (cv2.dilate(mask_u8, ring_k) > 0) & (mask_u8 == 0) & neutral
        for c in range(3):
            std   = min(float(frame[:, :, c][ring].std()) if ring.any() else 2.0, 3.0)
            noise = np.random.normal(0, std * 0.65, (h, w)).astype(np.float32)
            gauss[:, :, c][mask_u8 > 0] = np.clip(
                gauss[:, :, c][mask_u8 > 0] + noise[mask_u8 > 0], 0, 255
            )
        fill = gauss

    else:
        # ── DEEP PATH: complex/textured background — use LaMa ────────────────
        from simple_lama_inpainting import SimpleLama
        from PIL import Image
        global _lama_model
        if _lama_model is None:
            with _lama_lock:
                if _lama_model is None:
                    _lama_model = SimpleLama()

        scale = min(1.0, 1024 / max(w, h))
        nw, nh = int(w * scale) & ~1, int(h * scale) & ~1
        fr_s   = cv2.resize(frame,   (nw, nh), interpolation=cv2.INTER_AREA)
        mk_s   = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
        result_pil = _lama_model(
            Image.fromarray(cv2.cvtColor(fr_s, cv2.COLOR_BGR2RGB)),
            Image.fromarray(mk_s)
        )
        lama_full = cv2.resize(
            cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR),
            (w, h), interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32)

        # Color-correct LaMa fill to match the real boundary
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        boundary = (cv2.dilate(mask_u8, dilate_k) > 0) & (mask_u8 == 0)
        if boundary.any() and mask_u8.any():
            for c in range(3):
                wall_mean = float(frame[:, :, c][boundary].mean())
                fill_mean = float(lama_full[:, :, c][mask_u8 > 0].mean())
                lama_full[:, :, c][mask_u8 > 0] = np.clip(
                    lama_full[:, :, c][mask_u8 > 0] + (wall_mean - fill_mean), 0, 255
                )
        fill = lama_full

    # ── Composite: wide feathered blend (~60 px transition zone) ─────────────
    mask_f32 = (mask > 127).astype(np.float32)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    alpha    = cv2.GaussianBlur(cv2.erode(mask_f32, kernel, iterations=2), (61, 61), 0)
    result   = frame.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = frame[:, :, c] * (1.0 - alpha) + fill[:, :, c] * alpha

    cv2.imwrite(output_path, result.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 97])
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

        # ── STEP 1: Probe first frame → choose FPS + inpaint path ────────
        jobs[job_id]["progress"] = 5
        probe_dir = os.path.join(_STORAGE, job_id, "probe")
        probe_frames = extract_frames(video_path, probe_dir, fps=1,
                                      start_sec=start_sec,
                                      end_sec=min(start_sec + 2, end_sec))
        neutral_bg  = bool(probe_frames and _bg_is_neutral(probe_frames[0], hint_bbox))
        extract_fps = EXTRACT_FPS_FAST if neutral_bg else EXTRACT_FPS_LAMA
        log.info("background=%s → extract_fps=%d",
                 "neutral" if neutral_bg else "complex", extract_fps)

        # ── STEP 2: Extract frames at chosen FPS ─────────────────────────
        jobs[job_id]["progress"] = 8
        log.info("Step 2: extracting frames %.1fs–%.1fs at %d fps",
                 start_sec, end_sec, extract_fps)
        frame_paths = extract_frames(video_path, frames_dir,
                                     fps=extract_fps,
                                     start_sec=start_sec, end_sec=end_sec)
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {start_sec}s to {end_sec}s.")
        log.info("  extracted %d frames", len(frame_paths))

        # ── STEP 3: YOLO tracking with stride (1 YOLO run per second) ────
        jobs[job_id]["progress"] = 15
        log.info("Step 3: YOLO tracking (stride=%d)", extract_fps)
        yolo   = _load_yolo()
        stride = max(1, extract_fps)   # 1 detection per second of video
        per_frame_bboxes = []
        current_bbox = hint_bbox
        for i, fp in enumerate(frame_paths):
            if i % stride == 0:        # re-detect once per second
                detected     = _detect_in_frame(fp, current_bbox, yolo)
                current_bbox = detected or current_bbox
            per_frame_bboxes.append(current_bbox)
        log.info("  ran YOLO on %d/%d frames", len(frame_paths)//stride + 1,
                 len(frame_paths))

        # ── STEP 4: Generate masks ────────────────────────────────────────
        jobs[job_id]["progress"] = 22
        log.info("Step 4: generating masks")
        masks = [_bbox_mask(fp, bb)
                 for fp, bb in zip(frame_paths, per_frame_bboxes)]

        # ── STEP 5: Inpainting (parallel workers for fast path) ───────────
        jobs[job_id]["progress"] = 30
        n = len(frame_paths)
        log.info("Step 5: inpainting %d frames", n)
        os.makedirs(edited_dir, exist_ok=True)
        out_paths = [os.path.join(edited_dir, f"edited_{i:04d}.jpg")
                     for i in range(n)]

        def _inpaint_task(args):
            fp, mask, out_path = args
            _lama_inpaint(fp, mask, out_path)
            return out_path

        workers = min(4, os.cpu_count() or 1) if neutral_bg else 1
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_inpaint_task, (fp, m, op)): i
                    for i, (fp, m, op) in enumerate(zip(frame_paths, masks, out_paths))}
            for fut in as_completed(futs):
                fut.result()
                completed += 1
                jobs[job_id]["progress"] = 30 + int(completed / n * 42)
        log.info("  inpainting complete (%d workers)", workers)
        edited_paths = out_paths

        # ── STEP 6: Encode edited segment ────────────────────────────────
        jobs[job_id]["progress"] = 73
        tmp_dir = tempfile.mkdtemp(prefix="frameai_")
        seg_b   = os.path.join(tmp_dir, "seg_b.mp4")
        log.info("Step 6: encoding edited segment")
        _ffmpeg(
            "-framerate", str(extract_fps),
            "-i", os.path.join(edited_dir, "edited_%04d.jpg"),
            "-vf", f"fps={orig_fps:.3f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            seg_b, label="encode_seg"
        )

        # ── STEP 7: Assemble full video ───────────────────────────────────
        jobs[job_id]["progress"] = 83
        log.info("Step 7: assembling full video")
        no_audio = os.path.join(tmp_dir, "no_audio.mp4")
        _assemble(video_path, seg_b, no_audio, start_sec, end_sec, duration, tmp_dir, rotation)

        # ── STEP 8: Add audio ─────────────────────────────────────────────
        jobs[job_id]["progress"] = 93
        final = os.path.join(out_dir, "final.mp4")
        log.info("Step 8: adding audio → %s", final)
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


# ── Quick Edit (FFmpeg-based tools: trim, speed, rotate, filters) ─────────────

def _build_atempo(factor: float) -> str:
    """Build atempo filter chain. FFmpeg atempo only accepts 0.5–2.0 per node."""
    filters = []
    f = factor
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    filters.append(f"atempo={f:.4f}")
    return ",".join(filters)


def run_quick_edit(video_path: str, operation: dict, job_id: str = None):
    """
    Fast FFmpeg-based editing.  Handles: trim | speed | rotate | filter.
    operation dict keys depend on type — see each branch below.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== QUICK EDIT %s  op=%s ===", job_id, operation.get("type"))

    try:
        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        final   = os.path.join(out_dir, "final.mp4")
        op_type = operation.get("type", "")

        jobs[job_id]["progress"] = 20

        if op_type == "trim":
            start = float(operation.get("start", 0))
            end   = float(operation.get("end", 0))
            if end <= start:
                raise ValueError("End must be after start")
            _ffmpeg(
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                "-i", video_path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                final, label="trim"
            )

        elif op_type == "speed":
            factor = max(0.25, min(4.0, float(operation.get("factor", 1.0))))
            pts    = 1.0 / factor
            vf     = (f"setpts={pts:.6f}*PTS,"
                      "scale=trunc(iw/2)*2:trunc(ih/2)*2")
            af     = _build_atempo(factor)
            _ffmpeg(
                "-i", video_path,
                "-vf", vf, "-af", af,
                "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                final, label="speed"
            )

        elif op_type == "rotate":
            angle = int(operation.get("angle", 0)) % 360
            flip  = operation.get("flip", "")
            vf_parts = []
            if angle == 90:
                vf_parts.append("transpose=1")
            elif angle == 180:
                vf_parts.append("transpose=2,transpose=2")
            elif angle == 270:
                vf_parts.append("transpose=2")
            if flip == "h":
                vf_parts.append("hflip")
            elif flip == "v":
                vf_parts.append("vflip")
            vf_parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
            _ffmpeg(
                "-i", video_path,
                "-vf", ",".join(vf_parts),
                "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                final, label="rotate"
            )

        elif op_type == "filter":
            brightness  = max(-1.0, min(1.0, float(operation.get("brightness", 0))))
            contrast    = max(0.0,  min(3.0, float(operation.get("contrast",   1.0))))
            saturation  = max(0.0,  min(3.0, float(operation.get("saturation", 1.0))))
            blur        = max(0.0,  min(10.0, float(operation.get("blur",      0))))
            vf_parts    = [
                f"eq=brightness={brightness:.3f}:"
                f"contrast={contrast:.3f}:"
                f"saturation={saturation:.3f}"
            ]
            if blur > 0.1:
                vf_parts.append(f"gblur=sigma={blur:.1f}")
            vf_parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
            _ffmpeg(
                "-i", video_path,
                "-vf", ",".join(vf_parts),
                "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                final, label="filter"
            )

        else:
            raise ValueError(f"Unknown operation type: {op_type!r}")

        jobs[job_id]["progress"] = 90
        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== QUICK EDIT %s DONE → %s ===", job_id, final)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== QUICK EDIT %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id


# ── Magic Eraser ──────────────────────────────────────────────────────────────

def _combined_mask(frame_path: str, bboxes: list) -> np.ndarray:
    """Union mask covering every bbox in the list (with standard expansion pad)."""
    frame = cv2.imread(frame_path)
    h, w  = frame.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = max(20, int(min(x2 - x1, y2 - y1) * 0.17))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        combined[y1:y2, x1:x2] = 255
    return combined


def run_magic_erase(video_path: str, command: dict, job_id: str = None):
    """
    Remove multiple user-selected objects from a video segment.
    command keys: bboxes [[x1,y1,x2,y2],...], start_time, end_time
    Each object is tracked frame-by-frame with YOLO; all masks are merged
    into one combined mask before inpainting so a single fill pass covers
    every selected object.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== MAGIC ERASE JOB %s START ===", job_id)

    try:
        hint_bboxes = command.get("bboxes") or []
        if not hint_bboxes:
            raise ValueError("No objects selected")

        orig_fps, duration, rotation = _video_info(video_path)
        start_sec = float(command.get("start_time") or 0)
        end_sec   = min(float(command.get("end_time") or duration), duration)
        if start_sec >= end_sec:
            raise ValueError(f"Invalid range: {start_sec}–{end_sec}s")

        base       = _STORAGE
        frames_dir = os.path.join(base, job_id, "frames")
        edited_dir = os.path.join(base, job_id, "edited")
        out_dir    = os.path.join(base, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)

        # ── STEP 1: Probe first frame → choose FPS ───────────────────────
        jobs[job_id]["progress"] = 5
        probe_dir   = os.path.join(_STORAGE, job_id, "probe")
        probe_frames = extract_frames(video_path, probe_dir, fps=1,
                                      start_sec=start_sec,
                                      end_sec=min(start_sec + 2, end_sec))
        neutral_bg  = bool(probe_frames and
                           _bg_is_neutral(probe_frames[0], hint_bboxes[0]))
        extract_fps = EXTRACT_FPS_FAST if neutral_bg else EXTRACT_FPS_LAMA

        # ── STEP 2: Extract frames at chosen FPS ─────────────────────────
        jobs[job_id]["progress"] = 8
        frame_paths = extract_frames(video_path, frames_dir, fps=extract_fps,
                                     start_sec=start_sec, end_sec=end_sec)
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {start_sec}s to {end_sec}s")
        log.info("  extracted %d frames at %dfps", len(frame_paths), extract_fps)

        # ── STEP 3: Track each object with stride ────────────────────────
        jobs[job_id]["progress"] = 15
        yolo   = _load_yolo()
        stride = max(1, extract_fps)
        per_object_tracks = []
        for hint in hint_bboxes:
            track = []
            current = hint
            for i, fp in enumerate(frame_paths):
                if i % stride == 0:
                    detected = _detect_in_frame(fp, current, yolo)
                    current  = detected or current
                track.append(current)
            per_object_tracks.append(track)

        # ── STEP 4: Combined mask per frame ──────────────────────────────
        jobs[job_id]["progress"] = 25
        masks = []
        for fi, fp in enumerate(frame_paths):
            bboxes_this_frame = [per_object_tracks[oi][fi]
                                 for oi in range(len(hint_bboxes))]
            masks.append(_combined_mask(fp, bboxes_this_frame))

        # ── STEP 5: Parallel inpainting ───────────────────────────────────
        jobs[job_id]["progress"] = 30
        n = len(frame_paths)
        os.makedirs(edited_dir, exist_ok=True)
        out_paths = [os.path.join(edited_dir, f"edited_{i:04d}.jpg")
                     for i in range(n)]

        def _inpaint_task_me(args):
            fp, mask, op = args
            _lama_inpaint(fp, mask, op)

        workers   = min(4, os.cpu_count() or 1) if neutral_bg else 1
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_inpaint_task_me, (fp, m, op)): i
                    for i, (fp, m, op) in enumerate(zip(frame_paths, masks, out_paths))}
            for fut in as_completed(futs):
                fut.result()
                completed += 1
                jobs[job_id]["progress"] = 30 + int(completed / n * 45)
        edited_paths = out_paths

        # ── STEP 6: Encode edited segment ────────────────────────────────
        jobs[job_id]["progress"] = 76
        tmp_dir = tempfile.mkdtemp(prefix="frameai_")
        seg_b   = os.path.join(tmp_dir, "seg_b.mp4")
        _ffmpeg(
            "-framerate", str(extract_fps),
            "-i", os.path.join(edited_dir, "edited_%04d.jpg"),
            "-vf", f"fps={orig_fps:.3f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            seg_b, label="encode_magic"
        )

        # ── STEP 7: Assemble + audio ──────────────────────────────────────
        jobs[job_id]["progress"] = 87
        no_audio = os.path.join(tmp_dir, "no_audio.mp4")
        _assemble(video_path, seg_b, no_audio, start_sec, end_sec, duration, tmp_dir, rotation)

        jobs[job_id]["progress"] = 95
        final = os.path.join(out_dir, "final.mp4")
        _add_audio(no_audio, video_path, final)

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== MAGIC ERASE JOB %s DONE → %s ===", job_id, final)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== MAGIC ERASE JOB %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id


# ── Audio mute ────────────────────────────────────────────────────────────────

def run_mute_audio(video_path: str, segments: list, job_id: str = None):
    """
    Silence specific time segments in a video.
    segments: [{"start": float, "end": float}, ...]
    Uses FFmpeg volume filter — video stream is stream-copied (no re-encode).
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0}
    log.info("=== MUTE AUDIO JOB %s START  segments=%s ===", job_id, segments)

    try:
        if not segments:
            raise ValueError("No segments provided")

        out_dir = os.path.join(_STORAGE, job_id, "output")
        os.makedirs(out_dir, exist_ok=True)
        final = os.path.join(out_dir, "final.mp4")

        jobs[job_id]["progress"] = 30

        # Build FFmpeg volume-filter expression:
        # volume=enable='between(t,5,12)+between(t,20,25)':volume=0
        conditions = "+".join(
            f"between(t,{float(s['start']):.3f},{float(s['end']):.3f})"
            for s in segments
        )
        _ffmpeg(
            "-i", video_path,
            "-af", f"volume=enable='{conditions}':volume=0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            final, label="mute_audio"
        )

        jobs[job_id].update({"status": "completed", "output": final, "progress": 100})
        log.info("=== MUTE AUDIO JOB %s DONE → %s ===", job_id, final)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("=== MUTE AUDIO JOB %s FAILED ===\n%s", job_id, tb)
        jobs[job_id].update({"status": "failed", "error": str(exc), "trace": tb})

    return job_id
