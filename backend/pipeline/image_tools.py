"""
FrameAI Photo Editor pipeline.
All functions process single images (JPG/PNG/WebP) and return job IDs.
"""

import os
import uuid
import logging
import threading
import time as _time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from pipeline.engine import _STORAGE, jobs

log = logging.getLogger(__name__)

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_UPLOAD_DIR = os.path.join(_BACKEND, "img_uploads")
os.makedirs(IMG_UPLOAD_DIR, exist_ok=True)

_ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def _out_path(job_id: str, ext: str = ".jpg") -> str:
    out_dir = os.path.join(_STORAGE, job_id, "output")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "result" + ext)


def _new_img_job(job_id: str) -> dict:
    entry = {"status": "processing", "progress": 0, "_ts": _time.time(), "is_image": True}
    jobs[job_id] = entry
    return entry


# ── 1. BACKGROUND REMOVAL ─────────────────────────────────────────────────────

def run_image_bg_remove(img_path: str, bg: str = "transparent", job_id: str = None):
    """Remove background from image. bg: 'transparent'|'black'|'white'|'green'|'blue'"""
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)
    log.info("=== IMG-BG-REMOVE %s bg=%s ===", job_id, bg)

    def _run():
        try:
            from rembg import remove as rembg_remove, new_session
            jobs[job_id]["progress"] = 15
            session = new_session("u2net")
            img = Image.open(img_path).convert("RGB")
            jobs[job_id]["progress"] = 50
            result = rembg_remove(img, session=session)  # RGBA

            if bg == "transparent":
                out = _out_path(job_id, ".png")
                result.save(out, format="PNG")
            else:
                colors = {"black": (0,0,0), "white": (255,255,255),
                          "green": (0,255,0), "blue": (0,0,255), "red": (255,0,0)}
                rgb = colors.get(bg, (0, 0, 0))
                bg_img = Image.new("RGBA", result.size, (*rgb, 255))
                final = Image.alpha_composite(bg_img, result).convert("RGB")
                out = _out_path(job_id, ".jpg")
                final.save(out, format="JPEG", quality=95)

            jobs[job_id].update({"status": "completed", "output": out, "progress": 100})
            log.info("=== IMG-BG-REMOVE %s DONE ===", job_id)
        except Exception as exc:
            import traceback
            log.error("IMG-BG-REMOVE failed: %s", exc)
            jobs[job_id].update({"status": "failed", "error": str(exc),
                                  "trace": traceback.format_exc()})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 2. FILTERS & COLOR GRADING ────────────────────────────────────────────────

def run_image_filter(img_path: str, filters: dict, job_id: str = None):
    """Apply: brightness, contrast, saturation, sharpness, blur. All floats, 1.0=original."""
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)

    def _run():
        try:
            img = Image.open(img_path).convert("RGB")
            jobs[job_id]["progress"] = 20

            b = filters.get("brightness", 1.0)
            c = filters.get("contrast", 1.0)
            s = filters.get("saturation", 1.0)
            sh = filters.get("sharpness", 1.0)
            bl = filters.get("blur", 0.0)

            if b != 1.0:
                img = ImageEnhance.Brightness(img).enhance(b)
            if c != 1.0:
                img = ImageEnhance.Contrast(img).enhance(c)
            if s != 1.0:
                img = ImageEnhance.Color(img).enhance(s)
            if sh != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(sh)
            if bl > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=bl))

            jobs[job_id]["progress"] = 80
            out = _out_path(job_id, ".jpg")
            img.save(out, format="JPEG", quality=95)
            jobs[job_id].update({"status": "completed", "output": out, "progress": 100})
        except Exception as exc:
            jobs[job_id].update({"status": "failed", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 3. CROP & RESIZE ──────────────────────────────────────────────────────────

def run_image_crop(img_path: str, x: int, y: int, w: int, h: int,
                   out_w: int = 0, out_h: int = 0, job_id: str = None):
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)

    def _run():
        try:
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            x2 = min(x + w, iw)
            y2 = min(y + h, ih)
            cropped = img.crop((max(0, x), max(0, y), x2, y2))
            if out_w > 0 and out_h > 0:
                cropped = cropped.resize((out_w, out_h), Image.LANCZOS)
            out = _out_path(job_id, ".jpg")
            cropped.save(out, format="JPEG", quality=95)
            jobs[job_id].update({"status": "completed", "output": out, "progress": 100,
                                  "width": cropped.width, "height": cropped.height})
        except Exception as exc:
            jobs[job_id].update({"status": "failed", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 4. AI UPSCALE (LANCZOS + UNSHARP) ────────────────────────────────────────

def run_image_upscale(img_path: str, scale: int = 2, job_id: str = None):
    """Upscale 2× or 4× using LANCZOS + UnsharpMask for crisp detail."""
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)

    def _run():
        try:
            img = Image.open(img_path).convert("RGB")
            jobs[job_id]["progress"] = 20
            nw, nh = img.width * scale, img.height * scale
            up = img.resize((nw, nh), Image.LANCZOS)
            jobs[job_id]["progress"] = 70
            up = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
            jobs[job_id]["progress"] = 90
            out = _out_path(job_id, ".jpg")
            up.save(out, format="JPEG", quality=97)
            jobs[job_id].update({"status": "completed", "output": out, "progress": 100,
                                  "width": nw, "height": nh})
        except Exception as exc:
            jobs[job_id].update({"status": "failed", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 5. TEXT / WATERMARK OVERLAY ───────────────────────────────────────────────

_COLOR_MAP = {
    "white": "#ffffff", "black": "#000000", "yellow": "#fbbf24",
    "red": "#f87171",   "green": "#34d399", "cyan": "#38bdf8",
    "blue": "#818cf8",  "magenta": "#a78bfa",
}
_POSITION_MAP = {
    "tl": (0, 0), "tc": (0.5, 0), "tr": (1, 0),
    "bl": (0, 1), "bc": (0.5, 1), "br": (1, 1),
    "center": (0.5, 0.5),
}

def run_image_text(img_path: str, text: str, position: str = "bc",
                   size: int = 48, color: str = "white", job_id: str = None):
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)

    def _run():
        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            iw, ih = img.size
            font = None
            for fname in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    font = ImageFont.truetype(fname, size)
                    break
                except Exception:
                    pass
            if font is None:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            margin = max(24, size // 2)
            px, py = _POSITION_MAP.get(position, (0.5, 1.0))
            x = int((iw - tw) * px)
            y = int((ih - th - margin) * py) + (margin if py < 0.9 else 0)
            x = max(margin, min(iw - tw - margin, x))
            y = max(margin, min(ih - th - margin, y))

            hex_c = _COLOR_MAP.get(color, "#ffffff")
            # Drop shadow
            draw.text((x + 2, y + 2), text, font=font, fill="#00000099")
            draw.text((x, y), text, font=font, fill=hex_c)

            out = _out_path(job_id, ".jpg")
            img.save(out, format="JPEG", quality=95)
            jobs[job_id].update({"status": "completed", "output": out, "progress": 100})
        except Exception as exc:
            jobs[job_id].update({"status": "failed", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 6. OBJECT REMOVAL FROM IMAGE (LaMa inpainting) ───────────────────────────

def run_image_object_remove(img_path: str, bbox: list, job_id: str = None):
    """Remove a bounding-box region from an image using LaMa inpainting."""
    if not job_id:
        job_id = str(uuid.uuid4())
    _new_img_job(job_id)

    def _run():
        try:
            from simple_lama_inpainting import SimpleLama
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise RuntimeError("Cannot read image file")
            h, w = img_cv.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = max(30, int(min(x2 - x1, y2 - y1) * 0.25))
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

            mask = np.zeros((h, w), np.uint8)
            mask[y1:y2, x1:x2] = 255
            jobs[job_id]["progress"] = 30

            img_pil  = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask)
            lama = SimpleLama()
            result = lama(img_pil, mask_pil)
            jobs[job_id]["progress"] = 90

            out = _out_path(job_id, ".jpg")
            result.save(out, format="JPEG", quality=95)
            jobs[job_id].update({"status": "completed", "output": out, "progress": 100})
        except Exception as exc:
            import traceback
            jobs[job_id].update({"status": "failed", "error": str(exc),
                                  "trace": traceback.format_exc()})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── 7. DETECT OBJECTS IN IMAGE (for remove-object flow) ───────────────────────

def detect_image_objects(img_path: str) -> list:
    """Run YOLO on a single image, return list of detected objects."""
    from pipeline.engine import _load_yolo
    model = _load_yolo()
    results = model(img_path, verbose=False)[0]
    objects = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        label = results.names[int(box.cls[0])]
        objects.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})
    objects.sort(key=lambda o: o["confidence"], reverse=True)
    return objects[:20]
