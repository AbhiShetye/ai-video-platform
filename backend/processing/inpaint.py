import cv2
import numpy as np
import os
from PIL import Image

_lama = None

def _get_lama():
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
    return _lama


def _resize_for_lama(img_bgr: np.ndarray, mask: np.ndarray, max_px: int = 768):
    """Resize image and mask to max_px wide, keep aspect ratio, ensure even dims."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_px / max(w, h))
    nw = int(w * scale) & ~1   # ensure even
    nh = int(h * scale) & ~1
    img_s  = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask,    (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img_s, mask_s, w, h


def inpaint_frame(frame_path: str, mask: np.ndarray, output_path: str,
                  bg_frame: np.ndarray = None) -> str:
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")

    mask_u8 = (mask > 127).astype(np.uint8)

    if bg_frame is not None:
        # ── Fast path: paste real background pixels, feather the edge ────
        result = frame.copy()
        result[mask_u8 == 1] = bg_frame[mask_u8 == 1]

        alpha = mask_u8.astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (41, 41), 0)
        for c in range(3):
            result[:, :, c] = np.clip(
                frame[:, :, c] * (1.0 - alpha) + result[:, :, c] * alpha,
                0, 255
            ).astype(np.uint8)

    else:
        # ── LaMa deep-learning inpainting ────────────────────────────────
        lama = _get_lama()
        img_s, mask_s, orig_w, orig_h = _resize_for_lama(frame, mask_u8 * 255)

        img_pil  = Image.fromarray(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_s)

        result_pil = lama(img_pil, mask_pil)

        result_small = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        # Scale result back to original resolution
        result = cv2.resize(result_small, (orig_w, orig_h),
                            interpolation=cv2.INTER_LANCZOS4)

        # Blend only inside the mask — keep non-masked pixels from original
        alpha = mask_u8.astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
        for c in range(3):
            result[:, :, c] = np.clip(
                frame[:, :, c] * (1.0 - alpha) + result[:, :, c] * alpha,
                0, 255
            ).astype(np.uint8)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
    return output_path
