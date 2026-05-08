import numpy as np
import cv2


def generate_mask(frame_path: str, bbox: list) -> dict:
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1

    if bw < 4 or bh < 4:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return {"frame_path": frame_path, "bbox": [x1, y1, x2, y2], "mask": mask}

    # GrabCut: precise foreground segmentation from the bounding box
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, y1, bw, bh)

    try:
        cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)

        # Fall back to rect if GrabCut returned almost nothing
        if fg_mask.sum() < (bw * bh * 255 * 0.05):
            raise ValueError("sparse mask")

    except Exception:
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[y1:y2, x1:x2] = 255

    # Small dilation so no thin ring of the object remains at the edges
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    return {"frame_path": frame_path, "bbox": [x1, y1, x2, y2], "mask": fg_mask}


def generate_masks_for_frames(frame_paths: list, bbox: list) -> list:
    return [generate_mask(fp, bbox) for fp in frame_paths]
