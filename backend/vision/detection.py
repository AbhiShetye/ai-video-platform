from ultralytics import YOLO
import os
import threading

_BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YOLO_PT = os.path.join(_BASE, "yolov8n.pt")

model       = None
_model_lock = threading.Lock()


def load_model():
    global model
    if model is None:
        with _model_lock:
            if model is None:
                model = YOLO(_YOLO_PT)
    return model

def detect_objects(frame_paths):
    yolo = load_model()
    results = []

    for i, frame_path in enumerate(frame_paths):
        detections = yolo(frame_path, verbose=False)
        objects = []

        for det in detections[0].boxes:
            objects.append({
                "object_id": i,
                "class": detections[0].names[int(det.cls)],
                "bbox": [int(x) for x in det.xyxy[0].tolist()],
                "confidence": round(float(det.conf), 2)
            })

        results.append({
            "frame_id": i,
            "frame_path": frame_path,
            "objects": objects
        })

    return results

def detect_first_frame(video_path):
    from vision.frames import extract_frames
    import tempfile
    tmp = tempfile.mkdtemp()
    frames = extract_frames(video_path, tmp, fps=1, end_sec=10)
    if not frames:
        return []
    # Use a fresh local YOLO instance (not the global cache) so concurrent
    # requests don't race on the model's internal first-inference fuse step.
    yolo = YOLO(_YOLO_PT)
    all_objects = []
    seen = set()
    try:
        for frame in frames:
            detections = yolo(frame, verbose=False)
            for det in detections[0].boxes:
                label = detections[0].names[int(det.cls)]
                conf = round(float(det.conf), 2)
                bbox = [int(x) for x in det.xyxy[0].tolist()]
                if label not in seen and conf > 0.4:
                    seen.add(label)
                    all_objects.append({
                        "label": label,
                        "bbox": bbox,
                        "confidence": conf
                    })
    finally:
        del yolo
    return all_objects