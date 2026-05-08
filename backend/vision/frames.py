import subprocess
import os
import glob
import cv2


def extract_frames(video_path, output_folder, fps=10, start_sec=None, end_sec=None):
    """
    Extract frames using FFmpeg — handles rotation metadata, seeks accurately,
    works with all codecs. Falls back to OpenCV if FFmpeg fails.
    """
    os.makedirs(output_folder, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    # Seek before input for speed (keyframe-accurate)
    if start_sec and start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]

    cmd += ["-i", video_path]

    # Duration limit
    if end_sec is not None:
        duration = end_sec - (start_sec or 0)
        if duration <= 0:
            return []
        cmd += ["-t", f"{duration:.3f}"]

    # FFmpeg auto-applies rotation metadata when filtering
    cmd += [
        "-vf", f"fps={fps}",
        "-q:v", "2",                          # high-quality JPEG
        os.path.join(output_folder, "frame_%04d.jpg")
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    frames = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))

    # Fallback: OpenCV with manual rotation if FFmpeg produced nothing
    if not frames:
        frames = _extract_opencv(video_path, output_folder, fps, start_sec, end_sec)

    return frames


def _extract_opencv(video_path, output_folder, fps, start_sec, end_sec):
    """OpenCV fallback with manual rotation correction."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    total_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / orig_fps

    s = float(start_sec or 0)
    e = float(end_sec or total_dur)
    frame_interval = max(1, int(orig_fps / fps))

    # Seek by frame number (more reliable than MSEC)
    start_frame = int(s * orig_fps)
    end_frame   = int(e * orig_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_paths = []
    saved = 0
    f_idx = start_frame

    while f_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (f_idx - start_frame) % frame_interval == 0:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation in (270, -90):
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            path = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1

        f_idx += 1

    cap.release()
    return frame_paths
