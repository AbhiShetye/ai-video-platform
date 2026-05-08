import subprocess
import os

def frames_to_video(frames_folder, output_path, fps=5):
    frame_pattern = os.path.join(frames_folder, "edited_%04d.jpg")
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frames_to_video failed:\n{result.stderr[-2000:]}")
    return output_path


def frames_to_video_with_audio(frames_folder, audio_path, output_path, fps=5):
    frame_pattern = os.path.join(frames_folder, "edited_%04d.jpg")
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-y",
        output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frames_to_video_with_audio failed:\n{result.stderr[-2000:]}")
    return output_path
