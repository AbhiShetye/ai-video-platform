import subprocess
import os

def extract_audio(video_path, output_audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        "-y",
        output_audio_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extract_audio failed:\n{result.stderr[-2000:]}")
    return output_audio_path


def merge_audio_video(video_path, audio_path, output_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-y",
        output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge_audio_video failed:\n{result.stderr[-2000:]}")
    return output_path
