"""Sequential FFmpeg sampling, optionally with NVIDIA hardware decoding/scaling."""

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def run_command(command, timeout=600):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"FFmpeg execution failed: {error}") from error
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-3000:])
    return result.stdout


def check_backend(backend, device=0):
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise ValueError(f"{program} is required for --backend {backend}")
    if backend == "cuda":
        filters = run_command(["ffmpeg", "-hide_banner", "-filters"], 30).decode(errors="replace")
        if "scale_cuda" not in filters:
            raise ValueError("This FFmpeg build does not include scale_cuda")
        run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                     "-init_hw_device", f"cuda=timelapse:{device}", "-f", "lavfi",
                     "-i", "color=s=16x16", "-frames:v", "1", "-f", "null", "-"], 30)


def probe_video(path):
    data = json.loads(run_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,pix_fmt", "-of", "json", str(Path(path).resolve())]))
    try:
        stream = data["streams"][0]
        fps = float(Fraction(stream["avg_frame_rate"]))
        count, width, height = int(stream["nb_frames"]), int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Frame count/FPS metadata unavailable; use --backend opencv") from error
    if count < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValueError("Invalid source video metadata")
    return count, fps, width, height, stream.get("pix_fmt", "")


def extraction_command(path, indices, width, height, backend, device, pixel_format):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror"]
    if backend == "cuda":
        command += ["-hwaccel", "cuda", "-hwaccel_device", str(device), "-hwaccel_output_format", "cuda"]
    command += ["-noautorotate", "-i", str(Path(path).resolve()), "-map", "0:v:0", "-an", "-sn", "-dn"]
    select = (f"lte(n\\,{len(indices) - 1})" if list(indices) == list(range(len(indices)))
              else "+".join(f"eq(n\\,{int(index)})" for index in indices))
    filters = [f"select={select}"]
    if backend == "cuda":
        formats = {"yuv420p": "nv12", "yuvj420p": "nv12", "nv12": "nv12",
                   "yuv420p10le": "p010le", "p010le": "p010le"}
        if pixel_format not in formats:
            raise ValueError(f"CUDA sampling does not support input format {pixel_format}; use --backend opencv")
        filters += [f"scale_cuda={width}:{height}:interp_algo=bilinear", "hwdownload",
                    f"format={formats[pixel_format]}"]
    else:
        filters += [f"scale={width}:{height}:flags=bilinear"]
    filters += ["format=bgr24"]
    return command + ["-vf", ",".join(filters), "-vsync", "0", "-frames:v", str(len(indices)),
                      "-threads", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


def sample_frames_ffmpeg(path, count, max_width, backend="cuda", device=0, sampling="uniform"):
    frame_count, fps, width, height, pixel_format = probe_video(path)
    scale = min(1.0, max_width / width)
    # Even dimensions support NV12/P010 CUDA surfaces.
    width = max(2, int(width * scale) // 2 * 2)
    height = max(2, int(height * scale) // 2 * 2)
    indices = (np.arange(min(count, frame_count)) if sampling == "first"
               else np.unique(np.linspace(0, frame_count - 1, count).astype(int)))
    raw = run_command(extraction_command(path, indices, width, height, backend, device, pixel_format))
    expected = len(indices) * width * height * 3
    if len(raw) != expected:
        raise ValueError(f"FFmpeg returned {len(raw)} bytes instead of {expected}; incomplete decoding, no timestamps guessed")
    images = np.frombuffer(raw, np.uint8).reshape(len(indices), height, width, 3)
    selection = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return [(int(indices[i]), images[i]) for i in selection], fps, 0
