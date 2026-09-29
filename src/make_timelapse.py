"""Combine sampled frames from one device's recordings into a captioned MP4."""

import argparse
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

try:
    from src.detect_scene_changes import parse_date, parse_datetime, video_identity
    from src.timelapse_ffmpeg import check_backend, sample_frames_ffmpeg
except ModuleNotFoundError:
    from detect_scene_changes import parse_date, parse_datetime, video_identity
    from timelapse_ffmpeg import check_backend, sample_frames_ffmpeg


def sample_frames(path, count, max_width, sampling="uniform"):
    """Return resized images and their source indices; duplicate short clips safely."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    failures = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_count < 1:
            raise ValueError(f"Cannot determine frame count: {path}")
        indices = (np.arange(min(count, frame_count)) if sampling == "first"
                   else np.unique(np.linspace(0, frame_count - 1, count).astype(int)))
        for index in indices:
            if sampling != "first":
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                failures += 1
                continue
            scale = min(1.0, max_width / frame.shape[1])
            if scale < 1:
                frame = cv2.resize(frame, (max_width, max(2, round(frame.shape[0] * scale))),
                                   interpolation=cv2.INTER_AREA)
            frames.append((int(index), frame))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No readable sampled frames: {path}")
    # Keep the same output duration for a short or partly damaged recording.
    selection = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[i] for i in selection], fps, failures


def fit_frame(frame, size):
    width, height = size
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), np.uint8)
    top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def extract_video(path, count, max_width, backend="opencv", device=0, sampling="uniform"):
    try:
        if backend != "opencv":
            return sample_frames_ffmpeg(path, count, max_width, backend, device, sampling), None
        return sample_frames(path, count, max_width, sampling), None
    except (ValueError, cv2.error) as error:
        return None, str(error)


def ordered_samples(videos, count, max_width, workers, backend="opencv", device=0, sampling="uniform"):
    """Bound in-flight clips and yield them in recording order, not completion order."""
    if workers == 1:
        for stamp, path in videos:
            yield stamp, path, extract_video(path, count, max_width, backend, device, sampling)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        source, pending = iter(videos), deque()

        def enqueue():
            item = next(source, None)
            if item is not None:
                stamp, path = item
                pending.append((stamp, path, executor.submit(extract_video, path, count, max_width, backend, device, sampling)))

        for _ in range(workers):
            enqueue()
        try:
            while pending:
                stamp, path, future = pending.popleft()
                result = future.result()
                yield stamp, path, result
                del result, future
                enqueue()
        finally:
            for _, _, future in pending:
                future.cancel()


def caption_frame(frame, stamp, index, source_fps, device):
    """Time is estimated from the filename start time and sampled frame offset."""
    known_fps = math.isfinite(source_fps) and source_fps > 0
    captured_at = stamp + timedelta(seconds=index / source_fps) if known_fps else stamp
    label = captured_at.strftime("%Y-%m-%d %H:%M:%S")
    if not known_fps:
        label += " (video start; FPS unknown)"
    output = frame.copy()
    width = output.shape[1]
    margin = max(4, round(width * 0.012))
    scale = max(0.3, width / 1500)
    longest = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                  for text in (label, device))
    scale *= min(1, (width - 2 * margin) / max(longest, 1))
    thickness = max(1, round(scale * 2))
    text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1]
    line_height = text_height + margin
    band_height = min(output.shape[0], margin + 2 * line_height + margin)
    output[:band_height] = (output[:band_height].astype(np.float32) * 0.25).astype(np.uint8)
    for row, text in enumerate((label, device)):
        cv2.putText(output, text, (margin, margin + text_height + row * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output, captured_at if known_fps else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Full device name, e.g. ANU-25-summer-3")
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--pattern", default="*.mp4")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-date", type=parse_date)
    start.add_argument("--start-datetime", type=parse_datetime)
    end = parser.add_mutually_exclusive_group()
    end.add_argument("--end-date", type=parse_date)
    end.add_argument("--end-datetime", type=parse_datetime)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--sampling", choices=["first", "uniform"], default="first",
                        help="first: read the first frames consecutively (default, fast); uniform: sample across the entire video")
    parser.add_argument("--workers", type=int, default=1, help="Videos extracted concurrently; output remains chronological")
    parser.add_argument("--backend", choices=["opencv", "ffmpeg", "cuda"], default="opencv",
                        help="opencv: CPU seeking; ffmpeg: CPU sequential decode; cuda: NVIDIA sequential decode/resize")
    parser.add_argument("--gpu-device", type=int, default=0, help="NVIDIA GPU index for --backend cuda")
    parser.add_argument("--fps", type=float, default=24, help="Output FPS; 24 frames / 24 FPS = one second per video")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--output", type=Path, help="Output MP4; default bee_count_output/timelapse/DEVICE_FIRST_LAST.mp4")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.gpu_device < 0:
        parser.error("gpu-device must be nonnegative")
    if args.frames_per_video < 1 or not math.isfinite(args.fps) or args.fps <= 0 or args.max_width < 160:
        parser.error("Require frames-per-video >= 1, finite fps > 0, max-width >= 160")
    start_at = args.start_datetime or (datetime.combine(args.start_date, time.min) if args.start_date else None)
    end_at = args.end_datetime or (datetime.combine(args.end_date, time.max) if args.end_date else None)
    if start_at and end_at and start_at > end_at:
        parser.error("start must be on or before end")
    videos = []
    for path in args.video_dir.glob(args.pattern):
        if not path.is_file():
            continue
        try:
            device, stamp = video_identity(path)
        except ValueError:
            continue
        if device == args.device and (start_at is None or stamp >= start_at) and (end_at is None or stamp <= end_at):
            videos.append((stamp, path))
    videos.sort()
    if not videos:
        parser.error("No videos match the device and recording period")
    if args.backend != "opencv":
        try:
            check_backend(args.backend, args.gpu_device)
        except ValueError as error:
            parser.error(f"{args.backend} backend unavailable: {error}")
    output = args.output or Path("bee_count_output/timelapse") / (
        f"{args.device}_{videos[0][0]:%Y%m%d_%H%M%S}_{videos[-1][0]:%Y%m%d_%H%M%S}.mp4")
    if output.suffix.lower() != ".mp4":
        parser.error("output must have an .mp4 extension")
    if any(output.resolve() == path.resolve() for _, path in videos):
        parser.error("output cannot overwrite an input video")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(videos)} videos; up to {len(videos) * args.frames_per_video / args.fps:.2f} seconds", flush=True)
    writer, temporary = None, None
    written, rows, size = 0, [], None
    samples = ordered_samples(videos, args.frames_per_video, args.max_width, args.workers,
                              args.backend, args.gpu_device, args.sampling)
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as stream:
            temporary = stream.name
        for position, (stamp, path, extraction) in enumerate(samples, 1):
            row = {"video": str(path), "video_start": stamp.isoformat(sep=" "),
                   "backend": args.backend,
                   "sampling": args.sampling,
                   "status": "skipped", "output_start_sec": written / args.fps,
                   "output_frames": 0, "failed_samples": "", "first_frame_time": "", "last_frame_time": "", "note": ""}
            extracted, error = extraction
            if error is not None:
                row["note"] = str(error)
                rows.append(row)
                print(f"[{position}/{len(videos)}] SKIP {path.name}: {error}", flush=True)
                continue
            frames, source_fps, failures = extracted
            if writer is None:
                h, w = frames[0][1].shape[:2]
                size = (max(2, w // 2 * 2), max(2, h // 2 * 2))
                writer = cv2.VideoWriter(temporary, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                if not writer.isOpened():
                    raise RuntimeError("Cannot open MP4 video writer (mp4v codec)")
            times = []
            for index, frame in frames:
                frame, captured_at = caption_frame(fit_frame(frame, size), stamp, index, source_fps, args.device)
                writer.write(frame)
                written += 1
                times.append(captured_at.isoformat(sep=" ") if captured_at else "")
            row.update(status="written", output_frames=len(frames), failed_samples=failures,
                       first_frame_time=times[0], last_frame_time=times[-1],
                       note="missing samples replaced by available frames" if failures else "")
            rows.append(row)
            print(f"[{position}/{len(videos)}] {path.name}: {len(frames)} frames ({failures} unreadable samples)", flush=True)
        if writer is not None:
            writer.release()
            writer = None
        if written == 0:
            raise RuntimeError("No readable videos; no timelapse produced")
        os.replace(temporary, output)
        temporary = None
        manifest = output.with_suffix(".csv")
        with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        print(f"Saved: {output} ({written} frames, {written / args.fps:.2f} seconds)\nManifest: {manifest}")
    finally:
        samples.close()
        if writer is not None:
            writer.release()
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
