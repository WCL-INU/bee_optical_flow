"""Report persistent framing changes between chronologically ordered device videos."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
import zipfile

import cv2
import numpy as np

try:
    from src.scene_lines import compare_lines, extract_lines
except ModuleNotFoundError:
    from scene_lines import compare_lines, extract_lines


@dataclass
class Background:
    image: np.ndarray
    mask: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray | None
    usable: bool
    reason: str


def build_background(frames):
    """Suppress transient foreground and exclude temporally unstable keypoints."""
    if len(frames) < 5:
        raise ValueError("At least five readable frames are required")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Frame dimensions changed within a video")
    stack = np.stack(frames).astype(np.float32)
    # Correct global exposure offsets before measuring temporal variability.
    levels = np.median(stack, axis=(1, 2), keepdims=True)
    aligned = np.clip(stack - levels + np.median(levels), 0, 255)
    image = np.median(aligned, axis=0).astype(np.uint8)
    variation = np.percentile(np.abs(aligned - image), 80, axis=0)
    unstable = (variation > 18).astype(np.uint8)
    unstable = cv2.dilate(unstable, np.ones((9, 9), np.uint8))
    mask = (1 - unstable) * 255
    normalized = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(image)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=2500).detectAndCompute(normalized, mask)
    points = np.array([p.pt for p in keypoints], dtype=np.float32).reshape(-1, 2)
    dynamic_range = float(np.percentile(image, 95) - np.percentile(image, 5))
    stable_fraction = float(np.mean(mask > 0))
    structural_features = len(points) >= 80 or len(extract_lines(image, mask)) >= 3
    usable = structural_features and dynamic_range >= 25 and stable_fraction >= 0.25
    reason = "usable" if usable else "too few stable features, poor contrast, or excessive motion"
    return Background(image, mask, points, descriptors, usable, reason)


def sample_background(path, samples=21, max_width=800):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 5:
            raise ValueError(f"Insufficient video frames: {path}")
        # Interior samples avoid partially decoded opening/closing frames.
        indices = np.unique(np.linspace(0.05 * (count - 1), 0.95 * (count - 1), samples).astype(int))
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = min(1, max_width / gray.shape[1])
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)))
            frames.append(gray)
    finally:
        cap.release()
    return build_background(frames)


BACKGROUND_CACHE_VERSION = 1  # Increment when sampling or feature extraction changes.


def cached_background(path, samples, max_width, cache_dir):
    if cache_dir is None:
        return sample_background(path, samples, max_width), False
    path = Path(path)
    stat = path.stat()
    identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, samples, max_width,
                BACKGROUND_CACHE_VERSION, cv2.__version__, np.__version__]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{key}.npz"
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            image, mask = data["image"], data["mask"]
            points, descriptors = data["points"], data["descriptors"]
            if (image.ndim != 2 or points.ndim != 2 or descriptors.ndim != 2
                    or image.shape != mask.shape or image.dtype != np.uint8
                    or mask.dtype != np.uint8 or points.shape != (len(points), 2)
                    or descriptors.shape != (len(points), 128)):
                raise ValueError("Invalid cached background shapes")
            return Background(image, mask, points, descriptors if len(points) else None,
                              bool(data["usable"].item()), str(data["reason"].item())), True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    background = sample_background(path, samples, max_width)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return background, False  # Do not cache a file still being recorded.
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, image=background.image, mask=background.mask,
                                points=background.points,
                                descriptors=background.descriptors if background.descriptors is not None
                                else np.empty((0, 128), np.float32),
                                usable=background.usable, reason=background.reason)
        os.replace(temporary, cache_path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return background, False


def prepare_background(path, samples, max_width, cache_dir):
    started = perf_counter()
    try:
        background, hit = cached_background(path, samples, max_width, cache_dir)
        return background, hit, perf_counter() - started, None
    except (ValueError, cv2.error, OSError) as error:
        return None, False, perf_counter() - started, str(error)


def prepared_videos(videos, samples, max_width, cache_dir, workers):
    """Bound preparation to workers videos; preserve chronological evaluation."""
    if workers == 1:
        for timestamp, path in videos:
            yield timestamp, path, prepare_background(path, samples, max_width, cache_dir)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        source = iter(videos)
        pending = deque()

        def enqueue():
            item = next(source, None)
            if item is not None:
                timestamp, path = item
                pending.append((timestamp, path, executor.submit(
                    prepare_background, path, samples, max_width, cache_dir)))

        for _ in range(workers):
            enqueue()
        while pending:
            timestamp, path, future = pending.popleft()
            prepared = future.result()
            yield timestamp, path, prepared
            enqueue()


def aligned_structure(reference, current):
    """Check fixed-position edge agreement when repetitive texture defeats SIFT."""
    h, w = reference.image.shape
    target = cv2.resize(current.image, (w, h))
    mask = (reference.mask > 0) & (cv2.resize(current.mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    gradients = []
    for image in (reference.image, target):
        smooth = cv2.GaussianBlur(image, (5, 5), 1.2)
        gradients.append(np.stack([cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                                   cv2.Sobel(smooth, cv2.CV_32F, 0, 1)], axis=-1))
    agreeing, evaluated = [], 0
    for row in range(4):
        for col in range(4):
            region = np.s_[row * h // 4:(row + 1) * h // 4, col * w // 4:(col + 1) * w // 4]
            valid = mask[region]
            if valid.mean() < 0.25:
                continue
            a, b = [g[region][valid] for g in gradients]
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            if min(float(np.mean(a * a)), float(np.mean(b * b))) < 20:
                continue
            evaluated += 1
            correlation = float(np.sum(a * b) / max(norm, 1e-6))
            if correlation >= 0.70:
                agreeing.append((row, col))
    return (len(agreeing) >= 6 and len(agreeing) >= 0.6 * evaluated
            and len({r for r, c in agreeing}) >= 3 and len({c for r, c in agreeing}) >= 3)


def compare_backgrounds(reference, current, shift_fraction=0.015):
    """Return same/change/unknown; change is only evidence for temporal confirmation."""
    result = {"status": "unknown", "reason": "insufficient stable background",
              "matches": 0, "inliers": 0, "displacement_fraction": None}
    if not reference.usable or not current.usable:
        return result
    h, w = reference.image.shape
    ch, cw = current.image.shape
    if abs((w / h) / (cw / ch) - 1) > 0.02:
        return {**result, "status": "change", "reason": "aspect ratio changed"}
    line_metrics, stationary_lines, changed_lines = compare_lines(reference, current)
    result.update(line_metrics)
    if stationary_lines:
        return {**result, "status": "same", "reason": "long structural lines aligned"}

    def lost_correspondence():
        # Wet soil, shadows and seasonal texture changes can destroy descriptors.
        # Loss of matches alone is no longer positive change evidence.
        return {**result, "status": "change" if changed_lines else "unknown",
                "reason": "structural line layout changed" if changed_lines
                else "appearance changed without sufficient geometric evidence"}
    if aligned_structure(reference, current):
        return {**result, "status": "same", "reason": "stable edges aligned across image"}
    # A dense alignment can still verify an unchanged scene when descriptor
    # matching is ambiguous on repeated hive patterns or under new illumination.
    eh, ew = max(32, round(h * min(1, 400 / w))), min(w, 400)
    template = cv2.resize(reference.image, (ew, eh))
    target = cv2.resize(current.image, (ew, eh))
    mask = cv2.bitwise_and(
        cv2.resize(reference.mask, (ew, eh), interpolation=cv2.INTER_NEAREST),
        cv2.resize(current.mask, (ew, eh), interpolation=cv2.INTER_NEAREST))
    try:
        correlation, warp = cv2.findTransformECC(
            template, target, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4), mask, 5)
        probes = np.float32([[ew * x, eh * y] for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)])
        moved = cv2.transform(probes.reshape(-1, 1, 2), warp).reshape(-1, 2)
        displacement = float(np.median(np.linalg.norm(moved - probes, axis=1)) / np.hypot(ew, eh))
        if correlation >= 0.90 and displacement < shift_fraction:
            return {**result, "status": "same", "reason": "dense background alignment verified",
                    "displacement_fraction": displacement}
    except cv2.error:
        pass  # Nonconvergence is not itself evidence of a changed scene.
    if reference.descriptors is None or current.descriptors is None:
        return lost_correspondence()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(reference.descriptors, current.descriptors, k=2)
    backward = matcher.knnMatch(current.descriptors, reference.descriptors, k=2)
    reverse = {(a.trainIdx, a.queryIdx) for pair in backward if len(pair) == 2
               for a, b in [pair] if a.distance < 0.72 * b.distance}
    good = [a for pair in forward if len(pair) == 2 for a, b in [pair]
            if a.distance < 0.72 * b.distance and (a.queryIdx, a.trainIdx) in reverse]
    result["matches"] = len(good)
    if len(good) < 15:
        return lost_correspondence()
    src = np.array([reference.points[m.queryIdx] for m in good], np.float32)
    dst = np.array([current.points[m.trainIdx] for m in good], np.float32)
    dst *= np.array([w / cw, h / ch], np.float32)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return lost_correspondence()
    selected = inliers.ravel().astype(bool)
    result["inliers"] = int(selected.sum())
    if selected.sum() < 15 or selected.mean() < 0.35:
        return lost_correspondence()
    # Avoid trusting one bee cluster or one small patch of repeating hive texture.
    for points in (src[selected], dst[selected]):
        grid = np.clip((points / [w, h] * 4).astype(int), 0, 3)
        coverage = len(np.unique(grid, axis=0))
        area = cv2.contourArea(cv2.convexHull(points)) / (w * h)
        if coverage < 4 or area < 0.08:
            return {**result, "reason": "matches confined to a small region"}
    # Measure displacement over supported locations, not extrapolated corners.
    projected = cv2.perspectiveTransform(src[selected].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    displacement = float(np.median(np.linalg.norm(projected - src[selected], axis=1)) / np.hypot(w, h))
    if not np.isfinite(displacement):
        return result
    return {**result, "status": "change" if displacement >= shift_fraction else "same",
            "reason": "framing moved" if displacement >= shift_fraction else "background aligned",
            "displacement_fraction": displacement}


class ChangeTracker:
    """Keep a fixed segment reference; require consecutive comparable new views."""

    def __init__(self, shift_fraction=0.015, confirmations=2):
        if confirmations < 2:
            raise ValueError("At least two videos must confirm a change")
        self.shift_fraction = shift_fraction
        self.confirmations = confirmations
        self.reference = None
        self.reference_name = None
        self.last_normal = None
        self.pending = None

    def compare(self, a, b):
        return compare_backgrounds(a, b, self.shift_fraction)

    def observe(self, name, background):
        if not background.usable:
            return {"status": "unknown", "reason": background.reason}, None
        if self.reference is None:
            self.reference = background
            self.reference_name = self.last_normal = name
            return {"status": "baseline", "reason": "first usable video"}, None
        result = self.compare(self.reference, background)
        if result["status"] == "same":
            self.last_normal = name
            self.pending = None
            return result, None
        if result["status"] == "unknown":
            return result, None
        if self.pending is not None and self.compare(self.pending["background"], background)["status"] == "same":
            self.pending["count"] += 1
        else:
            self.pending = {"background": background, "name": name, "count": 1,
                            "reason": result["reason"]}
        if self.pending["count"] < self.confirmations:
            return {**result, "status": "pending"}, None
        event = {"last_normal_video": self.last_normal,
                 "changed_from_video": self.pending["name"], "confirmed_at_video": name,
                 "reference_video": self.reference_name, "reason": self.pending["reason"],
                 "confirming_videos": self.pending["count"]}
        old_reference = self.reference
        self.reference = self.pending["background"]
        self.reference_name = self.pending["name"]
        self.last_normal = name
        self.pending = None
        # Return images only transiently, not in the serializable event.
        return {**result, "status": "changed"}, (event, old_reference, self.reference)


def video_identity(path):
    match = re.fullmatch(r"(.+)_(\d{8}_\d{6})", path.stem)
    if not match:
        raise ValueError(f"Expected DEVICE_YYYYMMDD_HHMMSS video name: {path.name}")
    return match[1], datetime.strptime(match[2], "%Y%m%d_%H%M%S")


def save_evidence(path, before, after):
    panels = []
    for label, background in (("BEFORE", before), ("AFTER", after)):
        panel = cv2.cvtColor(background.image, cv2.COLOR_GRAY2BGR)
        # Excluded dynamic pixels are tinted red for inspection.
        excluded = background.mask == 0
        panel[excluded] = (panel[excluded] * 0.4 + np.array([0, 0, 150])).astype(np.uint8)
        for x1, y1, x2, y2 in extract_lines(background.image, background.mask).astype(int):
            cv2.line(panel, (x1, y1), (x2, y2), (255, 255, 0), 2)
        panel = cv2.resize(panel, (640, 480))
        cv2.putText(panel, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise OSError(f"Could not write {path}")


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a valid date: YYYY-MM-DD") from error


def parse_datetime(value):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value.replace("T", " "), fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Expected YYYY-MM-DD HH:MM[:SS] (filename time, no timezone)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--timelapse-dir", type=Path, help="Analyze deviceN.mp4 + deviceN.csv instead of original videos")
    parser.add_argument("--timelapse-ids", nargs="+", type=int, default=list(range(1, 21)),
                        help="Timelapse file numbers; default 1 through 20")
    parser.add_argument("--caption-fraction", type=float, default=0.12,
                        help="Fraction of timelapse height cropped from top to exclude captions")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--device", nargs="+", help="Full device names, e.g. ANU-25-summer-3")
    start_options = parser.add_mutually_exclusive_group()
    start_options.add_argument("--start-date", type=parse_date, help="First recording date, inclusive (YYYY-MM-DD)")
    start_options.add_argument("--start-datetime", type=parse_datetime,
                               help="First recording time, inclusive: YYYY-MM-DD HH:MM[:SS]")
    end_options = parser.add_mutually_exclusive_group()
    end_options.add_argument("--end-date", type=parse_date, help="Last recording date, inclusive (YYYY-MM-DD)")
    end_options.add_argument("--end-datetime", type=parse_datetime,
                             help="Last recording time, inclusive: YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--limit", type=int, help="Maximum videos per device for a trial run")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent video preparation (try 2; comparisons remain ordered)")
    parser.add_argument("--cache-dir", type=Path, default=Path("bee_count_output/scene_background_cache"))
    parser.add_argument("--no-cache", action="store_true", help="Disable reading and writing background cache")
    parser.add_argument("--max-width", type=int, default=800)
    parser.add_argument("--shift-fraction", type=float, default=0.015,
                        help="Change threshold as fraction of image diagonal (default 0.015)")
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    start_at = args.start_datetime or (datetime.combine(args.start_date, time.min) if args.start_date else None)
    end_at = args.end_datetime or (datetime.combine(args.end_date, time.max) if args.end_date else None)
    if start_at and end_at and start_at > end_at:
        parser.error("start must be on or before end")
    if args.samples < 5 or args.max_width < 160 or args.confirmations < 2 or not 0 < args.shift_fraction < 1:
        parser.error("Require samples >= 5, max-width >= 160, confirmations >= 2, 0 < shift-fraction < 1")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.workers < 1:
        parser.error("workers must be positive")
    if not 0 <= args.caption_fraction < 0.5 or any(i < 1 for i in args.timelapse_ids):
        parser.error("Require 0 <= caption-fraction < 0.5 and positive timelapse IDs")
    if args.output_dir is None:
        args.output_dir = Path("bee_count_output/timelapse_scene_changes" if args.timelapse_dir else "bee_count_output/scene_changes")
    if args.timelapse_dir:
        if __package__ in {None, ""}:
            # Direct script execution needs the project root for worker imports.
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.timelapse_scene_changes import run_timelapses
        run_timelapses(args, start_at, end_at)
        return
    groups = {}
    for path in sorted(args.video_dir.glob(args.pattern)):
        device, timestamp = video_identity(path)
        if args.device and device not in args.device:
            continue
        if start_at and timestamp < start_at:
            continue
        if end_at and timestamp > end_at:
            continue
        groups.setdefault(device, []).append((timestamp, path))
    if not groups:
        parser.error("No matching videos")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, unresolved = [], []
    columns = ["device", "video", "timestamp", "status", "reason", "matches", "inliers", "displacement_fraction",
               "reference_lines", "current_lines", "matched_lines", "line_match_ratio",
               "cache_hit", "background_sec", "comparison_sec"]
    with (args.output_dir / "video_checks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for device, videos in sorted(groups.items()):
            tracker = ChangeTracker(args.shift_fraction, args.confirmations)
            selected = sorted(videos)[:args.limit]
            prepared = prepared_videos(selected, args.samples, args.max_width,
                                       None if args.no_cache else args.cache_dir, args.workers)
            for index, (timestamp, path, preparation) in enumerate(prepared, 1):
                background, cache_hit, background_sec, error = preparation
                compare_started = perf_counter()
                try:
                    if error is not None:
                        raise ValueError(error)
                    check, event_data = tracker.observe(path.name, background)
                except (ValueError, cv2.error) as error:
                    check, event_data = {"status": "unknown", "reason": str(error)}, None
                writer.writerow({"device": device, "video": path.name,
                                 "timestamp": timestamp.isoformat(), "cache_hit": cache_hit,
                                 "background_sec": round(background_sec, 4),
                                 "comparison_sec": round(perf_counter() - compare_started, 4), **check})
                stream.flush()
                if event_data is not None:
                    event, before, after = event_data
                    evidence = args.output_dir / f"{Path(event['changed_from_video']).stem}_change.jpg"
                    save_evidence(evidence, before, after)
                    events.append({"device": device, **event, "evidence": str(evidence)})
                print(f"{device} [{index}/{len(selected)}] {path.name}: {check['status']}", flush=True)
            if tracker.pending is not None:
                unresolved.append({"device": device, "video": tracker.pending["name"],
                                   "reason": "not enough subsequent matching videos"})
    report = {"settings": {"samples": args.samples, "max_width": args.max_width,
                            "workers": args.workers,
                            "cache_dir": None if args.no_cache else str(args.cache_dir),
                            "start_date": args.start_date.isoformat() if args.start_date else None,
                            "end_date": args.end_date.isoformat() if args.end_date else None,
                            "start_datetime": start_at.isoformat() if start_at else None,
                            "end_datetime": end_at.isoformat() if end_at else None,
                            "shift_fraction": args.shift_fraction, "confirmations": args.confirmations},
              "changes": events, "unconfirmed": unresolved}
    (args.output_dir / "scene_changes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 기기별 화면 변경 보고", "", "자동 판정 후보이며 정확도는 현장 검증이 필요합니다.", ""]
    for event in events:
        lines.extend([f"- **{event['device']}**: `{event['changed_from_video']}`부터 화면 변경.",
                      f"  마지막 정상: `{event['last_normal_video']}`, 확인 영상: `{event['confirmed_at_video']}`.",
                      f"  [변경 전후 이미지]({Path(event['evidence']).name})", ""])
    if not events:
        lines.append("확정 조건을 충족한 변경 후보가 없습니다. 판단 불가 영상은 video_checks.csv를 확인하세요.")
    for pending in unresolved:
        lines.append(f"- 확인 대기: `{pending['video']}` (후속 비교 근거 부족)")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
