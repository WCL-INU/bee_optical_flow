"""Experimental, color-independent entrance proposals in the lower half of video.

Scores are geometric evidence, not calibrated probabilities. A long hive edge
can resemble an entrance; inspect the overlay before using coordinates to count.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect_frames(frames):
    """Return entrance/ROI proposals from equally sized BGR frames."""
    if len(frames) < 3:
        raise ValueError("At least three readable sample frames are required")
    height, width = frames[0].shape[:2]
    if min(height, width) < 32:
        raise ValueError("Sample images must be at least 32 pixels per side")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("Sample frame sizes differ")
    scale = min(1.0, 800 / width)
    w, h = round(width * scale), round(height * scale)
    half = h // 2
    grays = np.stack([
        cv2.cvtColor(cv2.resize(f, (w, h)), cv2.COLOR_BGR2GRAY)[half:]
        for f in frames
    ])
    background = np.median(grays, axis=0).astype(np.uint8)
    # Remove global brightness changes before estimating local temporal activity.
    centered = grays.astype(np.float32)
    centered -= np.median(centered, axis=(1, 2), keepdims=True)
    activity = np.median(np.abs(centered - np.median(centered, axis=0)), axis=0)
    activity = np.clip(activity - 3, 0, 30) / 30
    edges = cv2.Canny(cv2.GaussianBlur(background, (5, 5), 0), 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(20, w // 12),
        minLineLength=w // 5, maxLineGap=w // 25,
    )
    candidates = []
    band = max(6, round(h * 0.04))
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < w * 0.2 or abs(y2 - y1) > abs(x2 - x1) * 0.12:
                continue
            left, right = sorted((int(x1), int(x2)))
            y = round((int(y1) + int(y2)) / 2)
            top, bottom = max(0, y - band), min(h - half, y + band)
            if bottom - top < band:
                continue
            # An edge must be visible in multiple samples, even without motion.
            supports = []
            for gray in grays:
                gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
                supports.append(float(np.mean(
                    np.max(gradient[max(0, y - 4):y + 5, left:right], axis=0) > 35
                )))
            support = float(np.median(supports))
            motion = float(activity[top:bottom, left:right].mean())
            span = (right - left) / w
            score = 0.50 * span + 0.35 * support + 0.15 * motion
            candidates.append(dict(
                left=left, right=right, y=y, score=score,
                edge_support=support, activity=motion,
            ))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Merge duplicate edges of the same narrow entrance band.
    distinct = []
    for candidate in candidates:
        if all(abs(candidate["y"] - c["y"]) > band * 2 for c in distinct):
            distinct.append(candidate)
    result = {"status": "failed", "reason": "No stable horizontal entrance candidate",
              "roi": None, "entrance": None, "candidates": []}
    for candidate in distinct[:5]:
        left = round(candidate["left"] * width / w)
        right = round(candidate["right"] * width / w)
        top = max((height + 1) // 2, round((candidate["y"] + half - band) * height / h))
        bottom = min(height, round((candidate["y"] + half + band) * height / h))
        result["candidates"].append({**candidate, "entrance": [left, top, right, bottom]})
    if not result["candidates"]:
        return result
    best = result["candidates"][0]
    if best["score"] < 0.48 or best["edge_support"] < 0.45:
        result["reason"] = "Insufficient structural evidence"
        return result
    if len(distinct) > 1 and best["score"] - distinct[1]["score"] < 0.06:
        result["reason"] = "Multiple similar entrance candidates; manual review required"
        return result
    left, top, right, bottom = best["entrance"]
    pad = max(12, round(height * 0.05))
    # Geometry alone cannot distinguish an entrance from a board or pipe edge.
    # Even an unambiguous candidate remains a proposal, never an accepted detection.
    result.update(
        status="needs_review", reason="Geometric proposal only; entrance identity is unverified",
        entrance=best["entrance"],
        roi=[max(0, left - pad), max((height + 1) // 2, top - pad),
             min(width, right + pad), min(height, bottom + pad)],
    )
    return result


def detect_video(video_path, output_dir, samples=12):
    if samples < 3:
        raise ValueError("samples must be at least 3")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 3:
            raise ValueError(f"Not enough video frames: {video_path}")
        for index in np.unique(np.linspace(0, count - 1, samples).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    result = detect_frames(frames)
    result["video"] = str(video_path)
    result["sample_count"] = len(frames)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    overlay = frames[len(frames) // 2].copy()
    for index, candidate in enumerate(result["candidates"]):
        x1, y1, x2, y2 = candidate["entrance"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(overlay, f"candidate {index + 1}: {candidate['score']:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
    if result["roi"]:
        x1, y1, x2, y2 = result["roi"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, result["status"], (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 180, 255), 2)
    overlay_path = output_dir / f"{stem}_entrance_detection.jpg"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Cannot write overlay: {overlay_path}")
    (output_dir / f"{stem}_entrance_detection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("bee_count_output/entrance_detection"))
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    for video in args.videos:
        result = detect_video(video, args.output_dir, args.samples)
        print(f"{video.name}: {result['status']}: {result['reason']}")
        if result["roi"]:
            print("--roi", *result["roi"], "--entrance", *result["entrance"])


if __name__ == "__main__":
    main()
