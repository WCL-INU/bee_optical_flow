"""Analyze source-recording segments of captioned timelapses using CSV mappings."""

import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path, fps, frame_count):
    from src.detect_scene_changes import video_identity
    segments, expected, device, previous = [], 0, None, None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video", "video_start", "status", "output_start_sec", "output_frames"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            name = Path(row["video"].replace("\\", "/")).name
            source_device, _ = video_identity(Path(name))
            stamp = datetime.fromisoformat(row["video_start"])
            if stamp.tzinfo is not None:
                raise ValueError("Manifest timestamps must use filename local time without timezone")
            if device is not None and source_device != device:
                raise ValueError("One timelapse must contain exactly one source device")
            if previous is not None and stamp <= previous:
                raise ValueError(f"Non-increasing source timestamps at CSV line {line}")
            device, previous = source_device, stamp
            offset = float(row["output_start_sec"]) * fps
            count = int(row["output_frames"])
            if not math.isfinite(offset) or abs(offset - round(offset)) > 0.05 or round(offset) != expected:
                raise ValueError(f"Non-contiguous or invalid frame mapping at CSV line {line}")
            if row["status"] == "skipped":
                if count != 0:
                    raise ValueError(f"Skipped row has frames at CSV line {line}")
                continue
            if row["status"] != "written" or count <= 0:
                raise ValueError(f"Invalid segment at CSV line {line}")
            segments.append({"video": name, "timestamp": stamp, "frame_start": expected,
                             "frame_count": count, "timelapse_start_sec": float(row["output_start_sec"]),
                             "first_frame_time": row.get("first_frame_time", ""),
                             "last_frame_time": row.get("last_frame_time", ""),
                             "sampling": row.get("sampling", "legacy")})
            expected += count
    if expected != frame_count:
        raise ValueError(f"CSV describes {expected} frames but MP4 contains {frame_count}")
    if not segments:
        raise ValueError("No written source segments")
    return device, segments


def segment_background(cap, segment, max_width, samples, caption_fraction):
    from src.detect_scene_changes import build_background
    start, count = segment["frame_start"], segment["frame_count"]
    if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError(f"Cannot seek to timelapse frame {start}")
    selected = set(np.linspace(0, count - 1, min(samples, count)).round().astype(int))
    frames = []
    for offset in range(count):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Decode failed at timelapse frame {start + offset}; mapping cannot be trusted")
        if offset in selected:
            # Crop before measuring motion or extracting features: captions themselves change.
            image = image[math.ceil(image.shape[0] * caption_fraction):]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scale = min(1, max_width / gray.shape[1])
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), max(2, round(gray.shape[0] * scale))))
            frames.append(gray)
    return build_background(frames)


def analyze_timelapse(job):
    from src.detect_scene_changes import ChangeTracker, save_evidence
    # Each worker processes an independent device; limit nested OpenCV threads.
    cv2.setNumThreads(1)
    path = Path(job["path"])
    out = Path(job["output_dir"]) / path.stem
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid timelapse FPS")
        device, segments = read_manifest(path.with_suffix(".csv"), fps, round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if job["devices"] and device not in job["devices"]:
            return {"input": str(path), "status": "excluded", "device": device}
        selected = [s for s in segments if (job["start_at"] is None or s["timestamp"] >= job["start_at"])
                    and (job["end_at"] is None or s["timestamp"] <= job["end_at"])]
        selected = selected[:job["limit"]]
        lookup = {s["video"]: s for s in segments}
        tracker = ChangeTracker(job["shift_fraction"], job["confirmations"])
        events, unknown, processed = [], 0, 0
        columns = ["device", "video", "timestamp", "timelapse_start_sec", "frame_start", "frame_count",
                   "first_frame_time", "last_frame_time", "sampling", "status", "reason", "matches", "inliers",
                   "displacement_fraction", "reference_lines", "current_lines", "matched_lines", "line_match_ratio"]
        with (out / "video_checks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for s in selected:
                if s["frame_count"] < 5:
                    check, event_data = {"status": "unknown", "reason": "fewer than five output frames"}, None
                else:
                    bg = segment_background(cap, s, job["max_width"], job["samples"], job["caption_fraction"])
                    check, event_data = tracker.observe(s["video"], bg)
                writer.writerow({**s, "timestamp": s["timestamp"].isoformat(), "device": device, **check})
                stream.flush()
                processed += 1
                unknown += check["status"] == "unknown"
                if event_data:
                    event, before, after = event_data
                    changed = lookup[event["changed_from_video"]]
                    evidence = out / f"{Path(changed['video']).stem}_change.jpg"
                    save_evidence(evidence, before, after)
                    events.append({"device": device, **event,
                                   "changed_from_timestamp": changed["timestamp"].isoformat(),
                                   "last_normal_timestamp": lookup[event["last_normal_video"]]["timestamp"].isoformat(),
                                   "confirmed_at_timestamp": s["timestamp"].isoformat(),
                                   "timelapse_start_sec": changed["timelapse_start_sec"],
                                   "evidence": str(evidence)})
                if processed % 50 == 0:
                    print(f"{path.stem}: {processed}/{len(selected)} segments", flush=True)
        pending = tracker.pending["name"] if tracker.pending else None
        result = {"input": str(path), "manifest": str(path.with_suffix('.csv')), "status": "ok", "device": device,
                  "processed": processed, "unknown": unknown, "changes": events, "unconfirmed": pending,
                  "caption_fraction": job["caption_fraction"], "report": str(out / "report.md")}
        (out / "scene_changes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [f"# {device} 타임랩스 구도 변경", "", f"검사 {processed}개 / 판단 불가 {unknown}개 / 변경 후보 {len(events)}건", "",
                 "원본 촬영 일시는 CSV에서 복원했습니다. 타임랩스의 짧은 구간만 분석한 자동 판정입니다.", ""]
        for event in events:
            lines += [f"- **{event['changed_from_timestamp']}부터 변경 후보** (타임랩스 {event['timelapse_start_sec']:.2f}초)",
                      f"  마지막 정상 {event['last_normal_timestamp']} / 확인 {event['confirmed_at_timestamp']}",
                      f"  [전후 이미지]({Path(event['evidence']).name})", ""]
        if not events:
            lines.append("변경 확인 조건을 충족한 후보가 없습니다. 판단 불가 구간은 video_checks.csv를 확인하세요.")
        if pending:
            lines.append(f"후속 확인 대기: {pending}")
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{path.stem}: {processed} checked, {len(events)} changes, {unknown} unknown", flush=True)
        return result
    except (ValueError, OSError, cv2.error, OverflowError) as error:
        failure = {"input": str(path), "status": "error", "error": str(error)}
        (out / "error.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path.stem}: ERROR {error}", flush=True)
        return failure
    finally:
        cap.release()


def run_timelapses(args, start_at, end_at):
    jobs = []
    for number in sorted(set(args.timelapse_ids)):
        jobs.append(dict(path=str(args.timelapse_dir / f"device{number}.mp4"), output_dir=str(args.output_dir),
                         devices=args.device, start_at=start_at, end_at=end_at, limit=args.limit,
                         samples=args.samples, max_width=args.max_width, caption_fraction=args.caption_fraction,
                         shift_fraction=args.shift_fraction, confirmations=args.confirmations))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = [analyze_timelapse(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(analyze_timelapse, jobs))
    report = {"settings": {"input_mode": "timelapse", "limit_per_device": args.limit,
                            "start_datetime": start_at.isoformat() if start_at else None,
                            "end_datetime": end_at.isoformat() if end_at else None,
                            "samples": args.samples, "max_width": args.max_width,
                            "caption_fraction": args.caption_fraction, "shift_fraction": args.shift_fraction,
                            "confirmations": args.confirmations}, "devices": results}
    (args.output_dir / "scene_changes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 타임랩스 구도 변경 종합 보고", "", "CSV의 원본 촬영 일시 기준입니다. 짧은 영상 구간 분석으로 인한 판단 불가/오탐이 있을 수 있습니다.", "",
             "| 입력 | 검사 구간 | 변경 후보 | 판단 불가 | 결과 |", "|---|---:|---:|---:|---|"]
    for result in results:
        stem = Path(result["input"]).stem
        if result["status"] == "ok":
            lines.append(f"| {stem} | {result['processed']} | {len(result['changes'])} | {result['unknown']} | [보고서]({stem}/report.md) |")
        else:
            lines.append(f"| {stem} | — | — | — | {result.get('error', '제외')} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.output_dir / 'report.md'}")
    if any(result["status"] == "error" for result in results):
        raise SystemExit(1)
