"""Reproducible Raspberry Pi benchmark for the bee optical-flow pipeline.

Two modes are provided:

``offline``
    Read a video as fast as possible.  This measures sustainable throughput and
    includes H.264 decoding in the end-to-end latency.

``realtime``
    A producer replays the video at its recorded frame rate into a one-frame
    latest-image queue.  This approximates a live camera consumer and exposes
    deadline misses, queue delay, and dropped frames.

The primary benchmark is headless.  ``--preview`` can additionally quantify the
cost of the debug overlay and MP4 writer.  Raw per-frame timings are retained so
that paper figures and alternative statistics can be regenerated later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import random
import resource
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bee_optical_flow_matplotlib")

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bee_entrance_count import (
    apply_bidirectional_balance_filter,
    apply_component_area_filter,
    build_counting_boundary_band,
    build_entrance_mask,
    clamp_rect,
    compensate_global_flow,
    compute_optical_flow,
    compute_raw_flux,
    crop_roi,
    draw_preview,
    load_video_info,
    prepare_gray,
    update_persistence_filter,
)
from src.main import PRESETS, find_coordinate_preset, resolve_config_for_video


FRAME_COLUMNS = [
    "mode",
    "video",
    "repeat",
    "frame",
    "frame_gap",
    "source_fps",
    "budget_ms",
    "decode_ms",
    "queue_ms",
    "preprocess_ms",
    "flow_ms",
    "postprocess_ms",
    "preview_ms",
    "processing_ms",
    "end_to_end_ms",
    "processing_deadline_miss",
    "end_to_end_deadline_miss",
    "raw_in_flux",
    "raw_out_flux",
    "filtered_in_flux",
    "filtered_out_flux",
    "raw_candidate_pixels",
    "filtered_candidate_pixels",
]


@dataclass
class VideoState:
    roi_rect: tuple[int, int, int, int]
    entrance_rect: tuple[int, int, int, int]
    counting_boundary_band: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    background_mask: np.ndarray
    persistence: np.ndarray
    previous_gray: np.ndarray | None = None
    previous_frame_index: int | None = None


@dataclass
class Sample:
    monotonic_sec: float
    temperature_c: float | None
    frequency_mhz: float | None
    rss_mib: float


class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = interval_sec
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._sample()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()

    def _run(self):
        while not self._stop.wait(self.interval_sec):
            self._sample()

    def _sample(self):
        self.samples.append(
            Sample(
                monotonic_sec=time.perf_counter(),
                temperature_c=read_temperature_c(),
                frequency_mhz=read_frequency_mhz(),
                rss_mib=read_rss_mib(),
            )
        )


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def read_temperature_c() -> float | None:
    value = read_text("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(value) / 1000.0 if value is not None else None
    except ValueError:
        return None


def read_frequency_mhz() -> float | None:
    values = []
    for cpu in range(os.cpu_count() or 1):
        value = read_text(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        )
        try:
            if value is not None:
                values.append(float(value) / 1000.0)
        except ValueError:
            continue
    return float(np.mean(values)) if values else None


def read_rss_mib() -> float:
    status = read_text("/proc/self/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def read_cpu_counters() -> tuple[int, int] | None:
    first = (read_text("/proc/stat") or "").splitlines()
    if not first or not first[0].startswith("cpu "):
        return None
    values = [int(value) for value in first[0].split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def system_cpu_percent(
    start: tuple[int, int] | None, end: tuple[int, int] | None
) -> float | None:
    if start is None or end is None:
        return None
    total = end[0] - start[0]
    idle = end[1] - start[1]
    return 100.0 * (total - idle) / total if total > 0 else None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(args, videos: list[Path]) -> dict:
    governors = sorted(
        {
            value
            for cpu in range(os.cpu_count() or 1)
            if (
                value := read_text(
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                )
            )
        }
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": vars(args),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "logical_cpu_count": os.cpu_count(),
        "opencv_threads": cv2.getNumThreads(),
        "opencl_available": bool(cv2.ocl.haveOpenCL()),
        "opencl_enabled": bool(cv2.ocl.useOpenCL()),
        "cpu_governors": governors,
        "temperature_c_at_start": read_temperature_c(),
        "frequency_mhz_at_start": read_frequency_mhz(),
        "vcgencmd_get_throttled_at_start": command_output(
            ["vcgencmd", "get_throttled"]
        ),
        "uname": command_output(["uname", "-a"]),
        "lscpu": command_output(["lscpu"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "opencv_build_information": cv2.getBuildInformation(),
        "inputs": [
            {
                "path": str(video),
                "bytes": video.stat().st_size,
                "sha256": sha256_file(video),
            }
            for video in videos
        ],
    }


def finalize_environment(metadata: dict):
    metadata["temperature_c_at_end"] = read_temperature_c()
    metadata["frequency_mhz_at_end"] = read_frequency_mhz()
    metadata["vcgencmd_get_throttled_at_end"] = command_output(
        ["vcgencmd", "get_throttled"]
    )


def prepare_state(frame: np.ndarray, config, video_info: dict) -> VideoState:
    roi_rect = clamp_rect(
        (config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2),
        video_info["width"],
        video_info["height"],
        "ROI",
    )
    roi = crop_roi(frame, roi_rect)
    entrance_rect = (
        config.ent_x1 - roi_rect[0],
        config.ent_y1 - roi_rect[1],
        config.ent_x2 - roi_rect[0],
        config.ent_y2 - roi_rect[1],
    )
    entrance_mask = build_entrance_mask(roi.shape, entrance_rect)
    band, normal_x, normal_y = build_counting_boundary_band(
        entrance_mask, entrance_rect, config
    )
    return VideoState(
        roi_rect=roi_rect,
        entrance_rect=entrance_rect,
        counting_boundary_band=band,
        normal_x=normal_x,
        normal_y=normal_y,
        background_mask=entrance_mask == 0,
        persistence=np.zeros(roi.shape[:2], dtype=np.float32),
    )


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    state: VideoState,
    config,
    source_fps: float,
    writer: cv2.VideoWriter | None,
) -> tuple[dict | None, dict | None]:
    preprocess_start = time.perf_counter_ns()
    roi = crop_roi(frame, state.roi_rect)
    gray = prepare_gray(roi, config)
    preprocess_end = time.perf_counter_ns()

    if state.previous_gray is None:
        state.previous_gray = gray
        state.previous_frame_index = frame_index
        return None, None

    flow_start = time.perf_counter_ns()
    flow = compute_optical_flow(state.previous_gray, gray)
    flow_end = time.perf_counter_ns()

    post_start = time.perf_counter_ns()
    if config.use_global_flow_compensation:
        flow, _, _ = compensate_global_flow(flow, state.background_mask)
    raw_data = compute_raw_flux(
        flow,
        state.counting_boundary_band,
        state.normal_x,
        state.normal_y,
        config,
    )
    persistent_candidate, state.persistence = update_persistence_filter(
        raw_data["candidate"], state.persistence, config
    )
    filtered_candidate, _ = apply_component_area_filter(
        persistent_candidate, config
    )
    normal_flow = raw_data["normal_flow"]
    filtered_in_flux = float(
        np.sum(np.clip(normal_flow[filtered_candidate], 0, None))
    )
    filtered_out_flux = float(
        np.sum(np.clip(-normal_flow[filtered_candidate], 0, None))
    )
    filtered_in_flux, filtered_out_flux = apply_bidirectional_balance_filter(
        filtered_in_flux, filtered_out_flux, config
    )
    raw_pixels = int(np.count_nonzero(raw_data["candidate"]))
    filtered_pixels = int(np.count_nonzero(filtered_candidate))
    persistence_max = float(np.max(state.persistence))
    post_end = time.perf_counter_ns()

    preview_ms = 0.0
    if writer is not None and frame_index % max(1, config.preview_stride) == 0:
        preview_start = time.perf_counter_ns()
        preview_row = {
            "raw_in_flux": raw_data["raw_in_flux"],
            "raw_out_flux": raw_data["raw_out_flux"],
            "filtered_in_flux": filtered_in_flux,
            "filtered_out_flux": filtered_out_flux,
            "raw_candidate_pixels": raw_pixels,
            "filtered_candidate_pixels": filtered_pixels,
            "persistence_max": persistence_max,
        }
        writer.write(
            draw_preview(
                roi,
                state.entrance_rect,
                state.counting_boundary_band,
                raw_data["candidate"],
                filtered_candidate,
                raw_data,
                preview_row,
                frame_index / source_fps,
                config,
            )
        )
        preview_ms = (time.perf_counter_ns() - preview_start) / 1e6

    previous_index = state.previous_frame_index
    state.previous_gray = gray
    state.previous_frame_index = frame_index
    measurements = {
        "frame_gap": frame_index - int(previous_index),
        "preprocess_ms": (preprocess_end - preprocess_start) / 1e6,
        "flow_ms": (flow_end - flow_start) / 1e6,
        "postprocess_ms": (post_end - post_start) / 1e6,
        "preview_ms": preview_ms,
        "raw_in_flux": raw_data["raw_in_flux"],
        "raw_out_flux": raw_data["raw_out_flux"],
        "filtered_in_flux": filtered_in_flux,
        "filtered_out_flux": filtered_out_flux,
        "raw_candidate_pixels": raw_pixels,
        "filtered_candidate_pixels": filtered_pixels,
    }
    measurements["processing_ms"] = sum(
        measurements[name]
        for name in ["preprocess_ms", "flow_ms", "postprocess_ms", "preview_ms"]
    )
    return measurements, raw_data


def open_writer(
    output_dir: Path,
    video: Path,
    mode: str,
    repeat: int,
    frame: np.ndarray,
    config,
    source_fps: float,
    state: VideoState,
    enabled: bool,
) -> cv2.VideoWriter | None:
    if not enabled:
        return None
    roi = crop_roi(frame, state.roi_rect)
    height, width = roi.shape[:2]
    path = output_dir / "preview" / f"{video.stem}_{mode}_r{repeat}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, source_fps / max(1, config.preview_stride)),
        (width + max(1, int(config.preview_panel_width)), height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open preview writer: {path}")
    return writer


def monitored_run_start():
    monitor = ResourceMonitor()
    monitor.start()
    return {
        "monitor": monitor,
        "wall_start": time.perf_counter(),
        "process_cpu_start": time.process_time(),
        "system_cpu_start": read_cpu_counters(),
    }


def monitored_run_end(run_clock: dict) -> dict:
    wall_end = time.perf_counter()
    process_cpu_end = time.process_time()
    system_end = read_cpu_counters()
    run_clock["monitor"].stop()
    samples = run_clock["monitor"].samples
    temperatures = [s.temperature_c for s in samples if s.temperature_c is not None]
    frequencies = [s.frequency_mhz for s in samples if s.frequency_mhz is not None]
    rss_values = [s.rss_mib for s in samples]
    wall_sec = wall_end - run_clock["wall_start"]
    return {
        "wall_time_sec": wall_sec,
        "process_cpu_time_sec": process_cpu_end - run_clock["process_cpu_start"],
        "process_cpu_percent_one_core": (
            100.0
            * (process_cpu_end - run_clock["process_cpu_start"])
            / max(wall_sec, 1e-9)
        ),
        "system_cpu_percent": system_cpu_percent(
            run_clock["system_cpu_start"], system_end
        ),
        "temperature_start_c": temperatures[0] if temperatures else None,
        "temperature_end_c": temperatures[-1] if temperatures else None,
        "temperature_max_c": max(temperatures) if temperatures else None,
        "frequency_mean_mhz": float(np.mean(frequencies)) if frequencies else None,
        "frequency_min_mhz": min(frequencies) if frequencies else None,
        "rss_max_mib": max(rss_values) if rss_values else None,
    }


def add_common_row_fields(
    row: dict,
    mode: str,
    video: Path,
    repeat: int,
    frame_index: int,
    source_fps: float,
    decode_ms: float,
    queue_ms: float,
    end_to_end_ms: float,
):
    budget_ms = 1000.0 / source_fps
    row.update(
        {
            "mode": mode,
            "video": video.name,
            "repeat": repeat,
            "frame": frame_index,
            "source_fps": source_fps,
            "budget_ms": budget_ms,
            "decode_ms": decode_ms,
            "queue_ms": queue_ms,
            "end_to_end_ms": end_to_end_ms,
            "processing_deadline_miss": row["processing_ms"] > budget_ms,
            "end_to_end_deadline_miss": end_to_end_ms > budget_ms,
        }
    )


def run_offline(
    video: Path,
    repeat_index: int,
    config,
    output_dir: Path,
    warmup_pairs: int,
    max_frames: int | None,
    preview: bool,
    start_frame: int,
) -> tuple[list[dict], dict]:
    cap, info = load_video_info(video)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_fps = float(info["fps"])
    rows: list[dict] = []
    state = None
    writer = None
    measured_clock = None
    frames_read = 0
    pairs_seen = 0
    try:
        while max_frames is None or frames_read < max_frames:
            if measured_clock is None and frames_read == warmup_pairs + 1:
                measured_clock = monitored_run_start()
            decode_start = time.perf_counter_ns()
            ok, frame = cap.read()
            decode_end = time.perf_counter_ns()
            if not ok:
                break
            frame_index = start_frame + frames_read
            frames_read += 1
            if state is None:
                state = prepare_state(frame, config, info)
                writer = open_writer(
                    output_dir,
                    video,
                    "offline",
                    repeat_index,
                    frame,
                    config,
                    source_fps,
                    state,
                    preview,
                )
            measurements, _ = process_frame(
                frame, frame_index, state, config, source_fps, writer
            )
            if measurements is None:
                continue
            pairs_seen += 1
            if pairs_seen <= warmup_pairs:
                continue
            decode_ms = (decode_end - decode_start) / 1e6
            end_to_end_ms = decode_ms + measurements["processing_ms"]
            add_common_row_fields(
                measurements,
                "offline",
                video,
                repeat_index,
                frame_index,
                source_fps,
                decode_ms,
                0.0,
                end_to_end_ms,
            )
            rows.append(measurements)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not rows or measured_clock is None:
        raise RuntimeError(f"No measured frames for {video}")
    resource_summary = monitored_run_end(measured_clock)
    resource_summary.update(
        {
            "source_frames": frames_read,
            "dropped_frames": 0,
            "producer_schedule_late_frames": 0,
            "segment_start_frame": start_frame,
            "segment_end_frame": start_frame + frames_read - 1,
        }
    )
    return rows, resource_summary


def run_realtime(
    video: Path,
    repeat_index: int,
    config,
    output_dir: Path,
    warmup_pairs: int,
    max_frames: int | None,
    preview: bool,
    start_frame: int,
) -> tuple[list[dict], dict]:
    cap, info = load_video_info(video)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_fps = float(info["fps"])
    period_sec = 1.0 / source_fps
    latest: queue.Queue = queue.Queue(maxsize=1)
    producer_done = threading.Event()
    producer_error: list[BaseException] = []
    producer_stats = {
        "source_frames": 0,
        "dropped_frames": 0,
        "producer_schedule_late_frames": 0,
    }
    start_event = threading.Event()
    timing = {"start": 0.0}

    def producer():
        try:
            start_event.wait()
            frame_index = 0
            while max_frames is None or frame_index < max_frames:
                scheduled = timing["start"] + frame_index * period_sec
                remaining = scheduled - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    producer_stats["producer_schedule_late_frames"] += 1
                decode_start = time.perf_counter_ns()
                ok, frame = cap.read()
                decoded_at = time.perf_counter()
                decode_ms = (time.perf_counter_ns() - decode_start) / 1e6
                if not ok:
                    break
                item = (
                    start_frame + frame_index,
                    frame,
                    scheduled,
                    decoded_at,
                    decode_ms,
                )
                if latest.full():
                    try:
                        latest.get_nowait()
                        producer_stats["dropped_frames"] += 1
                    except queue.Empty:
                        pass
                latest.put_nowait(item)
                producer_stats["source_frames"] += 1
                frame_index += 1
        except BaseException as exc:  # propagate thread failures to the caller
            producer_error.append(exc)
        finally:
            cap.release()
            producer_done.set()

    thread = threading.Thread(target=producer, name="paced-video-source", daemon=True)
    timing["start"] = time.perf_counter() + 0.05
    thread.start()
    measured_clock = None
    start_event.set()
    rows: list[dict] = []
    state = None
    writer = None
    pairs_seen = 0
    try:
        while not producer_done.is_set() or not latest.empty():
            try:
                frame_index, frame, scheduled, decoded_at, decode_ms = latest.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue
            process_start = time.perf_counter()
            if state is None:
                state = prepare_state(frame, config, info)
                writer = open_writer(
                    output_dir,
                    video,
                    "realtime",
                    repeat_index,
                    frame,
                    config,
                    source_fps,
                    state,
                    preview,
                )
            if measured_clock is None and state.previous_gray is not None and pairs_seen >= warmup_pairs:
                measured_clock = monitored_run_start()
            measurements, _ = process_frame(
                frame, frame_index, state, config, source_fps, writer
            )
            if measurements is None:
                continue
            pairs_seen += 1
            if pairs_seen <= warmup_pairs:
                continue
            finished = time.perf_counter()
            queue_ms = max(0.0, (process_start - decoded_at) * 1000.0)
            end_to_end_ms = (finished - scheduled) * 1000.0
            add_common_row_fields(
                measurements,
                "realtime",
                video,
                repeat_index,
                frame_index,
                source_fps,
                decode_ms,
                queue_ms,
                end_to_end_ms,
            )
            rows.append(measurements)
    finally:
        thread.join(timeout=5.0)
        if writer is not None:
            writer.release()
    if measured_clock is None:
        raise RuntimeError(f"No measurement clock started for {video}")
    resources = monitored_run_end(measured_clock)
    if producer_error:
        raise RuntimeError("Paced source failed") from producer_error[0]
    if not rows:
        raise RuntimeError(f"No measured frames for {video}")
    resources.update(producer_stats)
    resources["segment_start_frame"] = start_frame
    resources["segment_end_frame"] = (
        start_frame + int(producer_stats["source_frames"]) - 1
    )
    return rows, resources


def percentile(series: pd.Series, value: float) -> float:
    return float(series.quantile(value))


def summarize_run(rows: list[dict], resources: dict, config, video: Path) -> dict:
    frame_df = pd.DataFrame(rows)
    first = frame_df.iloc[0]
    budget_ms = float(first["budget_ms"])
    wall_time = float(resources["wall_time_sec"])
    processed = len(frame_df)
    info_cap, video_info = load_video_info(video)
    info_cap.release()
    summary = {
        "mode": first["mode"],
        "video": first["video"],
        "repeat": int(first["repeat"]),
        "source_fps": float(first["source_fps"]),
        "frame_budget_ms": budget_ms,
        "width": int(video_info["width"]),
        "height": int(video_info["height"]),
        "roi_width": config.roi_x2 - config.roi_x1,
        "roi_height": config.roi_y2 - config.roi_y1,
        "processed_pairs": processed,
        "wall_time_sec": wall_time,
        "achieved_fps": processed / max(wall_time, 1e-9),
        "realtime_factor": (processed / max(wall_time, 1e-9))
        / float(first["source_fps"]),
        "dropped_frames": int(resources["dropped_frames"]),
        "drop_rate_pct": 100.0
        * int(resources["dropped_frames"])
        / max(int(resources["source_frames"]), 1),
        "producer_schedule_late_frames": int(
            resources["producer_schedule_late_frames"]
        ),
        "processing_deadline_miss_pct": 100.0
        * float(frame_df["processing_deadline_miss"].mean()),
        "end_to_end_deadline_miss_pct": 100.0
        * float(frame_df["end_to_end_deadline_miss"].mean()),
        "total_filtered_flux_checksum": float(
            frame_df["filtered_in_flux"].sum() + frame_df["filtered_out_flux"].sum()
        ),
    }
    for column in [
        "decode_ms",
        "queue_ms",
        "preprocess_ms",
        "flow_ms",
        "postprocess_ms",
        "preview_ms",
        "processing_ms",
        "end_to_end_ms",
    ]:
        summary[f"{column}_mean"] = float(frame_df[column].mean())
        summary[f"{column}_median"] = percentile(frame_df[column], 0.50)
        summary[f"{column}_p95"] = percentile(frame_df[column], 0.95)
        summary[f"{column}_p99"] = percentile(frame_df[column], 0.99)
        summary[f"{column}_max"] = float(frame_df[column].max())
    summary.update(resources)
    no_drop = summary["dropped_frames"] == 0
    summary["basic_realtime_pass"] = bool(
        no_drop and summary["end_to_end_ms_p95"] <= budget_ms
    )
    summary["recommended_headroom_pass"] = bool(
        no_drop and summary["end_to_end_ms_p95"] <= 0.8 * budget_ms
    )
    return summary


def bootstrap_mean_ci(values: np.ndarray, seed: int, iterations: int = 10000):
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def aggregate_runs(run_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for group_index, ((mode, video), group) in enumerate(
        run_df.groupby(["mode", "video"], sort=True)
    ):
        latency = group["end_to_end_ms_p95"].to_numpy(dtype=float)
        fps = group["achieved_fps"].to_numpy(dtype=float)
        latency_ci = bootstrap_mean_ci(latency, seed + group_index * 2)
        fps_ci = bootstrap_mean_ci(fps, seed + group_index * 2 + 1)
        rows.append(
            {
                "mode": mode,
                "video": video,
                "n_runs": len(group),
                "source_fps": float(group["source_fps"].iloc[0]),
                "frame_budget_ms": float(group["frame_budget_ms"].iloc[0]),
                "achieved_fps_mean": float(fps.mean()),
                "achieved_fps_sd": float(fps.std(ddof=1)) if len(fps) > 1 else 0.0,
                "achieved_fps_ci95_low": fps_ci[0],
                "achieved_fps_ci95_high": fps_ci[1],
                "end_to_end_p95_ms_mean": float(latency.mean()),
                "end_to_end_p95_ms_sd": (
                    float(latency.std(ddof=1)) if len(latency) > 1 else 0.0
                ),
                "end_to_end_p95_ms_ci95_low": latency_ci[0],
                "end_to_end_p95_ms_ci95_high": latency_ci[1],
                "processing_p95_ms_mean": float(group["processing_ms_p95"].mean()),
                "deadline_miss_pct_mean": float(
                    group["end_to_end_deadline_miss_pct"].mean()
                ),
                "drop_rate_pct_mean": float(group["drop_rate_pct"].mean()),
                "temperature_max_c": float(group["temperature_max_c"].max()),
                "rss_max_mib": float(group["rss_max_mib"].max()),
                "all_basic_realtime_pass": bool(group["basic_realtime_pass"].all()),
                "all_recommended_headroom_pass": bool(
                    group["recommended_headroom_pass"].all()
                ),
            }
        )
    return pd.DataFrame(rows)


def pooled_summary(frame_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, frames in frame_df.groupby("mode", sort=True):
        runs = run_df[run_df["mode"] == mode]
        source_frames = int(runs["source_frames"].sum())
        dropped_frames = int(runs["dropped_frames"].sum())
        row = {
            "mode": mode,
            "segments": len(runs),
            "measured_frame_pairs": len(frames),
            "source_frames": source_frames,
            "dropped_frames": dropped_frames,
            "drop_rate_pct": 100.0 * dropped_frames / max(source_frames, 1),
            "achieved_fps_run_mean": float(runs["achieved_fps"].mean()),
            "achieved_fps_run_sd": float(runs["achieved_fps"].std(ddof=1)),
            "deadline_miss_pct": 100.0
            * float(frames["end_to_end_deadline_miss"].mean()),
            "frame_gap_mean": float(frames["frame_gap"].mean()),
            "frame_gap_p95": percentile(frames["frame_gap"], 0.95),
            "temperature_max_c": float(runs["temperature_max_c"].max()),
            "frequency_min_mhz": float(runs["frequency_min_mhz"].min()),
            "rss_max_mib": float(runs["rss_max_mib"].max()),
        }
        for column in [
            "decode_ms",
            "preprocess_ms",
            "flow_ms",
            "postprocess_ms",
            "processing_ms",
            "end_to_end_ms",
        ]:
            row[f"{column}_mean"] = float(frames[column].mean())
            row[f"{column}_p95"] = percentile(frames[column], 0.95)
            row[f"{column}_p99"] = percentile(frames[column], 0.99)
        row["required_throughput_speedup"] = float(
            frames["source_fps"].median() / row["achieved_fps_run_mean"]
        )
        row["p95_latency_budget_ratio"] = float(
            row["end_to_end_ms_p95"] / frames["budget_ms"].median()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_plots(frame_df: pd.DataFrame, run_df: pd.DataFrame, output_dir: Path):
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    stages = ["decode_ms_mean", "preprocess_ms_mean", "flow_ms_mean", "postprocess_ms_mean"]
    stage_df = (
        run_df.groupby(["mode", "video"], as_index=False)[
            [*stages, "frame_budget_ms"]
        ]
        .mean()
        .sort_values(["mode", "video"])
    )
    labels = [f"{row.mode}\n{Path(row.video).stem}" for row in stage_df.itertuples()]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    bottom = np.zeros(len(stage_df))
    for stage, color in zip(stages, colors):
        values = stage_df[stage].to_numpy(dtype=float)
        ax.bar(labels, values, bottom=bottom, label=stage.removesuffix("_ms_mean"), color=color)
        bottom += values
    budgets = stage_df["frame_budget_ms"].to_numpy(dtype=float)
    ax.plot(labels, budgets, "k--", label="frame budget")
    ax.set_ylabel("Mean time per processed frame (ms)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_breakdown.png", dpi=300)
    plt.close(fig)

    groups = []
    group_labels = []
    for (mode, video), group in frame_df.groupby(["mode", "video"], sort=True):
        groups.append(group["end_to_end_ms"].to_numpy(dtype=float))
        group_labels.append(f"{mode}\n{Path(video).stem}")
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.1), 5))
    ax.boxplot(groups, tick_labels=group_labels, showfliers=False)
    budget = float(frame_df["budget_ms"].median())
    ax.axhline(budget, color="red", linestyle="--", label=f"deadline ({budget:.2f} ms)")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "latency_distribution.png", dpi=300)
    plt.close(fig)

    realtime = frame_df[frame_df["mode"] == "realtime"]
    if not realtime.empty:
        fig, axes = plt.subplots(
            realtime["video"].nunique(),
            1,
            figsize=(10, 3 * realtime["video"].nunique()),
            squeeze=False,
        )
        for ax, (video, group) in zip(axes[:, 0], realtime.groupby("video", sort=True)):
            for repeat, segment in group.groupby("repeat", sort=True):
                ax.plot(
                    segment["frame"] / segment["source_fps"],
                    segment["end_to_end_ms"],
                    linewidth=0.7,
                    label=f"segment {repeat}",
                )
            ax.axhline(float(group["budget_ms"].iloc[0]), color="red", linestyle="--")
            ax.set_title(video)
            ax.set_ylabel("Latency (ms)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("Source time (s)")
        fig.tight_layout()
        fig.savefig(plot_dir / "realtime_latency_timeline.png", dpi=300)
        plt.close(fig)


def format_bool(value) -> str:
    return "PASS" if bool(value) else "FAIL"


def write_report(
    output_dir: Path,
    args,
    frame_df: pd.DataFrame,
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    metadata: dict,
):
    lines = [
        "# Raspberry Pi optical-flow real-time benchmark",
        "",
        f"Generated: {metadata['created_utc']}",
        "",
        "## Experimental design",
        "",
        f"- Input: {len(metadata['inputs'])} recorded Raspberry Pi camera videos",
        f"- Modes: {', '.join(args.modes)}",
        f"- Temporal segments per video/mode: {args.repeats}",
        f"- Frames requested per segment: {args.max_frames or 'entire video'}",
        f"- Segment starts distributed across each video: {args.stratify_starts}",
        f"- Warm-up frame pairs excluded per run: {args.warmup_pairs}",
        f"- OpenCV worker threads: {args.opencv_threads}",
        f"- Optical-flow preset: `{args.preset}`",
        f"- Preview encoding: {args.preview}",
        f"- Queue policy in realtime mode: latest frame, capacity 1",
        "- Timing clock: Python `perf_counter_ns` (monotonic high-resolution clock)",
        "- Confidence intervals: deterministic segment-level non-parametric bootstrap, 10,000 resamples",
        "",
        "The basic criterion is zero dropped frames and p95 end-to-end latency no greater than the recorded frame period. The recommended criterion reserves 20% of the frame period for camera/OS integration.",
        "",
        "## Results",
        "",
        "| Mode | Video | Runs | FPS mean ± SD | p95 latency mean ± SD (ms) | Deadline misses (%) | Drops (%) | Max temp (°C) | Basic | 20% margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in aggregate_df.itertuples():
        lines.append(
            f"| {row.mode} | {row.video} | {row.n_runs} | "
            f"{row.achieved_fps_mean:.2f} ± {row.achieved_fps_sd:.2f} | "
            f"{row.end_to_end_p95_ms_mean:.2f} ± {row.end_to_end_p95_ms_sd:.2f} | "
            f"{row.deadline_miss_pct_mean:.2f} | {row.drop_rate_pct_mean:.2f} | "
            f"{row.temperature_max_c:.1f} | {format_bool(row.all_basic_realtime_pass)} | "
            f"{format_bool(row.all_recommended_headroom_pass)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    basic = bool(aggregate_df["all_basic_realtime_pass"].all())
    margin = bool(aggregate_df["all_recommended_headroom_pass"].all())
    if basic and margin:
        lines.append(
            "All tested workloads met both the 24 FPS deadline and the preregistered 20% headroom criterion. The optical-flow computation is therefore real-time capable for these recorded scenes and this software configuration."
        )
    elif basic:
        lines.append(
            "All tested workloads met the nominal 24 FPS deadline, but at least one workload failed the 20% integration-margin criterion. Real-time operation is demonstrated under the test conditions, while CSI integration should be treated as requiring confirmation."
        )
    else:
        lines.append(
            "At least one tested workload failed the nominal 24 FPS criterion. The current implementation cannot be described as reliably real-time across the evaluated workloads without optimization or frame-rate/ROI reduction."
        )
    lines.extend(
        [
            "",
            "## Validity and limitations",
            "",
            "The videos preserve representative image content, resolution, motion, and compression complexity. Offline mode includes H.264 file decoding, whereas a CSI camera normally delivers ISP-processed YUV/RGB frames through libcamera; the source overheads are therefore not identical. Realtime mode reproduces arrival cadence, a bounded latest-frame queue, processing backlog, and dropping behavior, but it cannot measure sensor exposure, ISP, CSI transfer, or libcamera latency. A short Picamera2 validation using the same processor remains necessary before claiming complete sensor-to-result latency.",
            "",
            "Runs share one physical device and are not independent Raspberry Pi specimens. When stratified starts are enabled, confidence intervals quantify variation among temporal scene segments on this device, not pure rerun noise or population-level hardware variation.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `frame_timings.csv`: raw per-frame observations",
            "- `run_summary.csv`: per-run statistics and resource measurements",
            "- `aggregate_summary.csv`: repeated-run statistics and bootstrap intervals",
            "- `pooled_summary.csv`: mode-level pooled descriptive statistics",
            "- `environment.json`: software, hardware, Git, command, and input hashes",
            "- `figures/`: publication-resolution timing figures",
            "",
            f"Total measured frame pairs: {len(frame_df):,}",
        ]
    )
    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_config(video: Path, args):
    class CoordinateArgs:
        coordinate_preset = "auto"
        roi = None
        entrance = None

    config = replace(PRESETS[args.preset], preview_stride=args.preview_stride)
    return resolve_config_for_video(video, config, CoordinateArgs())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument(
        "--modes", nargs="+", choices=["offline", "realtime"], default=["offline", "realtime"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-pairs", type=int, default=48)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--stratify-starts",
        action="store_true",
        help="Spread repeated finite segments evenly from the start to end of each video.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="selected")
    parser.add_argument("--opencv-threads", type=int, default=4)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-stride", type=int, default=3)
    parser.add_argument("--shuffle-seed", type=int, default=20260906)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.warmup_pairs < 0:
        parser.error("--warmup-pairs must be >= 0")
    if args.max_frames is not None and args.max_frames <= args.warmup_pairs + 1:
        parser.error("--max-frames must exceed --warmup-pairs + 1")
    return args


def main():
    args = parse_args()
    videos = args.videos or sorted(args.video_dir.glob(args.pattern))
    videos = [video.resolve() for video in videos]
    if not videos:
        raise RuntimeError("No input videos selected")
    missing = [video for video in videos if not video.exists()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.opencv_threads)
    cv2.ocl.setUseOpenCL(False)
    metadata = environment_metadata(args, videos)
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    frame_counts = {}
    for video in videos:
        info_cap, info = load_video_info(video)
        info_cap.release()
        frame_counts[video] = int(info["frame_count"])

    def segment_start(video: Path, repeat_index: int) -> int:
        if not args.stratify_starts or args.max_frames is None or args.repeats == 1:
            return 0
        available = max(0, frame_counts[video] - args.max_frames)
        return int(round(available * (repeat_index - 1) / (args.repeats - 1)))

    jobs = [
        (mode, video, repeat_index, segment_start(video, repeat_index))
        for repeat_index in range(1, args.repeats + 1)
        for mode in args.modes
        for video in videos
    ]
    random.Random(args.shuffle_seed).shuffle(jobs)
    all_frames: list[dict] = []
    summaries: list[dict] = []
    for index, (mode, video, repeat_index, start_frame) in enumerate(jobs, start=1):
        config = resolve_config(video, args)
        preset = find_coordinate_preset(video)
        print(
            f"[{index}/{len(jobs)}] mode={mode} repeat={repeat_index} "
            f"video={video.name} coordinates={preset or 'default'} "
            f"roi={config.roi_x2-config.roi_x1}x{config.roi_y2-config.roi_y1} "
            f"start_frame={start_frame}",
            flush=True,
        )
        runner = run_offline if mode == "offline" else run_realtime
        rows, resources = runner(
            video,
            repeat_index,
            config,
            args.output_dir,
            args.warmup_pairs,
            args.max_frames,
            args.preview,
            start_frame,
        )
        all_frames.extend(rows)
        summary = summarize_run(rows, resources, config, video)
        summaries.append(summary)
        print(
            f"  achieved={summary['achieved_fps']:.2f} FPS, "
            f"p95={summary['end_to_end_ms_p95']:.2f} ms, "
            f"miss={summary['end_to_end_deadline_miss_pct']:.2f}%, "
            f"drop={summary['drop_rate_pct']:.2f}%",
            flush=True,
        )

        pd.DataFrame(all_frames)[FRAME_COLUMNS].to_csv(
            args.output_dir / "frame_timings.partial.csv", index=False
        )
        pd.DataFrame(summaries).to_csv(
            args.output_dir / "run_summary.partial.csv", index=False
        )

    frame_df = pd.DataFrame(all_frames)[FRAME_COLUMNS]
    run_df = pd.DataFrame(summaries)
    aggregate_df = aggregate_runs(run_df, args.shuffle_seed)
    pooled_df = pooled_summary(frame_df, run_df)
    frame_df.to_csv(args.output_dir / "frame_timings.csv", index=False)
    run_df.to_csv(args.output_dir / "run_summary.csv", index=False)
    aggregate_df.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    pooled_df.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    make_plots(frame_df, run_df, args.output_dir)
    finalize_environment(metadata)
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, args, frame_df, run_df, aggregate_df, metadata)
    print(f"Report: {args.output_dir / 'benchmark_report.md'}", flush=True)


if __name__ == "__main__":
    main()
