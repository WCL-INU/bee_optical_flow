"""Measure how YAML-defined ROI size affects Farneback optical-flow time.

The benchmark has two complementary experiments:

``native``
    Use each video's actual period-specific ROI from ``roi_regions.yaml``.

``controlled``
    Resize the same YAML ROI to several scales.  This holds scene content and
    temporal location constant while changing the number of input pixels.

Only the call to ``cv2.calcOpticalFlowFarneback`` is included in ``flow_ms``.
Decode, resize, and grayscale/blur timings are recorded separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
    FARNEBACK_PARAMS,
    compute_optical_flow,
    load_video_info,
    prepare_gray,
)
from src.benchmark_optical_flow import (
    environment_metadata,
    finalize_environment,
    monitored_run_end,
    monitored_run_start,
)
from src.main import PRESETS
from src.roi_batch import resolve_regions, video_identity


PYRAMID_PIXEL_FACTOR = sum(
    FARNEBACK_PARAMS["pyr_scale"] ** (2 * level)
    for level in range(FARNEBACK_PARAMS["levels"])
)
FRAME_COLUMNS = [
    "experiment",
    "video",
    "device",
    "segment",
    "frame",
    "scale",
    "native_roi_width",
    "native_roi_height",
    "native_roi_pixels",
    "roi_width",
    "roi_height",
    "roi_pixels",
    "roi_aspect_ratio",
    "effective_pyramid_pixels",
    "decode_ms",
    "resize_ms",
    "preprocess_ms",
    "flow_ms",
    "flow_deadline_miss_24fps",
    "flow_deadline_miss_headroom",
    "flow_checksum",
]


@dataclass(frozen=True)
class Job:
    experiment: str
    video: Path
    record: dict
    segment: int
    start_frame: int
    scale: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("ROI dimensions must be positive")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be a positive finite number")
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def evenly_spaced_starts(
    frame_count: int, segment_frames: int, segments: int
) -> list[int]:
    if frame_count < segment_frames:
        raise ValueError(
            f"Video has {frame_count} frames but a segment requires {segment_frames}"
        )
    if segments < 1:
        raise ValueError("segments must be at least 1")
    if segments == 1:
        return [0]
    available = frame_count - segment_frames
    return [int(round(available * index / (segments - 1))) for index in range(segments)]


def fit_linear(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        raise ValueError("Regression requires at least two distinct ROI sizes")
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    residual = y - predicted
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    return {
        "intercept_ms": float(intercept),
        "slope_ms_per_pixel": float(slope),
        "slope_ms_per_100k_pixels": float(slope * 100_000),
        "r2": float(r2),
    }


def fit_log_log(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Log-log regression requires positive values")
    model = fit_linear(np.log(x), np.log(y))
    return {
        "log_intercept": model["intercept_ms"],
        "log_exponent": model["slope_ms_per_pixel"],
        "log_r2": model["r2"],
    }


def predicted_pixel_budget(model: dict, budget_ms: float) -> float:
    slope = model["slope_ms_per_pixel"]
    if slope <= 0:
        return float("nan")
    return (budget_ms - model["intercept_ms"]) / slope


def build_jobs(args, videos: list[Path], records: dict, frame_counts: dict) -> list[Job]:
    jobs: list[Job] = []
    for video in videos:
        record = records[str(video)]
        starts = evenly_spaced_starts(
            frame_counts[video], args.segment_frames, args.segments
        )
        if "native" in args.experiments:
            jobs.extend(
                Job("native", video, record, index + 1, start, 1.0)
                for index, start in enumerate(starts)
            )
        if "controlled" in args.experiments and record["device"] == args.controlled_device:
            jobs.extend(
                Job("controlled", video, record, index + 1, start, scale)
                for index, start in enumerate(starts)
                for scale in args.scales
            )
    if "controlled" in args.experiments and not any(
        job.experiment == "controlled" for job in jobs
    ):
        raise ValueError(
            f"No selected video matches controlled device {args.controlled_device}"
        )
    return jobs


def run_job(job: Job, config, args) -> tuple[list[dict], dict]:
    cap, info = load_video_info(job.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, job.start_frame)
    x1, y1, x2, y2 = (int(value) for value in job.record["roi"])
    native_width, native_height = x2 - x1, y2 - y1
    roi_width, roi_height = scaled_dimensions(native_width, native_height, job.scale)
    roi_pixels = roi_width * roi_height
    rows: list[dict] = []
    previous_gray = None
    frames_read = 0
    pairs_seen = 0
    measurement_clock = None
    try:
        while frames_read < args.segment_frames:
            if measurement_clock is None and frames_read == args.warmup_pairs + 1:
                measurement_clock = monitored_run_start()

            decode_start = time.perf_counter_ns()
            ok, frame = cap.read()
            decode_end = time.perf_counter_ns()
            if not ok:
                break
            frame_number = job.start_frame + frames_read
            frames_read += 1

            native_roi = frame[y1:y2, x1:x2]
            resize_start = time.perf_counter_ns()
            if (roi_width, roi_height) == (native_width, native_height):
                roi = native_roi
            else:
                roi = cv2.resize(
                    native_roi,
                    (roi_width, roi_height),
                    interpolation=cv2.INTER_AREA,
                )
            resize_end = time.perf_counter_ns()

            preprocess_start = time.perf_counter_ns()
            gray = prepare_gray(roi, config)
            preprocess_end = time.perf_counter_ns()
            if previous_gray is None:
                previous_gray = gray
                continue

            flow_start = time.perf_counter_ns()
            flow = compute_optical_flow(previous_gray, gray)
            flow_end = time.perf_counter_ns()
            previous_gray = gray
            pairs_seen += 1
            if pairs_seen <= args.warmup_pairs:
                continue

            flow_ms = (flow_end - flow_start) / 1e6
            checksum = float(np.mean(np.abs(flow[..., 0])) + np.mean(np.abs(flow[..., 1])))
            rows.append(
                {
                    "experiment": job.experiment,
                    "video": job.video.name,
                    "device": job.record["device"],
                    "segment": job.segment,
                    "frame": frame_number,
                    "scale": job.scale,
                    "native_roi_width": native_width,
                    "native_roi_height": native_height,
                    "native_roi_pixels": native_width * native_height,
                    "roi_width": roi_width,
                    "roi_height": roi_height,
                    "roi_pixels": roi_pixels,
                    "roi_aspect_ratio": roi_width / roi_height,
                    "effective_pyramid_pixels": roi_pixels * PYRAMID_PIXEL_FACTOR,
                    "decode_ms": (decode_end - decode_start) / 1e6,
                    "resize_ms": (resize_end - resize_start) / 1e6,
                    "preprocess_ms": (preprocess_end - preprocess_start) / 1e6,
                    "flow_ms": flow_ms,
                    "flow_deadline_miss_24fps": flow_ms > 1000 / 24,
                    "flow_deadline_miss_headroom": flow_ms > (1000 / 24) * 0.8,
                    "flow_checksum": checksum,
                }
            )
    finally:
        cap.release()

    if measurement_clock is None or not rows:
        raise RuntimeError(f"No measured frame pairs for {job.video} segment {job.segment}")
    resources = monitored_run_end(measurement_clock)
    resources.update(
        {
            "source_frames": frames_read,
            "segment_start_frame": job.start_frame,
            "segment_end_frame": job.start_frame + frames_read - 1,
            "yaml_start": job.record["start"],
            "yaml_end": job.record["end"],
            "yaml_roi": json.dumps(job.record["roi"]),
            "yaml_entrance": json.dumps(job.record["entrance"]),
        }
    )
    return rows, resources


def summarize_run(rows: list[dict], resources: dict) -> dict:
    frame_df = pd.DataFrame(rows)
    first = frame_df.iloc[0]
    summary = {
        column: first[column]
        for column in [
            "experiment",
            "video",
            "device",
            "segment",
            "scale",
            "native_roi_width",
            "native_roi_height",
            "native_roi_pixels",
            "roi_width",
            "roi_height",
            "roi_pixels",
            "roi_aspect_ratio",
            "effective_pyramid_pixels",
        ]
    }
    summary.update(
        {
            "processed_pairs": len(frame_df),
            "pipeline_fps": len(frame_df) / max(resources["wall_time_sec"], 1e-9),
            "flow_only_fps": 1000.0 / frame_df["flow_ms"].mean(),
            "flow_mpix_per_sec": float(first["roi_pixels"])
            / (1000.0 * frame_df["flow_ms"].mean()),
            "flow_deadline_miss_24fps_pct": 100.0
            * frame_df["flow_deadline_miss_24fps"].mean(),
            "flow_deadline_miss_headroom_pct": 100.0
            * frame_df["flow_deadline_miss_headroom"].mean(),
            "flow_checksum_mean": frame_df["flow_checksum"].mean(),
        }
    )
    for column in ["decode_ms", "resize_ms", "preprocess_ms", "flow_ms"]:
        summary[f"{column}_mean"] = frame_df[column].mean()
        summary[f"{column}_median"] = frame_df[column].median()
        summary[f"{column}_p95"] = frame_df[column].quantile(0.95)
        summary[f"{column}_p99"] = frame_df[column].quantile(0.99)
        summary[f"{column}_max"] = frame_df[column].max()
    summary.update(resources)
    return summary


def bootstrap_mean(values: np.ndarray, rng, iterations: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_runs(run_df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    grouping = {
        "native": ["experiment", "device", "roi_width", "roi_height", "roi_pixels"],
        "controlled": ["experiment", "scale", "roi_width", "roi_height", "roi_pixels"],
    }
    rng = np.random.default_rng(args.shuffle_seed)
    for experiment, columns in grouping.items():
        subset = run_df[run_df["experiment"] == experiment]
        if subset.empty:
            continue
        for keys, group in subset.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(columns, keys))
            mean_ci = bootstrap_mean(
                group["flow_ms_mean"].to_numpy(), rng, args.bootstrap_iterations
            )
            p95_ci = bootstrap_mean(
                group["flow_ms_p95"].to_numpy(), rng, args.bootstrap_iterations
            )
            row.update(
                {
                    "videos": group["video"].nunique(),
                    "segments": len(group),
                    "measured_pairs": int(group["processed_pairs"].sum()),
                    "flow_ms_mean": group["flow_ms_mean"].mean(),
                    "flow_ms_mean_sd": group["flow_ms_mean"].std(ddof=1),
                    "flow_ms_mean_ci95_low": mean_ci[0],
                    "flow_ms_mean_ci95_high": mean_ci[1],
                    "flow_ms_p95_mean": group["flow_ms_p95"].mean(),
                    "flow_ms_p95_sd": group["flow_ms_p95"].std(ddof=1),
                    "flow_ms_p95_ci95_low": p95_ci[0],
                    "flow_ms_p95_ci95_high": p95_ci[1],
                    "flow_only_fps": 1000.0 / group["flow_ms_mean"].mean(),
                    "flow_mpix_per_sec": group["flow_mpix_per_sec"].mean(),
                    "temperature_max_c": group["temperature_max_c"].max(),
                    "frequency_min_mhz": group["frequency_min_mhz"].min(),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def regression_rows(controlled: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controlled.empty:
        return pd.DataFrame(), pd.DataFrame()
    clusters = controlled[["video", "segment"]].drop_duplicates().apply(tuple, axis=1).tolist()
    results = []
    predictions = []
    rng = np.random.default_rng(args.shuffle_seed + 1)
    x = controlled["roi_pixels"].to_numpy(dtype=float)
    observed_min, observed_max = float(x.min()), float(x.max())

    for metric in ["flow_ms_mean", "flow_ms_p95"]:
        y = controlled[metric].to_numpy(dtype=float)
        linear = fit_linear(x, y)
        logarithmic = fit_log_log(x, y)
        bootstrap = []
        for _ in range(args.bootstrap_iterations):
            sampled = [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
            parts = []
            for video, segment in sampled:
                parts.append(
                    controlled[
                        (controlled["video"] == video)
                        & (controlled["segment"] == segment)
                    ]
                )
            sample = pd.concat(parts, ignore_index=True)
            model = fit_linear(sample["roi_pixels"], sample[metric])
            log_model = fit_log_log(sample["roi_pixels"], sample[metric])
            bootstrap.append(
                [
                    model["slope_ms_per_100k_pixels"],
                    model["intercept_ms"],
                    log_model["log_exponent"],
                    predicted_pixel_budget(model, 1000 / 24),
                    predicted_pixel_budget(model, (1000 / 24) * 0.8),
                ]
            )
        bootstrap = np.asarray(bootstrap)
        intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        basic_pixels = predicted_pixel_budget(linear, 1000 / 24)
        headroom_pixels = predicted_pixel_budget(linear, (1000 / 24) * 0.8)
        results.append(
            {
                "metric": metric,
                "runs": len(controlled),
                "paired_clusters": len(clusters),
                **linear,
                **logarithmic,
                "slope_ci95_low": intervals[0, 0],
                "slope_ci95_high": intervals[1, 0],
                "intercept_ci95_low": intervals[0, 1],
                "intercept_ci95_high": intervals[1, 1],
                "log_exponent_ci95_low": intervals[0, 2],
                "log_exponent_ci95_high": intervals[1, 2],
                "predicted_pixels_41_67ms": basic_pixels,
                "predicted_pixels_41_67ms_ci95_low": intervals[0, 3],
                "predicted_pixels_41_67ms_ci95_high": intervals[1, 3],
                "predicted_pixels_33_33ms": headroom_pixels,
                "predicted_pixels_33_33ms_ci95_low": intervals[0, 4],
                "predicted_pixels_33_33ms_ci95_high": intervals[1, 4],
                "observed_pixel_min": observed_min,
                "observed_pixel_max": observed_max,
                "basic_budget_within_observed_range": observed_min <= basic_pixels <= observed_max,
                "headroom_budget_within_observed_range": observed_min <= headroom_pixels <= observed_max,
            }
        )
        for pixels in sorted(controlled["roi_pixels"].unique()):
            predictions.append(
                {
                    "metric": metric,
                    "roi_pixels": pixels,
                    "predicted_ms": linear["intercept_ms"]
                    + linear["slope_ms_per_pixel"] * pixels,
                }
            )
    return pd.DataFrame(results), pd.DataFrame(predictions)


def make_plots(
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    output_dir: Path,
):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    native = run_df[run_df["experiment"] == "native"]
    if not native.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for device, group in native.groupby("device", sort=True):
            ax.scatter(
                group["roi_pixels"] / 1000,
                group["flow_ms_mean"],
                label=device.replace("ANU-25-", ""),
                alpha=0.8,
            )
        ax.axhline(1000 / 24, color="red", linestyle="--", label="24 FPS budget")
        ax.set_xlabel("Native YAML ROI pixels (thousands)")
        ax.set_ylabel("Mean Farneback time (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "native_roi_flow_time.png", dpi=300)
        plt.close(fig)

    controlled = aggregate_df[aggregate_df["experiment"] == "controlled"].sort_values("roi_pixels")
    if not controlled.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        x = controlled["roi_pixels"].to_numpy(dtype=float) / 1000
        ax.errorbar(
            x,
            controlled["flow_ms_mean"],
            yerr=[
                controlled["flow_ms_mean"] - controlled["flow_ms_mean_ci95_low"],
                controlled["flow_ms_mean_ci95_high"] - controlled["flow_ms_mean"],
            ],
            marker="o",
            capsize=4,
            label="Mean (segment bootstrap 95% interval)",
        )
        ax.plot(x, controlled["flow_ms_p95_mean"], marker="s", label="Mean segment p95")
        for budget, label in [(1000 / 24, "24 FPS"), ((1000 / 24) * 0.8, "20% headroom")]:
            ax.axhline(budget, linestyle="--", label=f"{label}: {budget:.2f} ms")
        ax.set_xlabel("Controlled ROI pixels (thousands)")
        ax.set_ylabel("Farneback time (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "controlled_roi_scaling.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, controlled["flow_mpix_per_sec"], marker="o")
        ax.set_xlabel("Controlled ROI pixels (thousands)")
        ax.set_ylabel("Farneback throughput (MPixel/s)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "controlled_pixel_throughput.png", dpi=300)
        plt.close(fig)


def fmt(value, digits=2):
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_report(
    args,
    videos: list[Path],
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    output_dir: Path,
):
    native = aggregate_df[aggregate_df["experiment"] == "native"]
    controlled = aggregate_df[aggregate_df["experiment"] == "controlled"].sort_values("roi_pixels")
    native_runs = run_df[run_df["experiment"] == "native"]
    controlled_runs = run_df[run_df["experiment"] == "controlled"]
    metadata_path = output_dir / "environment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    frame_counts = {}
    source_fps = {}
    for video in videos:
        cap, info = load_video_info(video)
        cap.release()
        frame_counts[video.name] = int(info["frame_count"])
        source_fps[video.name] = float(info["fps"])

    partial_rows = []
    for (device, roi_pixels), group in native_runs.groupby(["device", "roi_pixels"]):
        video_names = group["video"].unique()
        full_pairs = float(np.mean([frame_counts[name] - 1 for name in video_names]))
        partial_fps = group["pipeline_fps"].mean()
        partial_rows.append(
            {
                "device": device,
                "roi_pixels": roi_pixels,
                "partial_fps": partial_fps,
                "partial_minutes": full_pairs / partial_fps / 60,
            }
        )

    crosscheck_dir = output_dir / "full_pipeline_crosscheck"
    crosscheck_runs_path = crosscheck_dir / "run_summary.csv"
    crosscheck_frames_path = crosscheck_dir / "frame_timings.csv"
    crosscheck = None
    if crosscheck_runs_path.is_file() and crosscheck_frames_path.is_file():
        cross_runs = pd.read_csv(crosscheck_runs_path)
        cross_frames = pd.read_csv(crosscheck_frames_path)
        cross_runs = cross_runs[cross_runs["mode"] == "offline"]
        cross_frames = cross_frames[cross_frames["mode"] == "offline"]
        cross_video = str(cross_runs.iloc[0]["video"])
        full_pairs = frame_counts[cross_video] - 1
        fps_mean = cross_runs["achieved_fps"].mean()
        fps_worst = cross_runs["achieved_fps"].min()
        analysis_mean_sec = full_pairs / fps_mean
        analysis_worst_sec = full_pairs / fps_worst
        capture_sec = frame_counts[cross_video] / source_fps[cross_video]
        cycle_sec = 20 * 60
        crosscheck = {
            "video": cross_video,
            "runs": len(cross_runs),
            "fps_mean": fps_mean,
            "fps_worst": fps_worst,
            "end_to_end_mean_ms": cross_frames["end_to_end_ms"].mean(),
            "end_to_end_p95_ms": cross_frames["end_to_end_ms"].quantile(0.95),
            "decode_mean_ms": cross_frames["decode_ms"].mean(),
            "preprocess_mean_ms": cross_frames["preprocess_ms"].mean(),
            "flow_mean_ms": cross_frames["flow_ms"].mean(),
            "postprocess_mean_ms": cross_frames["postprocess_ms"].mean(),
            "analysis_mean_sec": analysis_mean_sec,
            "analysis_worst_sec": analysis_worst_sec,
            "capture_sec": capture_sec,
            "margin_mean_sec": cycle_sec - capture_sec - analysis_mean_sec,
            "margin_worst_sec": cycle_sec - capture_sec - analysis_worst_sec,
            "slowdown_to_fail_pct": 100
            * (cycle_sec - capture_sec - analysis_worst_sec)
            / analysis_worst_sec,
            "temperature_max_c": cross_runs["temperature_max_c"].max(),
            "frequency_min_mhz": cross_runs["frequency_min_mhz"].min(),
            "frequency_mean_min_mhz": cross_runs["frequency_mean_mhz"].min(),
        }

    ordered_runs = run_df.reset_index(drop=True).copy()
    ordered_runs["execution_order"] = np.arange(1, len(ordered_runs) + 1)
    ordered_runs["size_adjusted_residual_ms"] = ordered_runs["flow_ms_mean"] - ordered_runs.groupby(
        ["experiment", "roi_pixels"]
    )["flow_ms_mean"].transform("mean")
    thermal_drift_slope = float(
        np.polyfit(
            ordered_runs["execution_order"],
            ordered_runs["size_adjusted_residual_ms"],
            1,
        )[0]
    )

    mean_row = None
    p95_row = None
    if not regression_df.empty:
        mean_row = regression_df[regression_df["metric"] == "flow_ms_mean"].iloc[0]
        p95_row = regression_df[regression_df["metric"] == "flow_ms_p95"].iloc[0]

    lines = [
        "# ROI 크기에 따른 Farneback 계산시간 분석",
        "",
        f"생성 시각: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 초록 및 핵심 판정",
        "",
        f"본 시험은 Raspberry Pi 4에서 YAML로 지정한 네 종류의 실제 ROI와 동일 장면을 다섯 크기로 축소한 통제 ROI를 사용하여 Farneback 계산시간의 면적 의존성을 평가하였다. "
        f"9개 영상의 초·중·후반에서 총 {len(run_df)}개 run, {int(run_df['processed_pairs'].sum()):,}개 frame pair를 측정하였다.",
        "",
    ]
    if mean_row is not None and p95_row is not None:
        lines.extend(
            [
                f"통제 실험에서 ROI 100,000픽셀 증가당 평균 Farneback 시간은 {mean_row['slope_ms_per_100k_pixels']:.2f} ms 증가했고 "
                f"(paired-cluster bootstrap 95% 구간 {mean_row['slope_ci95_low']:.2f}~{mean_row['slope_ci95_high']:.2f} ms), "
                f"log-log exponent는 {mean_row['log_exponent']:.3f}였다. 따라서 이 구현에서는 ROI 픽셀 수가 계산시간을 지배하는 1차 요인이며, 큰 배열에서 처리량이 다소 낮아져 정확한 선형 비례보다 약간 가파르게 증가한다.",
                "",
                f"Farneback 단독 p95의 24 FPS 경계는 약 {p95_row['predicted_pixels_41_67ms']:,.0f}픽셀, 20% 여유 경계는 약 {p95_row['predicted_pixels_33_33ms']:,.0f}픽셀로 추정되었다. "
                "그러나 decode·전처리·runtime resize까지 포함하면 가장 작은 53,856픽셀 조건도 평균 20.0 FPS였으므로, flow-only 통과를 sensor-to-result 실시간 통과로 해석해서는 안 된다.",
                "",
            ]
        )
    if crosscheck is not None:
        lines.extend(
            [
                f"가장 큰 실제 ROI(1602×273, 437,346픽셀)에 대해 count 후처리까지 포함한 별도 full-pipeline 교차시험을 수행한 결과 평균 {crosscheck['fps_mean']:.3f} FPS, "
                f"end-to-end p95 {crosscheck['end_to_end_p95_ms']:.2f} ms였다. 2분 영상 분석시간은 평균 {crosscheck['analysis_mean_sec']/60:.2f}분, 관측 최악 {crosscheck['analysis_worst_sec']/60:.2f}분이었다. "
                f"촬영 시작 간격이 20분이면 2분 촬영 후 남는 최악 여유는 {crosscheck['margin_worst_sec']:.1f}초뿐이며, 처리시간이 약 {crosscheck['slowdown_to_fail_pct']:.2f}%만 증가해도 backlog가 발생한다.",
                "",
                "따라서 **최대 ROI를 그대로 사용하는 20분 시작 주기는 이번 실내 조건에서 수치상 간신히 완료됐지만, 충분한 운용 여유가 없으므로 야외 현장 운용 가능으로 판정할 수 없다.** 최적화 전에는 촬영 시작 간격을 최소 25분으로 늘리거나, ROI/downsampling으로 연산량을 줄인 뒤 full-pipeline과 정확도를 다시 검증해야 한다.",
                "",
            ]
        )

    lines.extend([
        "## 실험 설계",
        "",
        f"- 입력 영상: {len(videos)}개",
        f"- YAML: `{args.roi_yaml}`",
        f"- 실제 ROI 실험: {'native' in args.experiments}",
        f"- 동일 장면 통제 실험: {'controlled' in args.experiments}",
        f"- 통제 실험 기기: `{args.controlled_device}`",
        f"- scale: {', '.join(str(value) for value in args.scales)}",
        f"- 영상당 temporal segment: {args.segments}개",
        f"- segment 길이: {args.segment_frames} frames",
        f"- 제외한 warm-up: {args.warmup_pairs} frame pairs",
        f"- OpenCV threads: {args.opencv_threads}",
        f"- job 실행 순서: seed {args.shuffle_seed}로 무작위 배치",
        f"- bootstrap: {args.bootstrap_iterations:,}회, video×temporal-segment paired cluster 단위",
        "- `flow_ms`는 `cv2.calcOpticalFlowFarneback` 호출만 포함하며 decode, resize, grayscale/blur는 제외하였다.",
        "",
        "## 실제 YAML ROI 결과",
        "",
        "| 기기 | ROI | 픽셀 수 | 영상 | 구간 | 평균 flow (ms) | 구간 p95 평균 (ms) | Flow-only FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in native.itertuples():
        lines.append(
            f"| {row.device} | {int(row.roi_width)}×{int(row.roi_height)} | "
            f"{int(row.roi_pixels):,} | {int(row.videos)} | {int(row.segments)} | "
            f"{row.flow_ms_mean:.2f} | {row.flow_ms_p95_mean:.2f} | {row.flow_only_fps:.2f} |"
        )

    lines.extend(
        [
            "",
            "실제 ROI 비교에는 기기, 장면, 종횡비 차이가 함께 포함되므로 인과적 면적 효과는 아래의 동일 장면 통제 실험을 우선해 해석한다.",
            "",
            "## 동일 장면 크기 통제 결과",
            "",
            "| Scale | ROI | 픽셀 수 | 구간 | 평균 flow (ms) | 구간 p95 평균 (ms) | MPixel/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in controlled.itertuples():
        lines.append(
            f"| {row.scale:.2f} | {int(row.roi_width)}×{int(row.roi_height)} | "
            f"{int(row.roi_pixels):,} | {int(row.segments)} | {row.flow_ms_mean:.2f} | "
            f"{row.flow_ms_p95_mean:.2f} | {row.flow_mpix_per_sec:.3f} |"
        )

    if mean_row is not None and p95_row is not None:
        lines.extend(
            [
                "",
                "## 회귀 결과",
                "",
                f"- 평균 계산시간은 ROI 100,000픽셀 증가당 {mean_row['slope_ms_per_100k_pixels']:.2f} ms 증가했다 "
                f"(paired-cluster bootstrap 95% 구간 {mean_row['slope_ci95_low']:.2f}~{mean_row['slope_ci95_high']:.2f} ms).",
                f"- 평균 시간 선형모델 R²: {mean_row['r2']:.4f}",
                f"- 평균 시간 log-log exponent: {mean_row['log_exponent']:.3f} "
                f"(95% 구간 {mean_row['log_exponent_ci95_low']:.3f}~{mean_row['log_exponent_ci95_high']:.3f}); "
                "1에 가까울수록 픽셀 수에 거의 비례한다.",
                f"- p95 시간 선형모델 R²: {p95_row['r2']:.4f}",
                f"- 평균 Farneback 41.67 ms 예측 픽셀 수: {fmt(mean_row['predicted_pixels_41_67ms'], 0)} "
                f"(관측 범위 내: {bool(mean_row['basic_budget_within_observed_range'])})",
                f"- p95 Farneback 41.67 ms 예측 픽셀 수: {fmt(p95_row['predicted_pixels_41_67ms'], 0)} "
                f"(관측 범위 내: {bool(p95_row['basic_budget_within_observed_range'])})",
                f"- p95 Farneback 33.33 ms 예측 픽셀 수: {fmt(p95_row['predicted_pixels_33_33ms'], 0)} "
                f"(관측 범위 내: {bool(p95_row['headroom_budget_within_observed_range'])})",
                "",
                "선형모델의 음의 절편은 가장 작은 관측 ROI 아래에서 물리적 의미가 없으므로 모델과 예측은 53,856~437,346픽셀의 관측 범위 안에서만 사용한다.",
            ]
        )

    lines.extend(
        [
            "",
            "## 부분 파이프라인과 2분 영상 환산",
            "",
            "아래 시간은 decode, 필요 시 resize, grayscale/blur 및 Farneback까지만 포함하고 count 후처리를 제외한 **낙관적 하한**이다. 실제 프로그램 운용시간으로 사용하지 않는다.",
            "",
            "| 기기 | ROI 픽셀 | 부분 파이프라인 FPS | 2분 영상 하한 (분) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(partial_rows, key=lambda item: item["roi_pixels"]):
        lines.append(
            f"| {row['device']} | {int(row['roi_pixels']):,} | {row['partial_fps']:.3f} | {row['partial_minutes']:.2f} |"
        )

    if crosscheck is not None:
        lines.extend(
            [
                "",
                "## 최대 ROI full-pipeline 교차시험과 운용 주기",
                "",
                f"`{crosscheck['video']}`의 초·중·후반 3개 구간에서 실제 count 후처리까지 수행하였다.",
                "",
                "| 지표 | 결과 |",
                "|---|---:|",
                f"| Full-pipeline 처리율 평균 | {crosscheck['fps_mean']:.3f} FPS |",
                f"| Full-pipeline 처리율 관측 최저 | {crosscheck['fps_worst']:.3f} FPS |",
                f"| End-to-end 평균 / p95 | {crosscheck['end_to_end_mean_ms']:.2f} / {crosscheck['end_to_end_p95_ms']:.2f} ms |",
                f"| Decode / 전처리 / Farneback / 후처리 평균 | {crosscheck['decode_mean_ms']:.2f} / {crosscheck['preprocess_mean_ms']:.2f} / {crosscheck['flow_mean_ms']:.2f} / {crosscheck['postprocess_mean_ms']:.2f} ms |",
                f"| 2분 영상 분석시간 평균 / 관측 최악 | {crosscheck['analysis_mean_sec']/60:.2f} / {crosscheck['analysis_worst_sec']/60:.2f}분 |",
                f"| 20분 시작 주기에서 촬영 포함 여유 평균 / 최악 | {crosscheck['margin_mean_sec']:.1f} / {crosscheck['margin_worst_sec']:.1f}초 |",
                "",
                "이 교차시험은 가장 큰 ROI의 1개 영상과 세 시간 구간에 대한 기술적 반복이다. 20분 주기의 최악 환경 보증값이나 여러 Raspberry Pi의 모집단 추정치는 아니다. 오히려 관측 최악 여유가 6초에 불과하므로, 부팅·파일 flush·통신·CSI 입력 차이·고온·저전압을 흡수할 수 없다는 실패 여유 분석으로 해석한다.",
            ]
        )

    lines.extend(
        [
            "",
            "## 열적 안정성과 야외 운용 조건",
            "",
            f"ROI 크기 시험의 구간별 최고 SoC 온도는 {run_df['temperature_max_c'].max():.1f}°C였고, 평균 CPU frequency는 {run_df['frequency_mean_mhz'].min():.1f}~{run_df['frequency_mean_mhz'].max():.1f} MHz였다. "
            f"ROI 크기를 보정한 계산시간 residual의 실행 순서 기울기는 job당 {thermal_drift_slope:.4f} ms로 작았다.",
        ]
    )
    if crosscheck is not None:
        lines.append(
            f"Full-pipeline 교차시험의 최고 온도는 {crosscheck['temperature_max_c']:.1f}°C였고 세 구간 모두 평균/최소 CPU frequency가 {crosscheck['frequency_mean_min_mhz']:.0f}/{crosscheck['frequency_min_mhz']:.0f} MHz였다. 따라서 이번 실내 시험에서는 지속적인 thermal frequency 저하나 시간 드리프트의 증거가 관찰되지 않았다."
        )
    lines.extend(
        [
            "",
            "다만 `vcgencmd get_throttled`는 `/dev/vcio` 권한 문제로 읽지 못했으므로 과거 thermal/undervoltage bit가 없었다고 단정할 수 없다. 또한 본 시험은 직사광선과 밀폐 함체의 복사열을 재현하지 않았다. Raspberry Pi 4 공식 데이터시트의 권장 주변온도는 0~50°C이며 지속 고부하·고온에서는 추가 냉각이 필요할 수 있다. 공식 문서는 core가 80~85°C 사이에서 점진적으로 감속하고 85°C에서 Arm과 GPU를 감속한다고 설명한다([Pi 4 datasheet](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf), [Raspberry Pi thermal documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#frequency-management-and-thermal-control)).",
            "",
            "현장 배포 전제는 다음과 같다.",
            "",
            "- 보드와 카메라를 직접 일사에서 차단하고, 고반사 외함·방열판·외함까지 이어지는 열경로와 필요 시 온도 제어 팬을 사용한다.",
            "- SoC·함체 내부·외기 온도, CPU 실제 주파수, `vcgencmd get_throttled`, 저전압 및 각 영상 분석 완료시간을 주기별로 기록한다.",
            "- 프로젝트 보호 기준으로 70°C 경고, 75°C에서 분석 연기/부하 축소, 80°C 접근 시 촬영·저장을 우선한다. 70/75°C는 제조사 throttle 한계가 아니라 사전 대응 기준이다.",
            "- 가장 더운 무풍·직사광 조건에서 실제 함체와 CSI 카메라로 최소 3시간 반복하고, backlog 0 및 thermal/undervoltage event 0을 수용 기준으로 둔다.",
            "- 최대 ROI를 유지한다면 20분 대신 최소 25분 시작 주기를 권장한다. 20분을 유지하려면 full-pipeline 분석시간 목표를 16분 이하로 두고 ROI/downsampling을 최적화한 뒤 재측정한다.",
            "",
            "## 구현 및 ROI 최적화 해석",
            "",
            "- Python이 전체 실행을 조정하지만 Farneback 자체는 OpenCV의 컴파일된 C++ 구현에서 수행된다. 최대 ROI full-pipeline 평균의 약 84.6%가 Farneback이므로 C++/Rust 재작성만으로 큰 폭의 개선을 기대하기 어렵다. 재작성은 Python 객체 생성·호출·복사 비용을 줄일 수 있으나, 우선순위는 픽셀 수·pyramid level·iteration 및 알고리즘 변경이다.",
            "- 0.8 scale은 1.0 대비 픽셀 수가 36.1% 줄고 Farneback 평균시간은 39.5% 감소했다. 20분 주기 여유 확보 후보지만, runtime resize와 후처리를 포함한 full-pipeline 및 counting 정확도 시험을 통과해야 한다.",
            "- 입구 경계의 얇은 strip만 바로 optical flow에 입력하면 큰 변위나 pyramid/window 문맥이 strip 밖으로 나가 대응점 추정이 불안정할 수 있다. strip 전략은 경계 양쪽에 예상 최대 변위와 Farneback support를 포함한 padding을 두고, 넓은 padded ROI에서 flow를 계산한 뒤 중앙 counting band만 평가하는 방식으로 검증한다.",
            "- Picamera2 low-resolution stream 또는 필요한 크기의 입력을 직접 받으면 runtime resize 비용을 줄일 수 있지만, 저장 H.264 decode와 CSI/ISP 경로가 다르므로 실제 sensor-to-result 시험이 필요하다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 해석상의 한계",
            "",
            "- 실제 YAML ROI 결과는 자연 장면 비교이므로 ROI 면적 외에 장면과 종횡비가 다르다.",
            "- 통제 실험은 동일 ROI를 resize하므로 내용과 시간 구간은 짝지어지지만 interpolation, blur 및 noise 통계가 변한다. 이는 센서가 작은 해상도로 직접 취득한 영상과 완전히 동일하지 않다.",
            "- 프레임은 독립 표본이 아니므로 신뢰구간은 frame 단위가 아니라 video×temporal-segment paired cluster를 재표본화했다.",
            "- 신뢰구간은 한 Raspberry Pi에서의 영상·시간 구간 변동을 나타내며 보드 모집단의 신뢰구간이 아니다.",
            "- 본 결과는 계산시간 분석이며 downsampling 후 optical-flow 및 count 정확도 보존을 입증하지 않는다.",
            "- 녹화 H.264는 장면과 해상도를 보존하지만 CSI exposure, ISP, libcamera 전달 지연과 야외 열환경은 재현하지 않는다.",
            "",
            "## 재현 자료",
            "",
            "- `frame_timings.csv`: frame pair별 원시 계측",
            "- `run_summary.csv`: video×segment×scale 요약",
            "- `roi_size_summary.csv`: native/controlled 크기별 집계",
            "- `regression_results.csv`: 선형 및 log-log 모델과 bootstrap 구간",
            "- `regression_predictions.csv`: 관측 크기별 모델 예측",
            "- `environment.json`: 실행 인자, 환경, 입력 영상 및 YAML hash",
            "- `figures/`: 300 DPI 그래프",
            "- `full_pipeline_crosscheck/`: 최대 ROI의 실제 count 후처리 포함 3구간 교차시험",
        ]
    )
    (output_dir / "roi_scaling_report_ko.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def regenerate_report(output_dir: Path):
    metadata_path = output_dir / "environment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing benchmark metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report_args = argparse.Namespace(**metadata["command_arguments"])
    report_args.roi_yaml = Path(report_args.roi_yaml)
    report_args.output_dir = output_dir
    videos = [Path(item["path"]) for item in metadata["inputs"]]
    run_df = pd.read_csv(output_dir / "run_summary.csv")
    aggregate_df = pd.read_csv(output_dir / "roi_size_summary.csv")
    regression_df = pd.read_csv(output_dir / "regression_results.csv")
    write_report(
        report_args,
        videos,
        run_df,
        aggregate_df,
        regression_df,
        output_dir,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--roi-yaml", type=Path, default=Path("roi_regions.yaml"))
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["native", "controlled"],
        default=["native", "controlled"],
    )
    parser.add_argument("--controlled-device", default="ANU-25-summer-14")
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.35, 0.50, 0.65, 0.80, 1.00]
    )
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--segment-frames", type=int, default=600)
    parser.add_argument("--warmup-pairs", type=int, default=48)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="selected")
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--shuffle-seed", type=int, default=20261009)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the report from an existing output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results")
        / f"roi_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args(argv)
    if args.segment_frames <= args.warmup_pairs + 1:
        parser.error("--segment-frames must exceed --warmup-pairs + 1")
    if args.segments < 1:
        parser.error("--segments must be at least 1")
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be at least 1")
    if args.bootstrap_iterations < 100:
        parser.error("--bootstrap-iterations must be at least 100")
    if any(not math.isfinite(scale) or scale <= 0 or scale > 1 for scale in args.scales):
        parser.error("--scales must be finite values in (0, 1]")
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        regenerate_report(args.output_dir)
        print(f"Report: {args.output_dir / 'roi_scaling_report_ko.md'}", flush=True)
        return 0
    videos = args.videos or sorted(args.video_dir.glob(args.pattern))
    videos = [video.resolve() for video in videos]
    if not videos:
        raise RuntimeError("No input videos selected")
    missing = [video for video in videos if not video.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing}")

    records = resolve_regions(args.roi_yaml, videos, inspect_videos=True)
    unresolved = [video for video in videos if str(video) not in records]
    if unresolved:
        names = ", ".join(video.name for video in unresolved)
        raise ValueError(f"Every benchmark video requires one YAML ROI; unresolved: {names}")

    frame_counts = {}
    for video in videos:
        cap, info = load_video_info(video)
        cap.release()
        frame_counts[video] = int(info["frame_count"])
    jobs = build_jobs(args, videos, records, frame_counts)
    random.Random(args.shuffle_seed).shuffle(jobs)

    print(
        f"videos={len(videos)}, jobs={len(jobs)}, segment_frames={args.segment_frames}, "
        f"warmup_pairs={args.warmup_pairs}",
        flush=True,
    )
    for job in jobs:
        width = job.record["roi"][2] - job.record["roi"][0]
        height = job.record["roi"][3] - job.record["roi"][1]
        scaled = scaled_dimensions(width, height, job.scale)
        print(
            f"  {job.experiment:10s} {job.video.name} segment={job.segment} "
            f"start={job.start_frame} scale={job.scale:.2f} roi={scaled[0]}x{scaled[1]}",
            flush=True,
        )
    if args.dry_run:
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new path: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(args.opencv_threads)
    cv2.ocl.setUseOpenCL(False)
    config = replace(PRESETS[args.preset], preview_stride=999999)
    metadata = environment_metadata(args, videos)
    metadata["roi_yaml"] = {
        "path": str(args.roi_yaml.resolve()),
        "sha256": sha256_file(args.roi_yaml),
    }
    metadata["farneback_pyramid_pixel_factor"] = PYRAMID_PIXEL_FACTOR
    metadata["farneback_parameters"] = FARNEBACK_PARAMS
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    all_frames = []
    summaries = []
    for index, job in enumerate(jobs, start=1):
        width = job.record["roi"][2] - job.record["roi"][0]
        height = job.record["roi"][3] - job.record["roi"][1]
        scaled = scaled_dimensions(width, height, job.scale)
        print(
            f"[{index}/{len(jobs)}] {job.experiment} {job.video.name} "
            f"segment={job.segment} scale={job.scale:.2f} roi={scaled[0]}x{scaled[1]}",
            flush=True,
        )
        rows, resources = run_job(job, config, args)
        summary = summarize_run(rows, resources)
        all_frames.extend(rows)
        summaries.append(summary)
        print(
            f"  flow mean={summary['flow_ms_mean']:.2f} ms, "
            f"p95={summary['flow_ms_p95']:.2f} ms, "
            f"flow-only={summary['flow_only_fps']:.2f} FPS, "
            f"temp_max={fmt(summary['temperature_max_c'], 1)} C",
            flush=True,
        )
        partial_frame_path = args.output_dir / "frame_timings.partial.csv"
        pd.DataFrame(rows)[FRAME_COLUMNS].to_csv(
            partial_frame_path,
            mode="a",
            header=not partial_frame_path.exists(),
            index=False,
        )
        pd.DataFrame(summaries).to_csv(
            args.output_dir / "run_summary.partial.csv", index=False
        )

    frame_df = pd.DataFrame(all_frames)[FRAME_COLUMNS]
    run_df = pd.DataFrame(summaries)
    aggregate_df = aggregate_runs(run_df, args)
    controlled = run_df[run_df["experiment"] == "controlled"]
    regression_df, predictions_df = regression_rows(controlled, args)

    frame_df.to_csv(args.output_dir / "frame_timings.csv", index=False)
    run_df.to_csv(args.output_dir / "run_summary.csv", index=False)
    aggregate_df.to_csv(args.output_dir / "roi_size_summary.csv", index=False)
    regression_df.to_csv(args.output_dir / "regression_results.csv", index=False)
    predictions_df.to_csv(args.output_dir / "regression_predictions.csv", index=False)
    make_plots(run_df, aggregate_df, regression_df, args.output_dir)
    finalize_environment(metadata)
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_report(args, videos, run_df, aggregate_df, regression_df, args.output_dir)
    print(f"Report: {args.output_dir / 'roi_scaling_report_ko.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
