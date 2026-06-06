from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from src.bee_entrance_count import (
        Config,
        apply_component_area_filter,
        build_counting_boundary_band,
        build_entrance_mask,
        clamp_rect,
        compensate_global_flow,
        compute_optical_flow,
        compute_raw_flux,
        crop_roi,
        prepare_gray,
        update_persistence_filter,
        load_video_info,
    )
    from src.main import (
        COORDINATE_AUTO,
        COORDINATE_DEFAULT,
        COORDINATE_PRESETS,
        PRESETS,
        apply_coordinate_preset,
        apply_coordinate_rects,
        find_coordinate_preset,
    )
except ModuleNotFoundError:
    from bee_entrance_count import (
        Config,
        apply_component_area_filter,
        build_counting_boundary_band,
        build_entrance_mask,
        clamp_rect,
        compensate_global_flow,
        compute_optical_flow,
        compute_raw_flux,
        crop_roi,
        prepare_gray,
        update_persistence_filter,
        load_video_info,
    )
    from main import (
        COORDINATE_AUTO,
        COORDINATE_DEFAULT,
        COORDINATE_PRESETS,
        PRESETS,
        apply_coordinate_preset,
        apply_coordinate_rects,
        find_coordinate_preset,
    )


DEFAULT_OUTPUT_DIR = Path("analysis") / "video_features" / "output"
DEFAULT_VIDEO_DIR = Path("videos")
DEFAULT_PATTERNS = ["ANU-25-summer-*.mp4"]
WINDOW_SEC = 120.0
ANGLE_BINS = 12


def finite(value: float | int | np.floating | None) -> float:
    if value is None:
        return math.nan
    value = float(value)
    return value if math.isfinite(value) else math.nan


def safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return math.nan
    return float(numerator) / denominator


def parse_video_name(video_path: Path) -> dict[str, object]:
    match = re.search(r"ANU-25-summer-(\d+)_(\d{8})_(\d{6})", video_path.stem)
    if not match:
        return {
            "device": "",
            "date": "",
            "time": "",
            "datetime": "",
            "hour": math.nan,
        }

    date_text = match.group(2)
    time_text = match.group(3)
    timestamp = pd.to_datetime(date_text + time_text, format="%Y%m%d%H%M%S")
    return {
        "device": int(match.group(1)),
        "date": timestamp.date().isoformat(),
        "time": int(time_text),
        "datetime": timestamp.isoformat(),
        "hour": int(timestamp.hour),
    }


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return finite(np.percentile(values, q))


def entropy_from_uint8(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    hist = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    prob = hist[hist > 0] / values.size
    return finite(-np.sum(prob * np.log2(prob)))


def weighted_angle_entropy(dx: np.ndarray, dy: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    weights = weight[mask].astype(np.float64)
    if weights.sum() <= 0:
        return math.nan
    angles = np.arctan2(dy[mask], dx[mask])
    bins = np.linspace(-math.pi, math.pi, ANGLE_BINS + 1)
    hist, _ = np.histogram(angles, bins=bins, weights=weights)
    prob = hist[hist > 0] / hist.sum()
    return finite(-np.sum(prob * np.log2(prob)) / math.log2(ANGLE_BINS))


def canny_edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return safe_div(np.count_nonzero(edges), edges.size)


def laplacian_var(gray: np.ndarray) -> float:
    return finite(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_mean(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return finite(np.mean(gx * gx + gy * gy))


def gray_stats(prefix: str, gray: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = gray[mask] if mask is not None else gray.reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_gray_mean": math.nan,
            f"{prefix}_gray_std": math.nan,
            f"{prefix}_gray_p05": math.nan,
            f"{prefix}_gray_p50": math.nan,
            f"{prefix}_gray_p95": math.nan,
            f"{prefix}_gray_dynamic_range": math.nan,
            f"{prefix}_dark_ratio": math.nan,
            f"{prefix}_bright_ratio": math.nan,
            f"{prefix}_saturated_ratio": math.nan,
            f"{prefix}_gray_entropy": math.nan,
        }
    p05 = percentile(values, 5)
    p50 = percentile(values, 50)
    p95 = percentile(values, 95)
    return {
        f"{prefix}_gray_mean": finite(np.mean(values)),
        f"{prefix}_gray_std": finite(np.std(values)),
        f"{prefix}_gray_p05": p05,
        f"{prefix}_gray_p50": p50,
        f"{prefix}_gray_p95": p95,
        f"{prefix}_gray_dynamic_range": p95 - p05,
        f"{prefix}_dark_ratio": safe_div(np.count_nonzero(values < 30), values.size),
        f"{prefix}_bright_ratio": safe_div(np.count_nonzero(values > 225), values.size),
        f"{prefix}_saturated_ratio": safe_div(
            np.count_nonzero((values <= 5) | (values >= 250)),
            values.size,
        ),
        f"{prefix}_gray_entropy": entropy_from_uint8(values.astype(np.uint8)),
    }


def color_stats(prefix: str, roi: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    sat_values = sat[mask] if mask is not None else sat.reshape(-1)
    val_values = val[mask] if mask is not None else val.reshape(-1)
    if sat_values.size == 0:
        return {
            f"{prefix}_saturation_mean": math.nan,
            f"{prefix}_saturation_std": math.nan,
            f"{prefix}_value_mean": math.nan,
            f"{prefix}_value_std": math.nan,
        }
    return {
        f"{prefix}_saturation_mean": finite(np.mean(sat_values)),
        f"{prefix}_saturation_std": finite(np.std(sat_values)),
        f"{prefix}_value_mean": finite(np.mean(val_values)),
        f"{prefix}_value_std": finite(np.std(val_values)),
    }


def component_area_stats(prefix: str, mask: np.ndarray) -> dict[str, float]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if num_labels > 1 else np.array([])
    total_pixels = mask.size
    return {
        f"{prefix}_component_count": int(len(areas)),
        f"{prefix}_component_area_sum": finite(np.sum(areas)) if areas.size else 0.0,
        f"{prefix}_component_area_ratio": safe_div(np.sum(areas), total_pixels) if areas.size else 0.0,
        f"{prefix}_component_area_mean": finite(np.mean(areas)) if areas.size else math.nan,
        f"{prefix}_component_area_p50": percentile(areas, 50),
        f"{prefix}_component_area_p90": percentile(areas, 90),
        f"{prefix}_component_area_max": finite(np.max(areas)) if areas.size else 0.0,
    }


def build_edge_masks(shape: tuple[int, int], entrance_rect: tuple[int, int, int, int], band_px: int) -> dict[str, np.ndarray]:
    h, w = shape
    x1, y1, x2, y2 = entrance_rect
    yy, xx = np.indices((h, w))
    band = int(band_px)
    return {
        "top": (np.abs(yy - y1) <= band) & (xx >= max(0, x1 - band)) & (xx <= min(w - 1, x2 - 1 + band)),
        "bottom": (np.abs(yy - (y2 - 1)) <= band) & (xx >= max(0, x1 - band)) & (xx <= min(w - 1, x2 - 1 + band)),
        "left": (np.abs(xx - x1) <= band) & (yy >= max(0, y1 - band)) & (yy <= min(h - 1, y2 - 1 + band)),
        "right": (np.abs(xx - (x2 - 1)) <= band) & (yy >= max(0, y1 - band)) & (yy <= min(h - 1, y2 - 1 + band)),
    }


def summarize_rows(rows: list[dict[str, float]], prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict[str, float] = {}
    skip = {"frame", "time_sec", "window"}
    for column in df.columns:
        if column in skip:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        col = f"{prefix}{column}"
        out[f"{col}_mean"] = finite(values.mean())
        out[f"{col}_std"] = finite(values.std(ddof=0))
        out[f"{col}_p10"] = finite(values.quantile(0.10))
        out[f"{col}_p50"] = finite(values.quantile(0.50))
        out[f"{col}_p90"] = finite(values.quantile(0.90))
        out[f"{col}_min"] = finite(values.min())
        out[f"{col}_max"] = finite(values.max())
    return out


def window_summary(rows: list[dict[str, float]], video_name: str, window_sec: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["window"] = (df["time_sec"] // window_sec).astype(int)
    summary_rows = []
    for window, group in df.groupby("window"):
        row = {
            "video": video_name,
            "window": int(window),
            "start_sec": float(window * window_sec),
            "end_sec": float((window + 1) * window_sec),
            "sampled_frame_pairs": int(len(group)),
        }
        numeric = group.drop(columns=["frame", "time_sec", "window"], errors="ignore")
        for column in numeric.columns:
            values = pd.to_numeric(numeric[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            if column.endswith("_flux") or column.endswith("_pixels"):
                row[f"{column}_sum"] = finite(values.sum())
            row[f"{column}_mean"] = finite(values.mean())
            row[f"{column}_p90"] = finite(values.quantile(0.90))
            row[f"{column}_max"] = finite(values.max())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def unique_sorted_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in sorted(paths):
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def discover_videos(video_dir: Path, patterns: list[str]) -> list[Path]:
    video_dir = Path(video_dir)
    videos = []
    for pattern in patterns:
        videos.extend(video_dir.glob(pattern))
    return unique_sorted_paths(videos)


def select_videos(args: argparse.Namespace) -> list[Path]:
    videos = (
        unique_sorted_paths([Path(path) for path in args.videos])
        if args.videos
        else discover_videos(args.video_dir, args.pattern)
    )
    if args.start is not None or args.end is not None:
        videos = videos[args.start or 0 : args.end]
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise RuntimeError("No videos selected.")
    missing = [path for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing video files:\n" + "\n".join(str(path) for path in missing))
    return videos


def resolve_config(video_path: Path, base_config: Config, args: argparse.Namespace) -> Config:
    config = base_config
    if args.coordinate_preset == COORDINATE_AUTO:
        preset_name = find_coordinate_preset(video_path)
        if preset_name:
            config = apply_coordinate_preset(config, preset_name)
    elif args.coordinate_preset != COORDINATE_DEFAULT:
        config = apply_coordinate_preset(config, args.coordinate_preset)
    return apply_coordinate_rects(config, roi_rect=args.roi, entrance_rect=args.entrance)


def build_config(args: argparse.Namespace) -> Config:
    config = PRESETS[args.preset]
    updates = {}
    for name in [
        "boundary_band_px",
        "blur_kernel",
        "flow_mag_threshold",
        "normal_flow_threshold",
        "persist_decay",
        "persist_threshold",
        "min_flow_component_area",
        "balance_ratio_threshold",
    ]:
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    if args.no_persistence_filter:
        updates["use_persistence_filter"] = False
    if args.use_component_area_filter:
        updates["use_component_area_filter"] = True
    if args.no_component_area_filter:
        updates["use_component_area_filter"] = False
    if args.use_global_flow_compensation:
        updates["use_global_flow_compensation"] = True
    if args.use_bidirectional_balance_filter:
        updates["use_bidirectional_balance_filter"] = True
    return replace(config, **updates)


def frame_feature_row(
    frame_idx: int,
    time_sec: float,
    roi: np.ndarray,
    gray: np.ndarray,
    prev_gray: np.ndarray,
    entrance_mask: np.ndarray,
    counting_boundary_band: np.ndarray,
    edge_masks: dict[str, np.ndarray],
    background_mask: np.ndarray,
    persistence: np.ndarray,
    config: Config,
) -> tuple[dict[str, float], np.ndarray]:
    flow = compute_optical_flow(prev_gray, gray)
    global_dx = 0.0
    global_dy = 0.0
    if config.use_global_flow_compensation:
        flow, global_dx, global_dy = compensate_global_flow(flow, background_mask)

    entrance_bool = entrance_mask > 0
    diff = cv2.absdiff(prev_gray, gray)
    raw_data = compute_raw_flux(
        flow,
        counting_boundary_band,
        *build_normals_from_band(counting_boundary_band, edge_masks),
        config,
    )

    persistent_candidate, persistence = update_persistence_filter(raw_data["candidate"], persistence, config)
    filtered_candidate, component_stats = apply_component_area_filter(persistent_candidate, config)

    dx = raw_data["dx"]
    dy = raw_data["dy"]
    mag = raw_data["mag"]
    normal_flow = raw_data["normal_flow"]
    filtered_in_flux = float(np.sum(np.clip(normal_flow[filtered_candidate], 0, None)))
    filtered_out_flux = float(np.sum(np.clip(-normal_flow[filtered_candidate], 0, None)))
    raw_traffic = raw_data["raw_in_flux"] + raw_data["raw_out_flux"]
    filtered_traffic = filtered_in_flux + filtered_out_flux
    band_pixels = int(np.count_nonzero(counting_boundary_band))
    roi_pixels = int(gray.size)

    row: dict[str, float] = {
        "frame": frame_idx,
        "time_sec": time_sec,
        "raw_in_flux": raw_data["raw_in_flux"],
        "raw_out_flux": raw_data["raw_out_flux"],
        "raw_traffic_flux": raw_traffic,
        "filtered_in_flux": filtered_in_flux,
        "filtered_out_flux": filtered_out_flux,
        "filtered_traffic_flux": filtered_traffic,
        "filtered_to_raw_traffic_ratio": safe_div(filtered_traffic, raw_traffic),
        "direction_balance_abs": safe_div(abs(filtered_in_flux - filtered_out_flux), filtered_traffic),
        "in_flux_share": safe_div(filtered_in_flux, filtered_traffic),
        "out_flux_share": safe_div(filtered_out_flux, filtered_traffic),
        "raw_candidate_pixels": int(np.count_nonzero(raw_data["candidate"])),
        "persistent_candidate_pixels": int(np.count_nonzero(persistent_candidate)),
        "filtered_candidate_pixels": int(np.count_nonzero(filtered_candidate)),
        "raw_candidate_ratio_band": safe_div(np.count_nonzero(raw_data["candidate"]), band_pixels),
        "filtered_candidate_ratio_band": safe_div(np.count_nonzero(filtered_candidate), band_pixels),
        "persistence_mean": finite(np.mean(persistence)),
        "persistence_max": finite(np.max(persistence)),
        "global_flow_dx": finite(global_dx),
        "global_flow_dy": finite(global_dy),
        "global_flow_abs": finite(math.hypot(global_dx, global_dy)),
        "roi_flow_mag_mean": finite(np.mean(mag)),
        "roi_flow_mag_p90": percentile(mag.reshape(-1), 90),
        "roi_flow_mag_p99": percentile(mag.reshape(-1), 99),
        "roi_flow_active_ratio": safe_div(np.count_nonzero(mag > config.flow_mag_threshold), roi_pixels),
        "band_flow_mag_mean": finite(np.mean(mag[counting_boundary_band])),
        "band_flow_mag_p90": percentile(mag[counting_boundary_band], 90),
        "band_normal_abs_mean": finite(np.mean(np.abs(normal_flow[counting_boundary_band]))),
        "band_normal_abs_p90": percentile(np.abs(normal_flow[counting_boundary_band]), 90),
        "band_direction_entropy": weighted_angle_entropy(dx, dy, mag, counting_boundary_band),
        "candidate_direction_entropy": weighted_angle_entropy(dx, dy, mag, raw_data["candidate"]),
        "frame_diff_mean": finite(np.mean(diff)),
        "frame_diff_p90": percentile(diff.reshape(-1), 90),
        "frame_diff_changed_ratio_10": safe_div(np.count_nonzero(diff > 10), diff.size),
        "frame_diff_changed_ratio_25": safe_div(np.count_nonzero(diff > 25), diff.size),
        "roi_laplacian_var": laplacian_var(gray),
        "roi_tenengrad_mean": tenengrad_mean(gray),
        "roi_edge_density": canny_edge_density(gray),
    }
    row.update(gray_stats("roi", gray))
    row.update(gray_stats("entrance", gray, entrance_bool))
    row.update(gray_stats("background", gray, background_mask))
    row.update(color_stats("roi", roi))
    row.update(color_stats("entrance", roi, entrance_bool))
    row.update(color_stats("background", roi, background_mask))
    row.update(component_area_stats("raw_candidate", raw_data["candidate"]))
    row.update(component_area_stats("filtered_candidate", filtered_candidate))
    row.update(
        {
            "raw_component_count": component_stats.get("raw_component_count", math.nan),
            "valid_component_count": component_stats.get("valid_component_count", math.nan),
            "rejected_component_count": component_stats.get("rejected_component_count", math.nan),
        }
    )

    for edge, mask in edge_masks.items():
        edge_candidate = raw_data["candidate"] & mask
        edge_filtered = filtered_candidate & mask
        edge_candidate_normal = normal_flow[edge_candidate]
        edge_filtered_normal = normal_flow[edge_filtered]
        row[f"{edge}_raw_candidate_pixels"] = int(np.count_nonzero(edge_candidate))
        row[f"{edge}_filtered_candidate_pixels"] = int(np.count_nonzero(edge_filtered))
        row[f"{edge}_raw_in_flux"] = float(np.sum(np.clip(edge_candidate_normal, 0, None)))
        row[f"{edge}_raw_out_flux"] = float(np.sum(np.clip(-edge_candidate_normal, 0, None)))
        row[f"{edge}_filtered_in_flux"] = float(np.sum(np.clip(edge_filtered_normal, 0, None)))
        row[f"{edge}_filtered_out_flux"] = float(np.sum(np.clip(-edge_filtered_normal, 0, None)))

    return row, persistence


def build_normals_from_band(
    counting_boundary_band: np.ndarray,
    edge_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    normal_x = np.zeros(counting_boundary_band.shape, dtype=np.float32)
    normal_y = np.zeros(counting_boundary_band.shape, dtype=np.float32)
    normal_y[counting_boundary_band & edge_masks["top"]] = 1.0
    normal_y[counting_boundary_band & edge_masks["bottom"]] = -1.0
    normal_x[counting_boundary_band & edge_masks["left"]] = 1.0
    normal_x[counting_boundary_band & edge_masks["right"]] = -1.0
    return normal_x, normal_y


def extract_video_features(video_path: Path, output_dir: Path, config: Config, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    cap, info = load_video_info(video_path)
    fps = info["fps"]
    roi_rect = clamp_rect(
        (config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2),
        info["width"],
        info["height"],
        "ROI",
    )
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    first_roi = crop_roi(first_frame, roi_rect)
    roi_h, roi_w = first_roi.shape[:2]
    entrance_rect = (
        config.ent_x1 - roi_rect[0],
        config.ent_y1 - roi_rect[1],
        config.ent_x2 - roi_rect[0],
        config.ent_y2 - roi_rect[1],
    )
    entrance_mask = build_entrance_mask(first_roi.shape, entrance_rect)
    counting_boundary_band, _, _ = build_counting_boundary_band(entrance_mask, entrance_rect, config)
    edge_masks = build_edge_masks((roi_h, roi_w), entrance_rect, config.boundary_band_px)
    background_mask = entrance_mask == 0
    prev_gray = prepare_gray(first_roi, config)
    persistence = np.zeros((roi_h, roi_w), dtype=np.float32)

    frame_rows: list[dict[str, float]] = []
    frame_idx = 1
    while True:
        if args.max_frame_pairs is not None and len(frame_rows) >= args.max_frame_pairs:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_stride != 0:
            roi = crop_roi(frame, roi_rect)
            prev_gray = prepare_gray(roi, config)
            frame_idx += 1
            continue
        roi = crop_roi(frame, roi_rect)
        gray = prepare_gray(roi, config)
        row, persistence = frame_feature_row(
            frame_idx,
            frame_idx / fps,
            roi,
            gray,
            prev_gray,
            entrance_mask,
            counting_boundary_band,
            edge_masks,
            background_mask,
            persistence,
            config,
        )
        frame_rows.append(row)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    if not frame_rows:
        raise RuntimeError(f"No frame pairs processed: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_df = pd.DataFrame(frame_rows)
    if args.write_frame_csv:
        frame_df.to_csv(output_dir / f"{video_path.stem}_image_flow_features_frame.csv", index=False)

    window_df = window_summary(frame_rows, video_path.name, args.window_sec)
    summary: dict[str, object] = {
        "video": video_path.name,
        "video_path": str(video_path),
        **parse_video_name(video_path),
        "fps": fps,
        "width": info["width"],
        "height": info["height"],
        "frame_count": info["frame_count"],
        "video_duration_sec": safe_div(info["frame_count"], fps),
        "processed_frame_pairs": len(frame_rows),
        "frame_stride": args.frame_stride,
        "roi_x1": roi_rect[0],
        "roi_y1": roi_rect[1],
        "roi_x2": roi_rect[2],
        "roi_y2": roi_rect[3],
        "ent_x1": config.ent_x1,
        "ent_y1": config.ent_y1,
        "ent_x2": config.ent_x2,
        "ent_y2": config.ent_y2,
        "entrance_roi_x1": entrance_rect[0],
        "entrance_roi_y1": entrance_rect[1],
        "entrance_roi_x2": entrance_rect[2],
        "entrance_roi_y2": entrance_rect[3],
        "roi_area_px": roi_h * roi_w,
        "entrance_area_px": int(np.count_nonzero(entrance_mask)),
        "boundary_band_area_px": int(np.count_nonzero(counting_boundary_band)),
        "boundary_band_px": config.boundary_band_px,
        "blur_kernel": config.blur_kernel,
        "flow_mag_threshold": config.flow_mag_threshold,
        "normal_flow_threshold": config.normal_flow_threshold,
        "use_persistence_filter": config.use_persistence_filter,
        "persist_decay": config.persist_decay,
        "persist_threshold": config.persist_threshold,
        "use_component_area_filter": config.use_component_area_filter,
        "min_flow_component_area": config.min_flow_component_area,
        "use_global_flow_compensation": config.use_global_flow_compensation,
        "use_bidirectional_balance_filter": config.use_bidirectional_balance_filter,
        "balance_ratio_threshold": config.balance_ratio_threshold,
        "processing_time_sec": finite(time.perf_counter() - started),
    }
    summary.update(summarize_rows(frame_rows))

    total_raw_in = frame_df["raw_in_flux"].sum()
    total_raw_out = frame_df["raw_out_flux"].sum()
    total_filtered_in = frame_df["filtered_in_flux"].sum()
    total_filtered_out = frame_df["filtered_out_flux"].sum()
    total_raw_traffic = total_raw_in + total_raw_out
    total_filtered_traffic = total_filtered_in + total_filtered_out
    summary.update(
        {
            "total_raw_in_flux": finite(total_raw_in),
            "total_raw_out_flux": finite(total_raw_out),
            "total_raw_traffic_flux": finite(total_raw_traffic),
            "total_filtered_in_flux": finite(total_filtered_in),
            "total_filtered_out_flux": finite(total_filtered_out),
            "total_filtered_traffic_flux": finite(total_filtered_traffic),
            "total_filtered_to_raw_traffic_ratio": safe_div(total_filtered_traffic, total_raw_traffic),
            "total_direction_balance_abs": safe_div(abs(total_filtered_in - total_filtered_out), total_filtered_traffic),
            "total_in_flux_share": safe_div(total_filtered_in, total_filtered_traffic),
            "total_out_flux_share": safe_div(total_filtered_out, total_filtered_traffic),
        }
    )
    return summary, window_df


def write_data_dictionary(path: Path) -> None:
    rows = [
        ("roi_*", "ROI brightness, contrast, exposure, entropy, blur, edge and color features."),
        ("entrance_*", "Same image-quality features inside the entrance rectangle."),
        ("background_*", "Same image-quality features outside the entrance rectangle but inside ROI."),
        ("frame_diff_*", "Temporal image-change features between consecutive sampled frames."),
        ("*_flow_*", "Optical-flow magnitude, normal-flow and activity ratios."),
        ("raw_candidate_*", "Pixels/components passing raw optical-flow thresholds before persistence/area filters."),
        ("filtered_candidate_*", "Pixels/components remaining after persistence/area filters."),
        ("top/bottom/left/right_*", "Boundary-edge-specific candidate pixels and directional flux."),
        ("*_direction_entropy", "0-1 normalized direction entropy; higher means more mixed motion directions."),
        ("*_retention/ratio/share/balance", "Filter retention, in/out share and bidirectional balance indicators."),
    ]
    pd.DataFrame(rows, columns=["column_pattern", "meaning"]).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image-only and optical-flow reliability features from bee entrance videos."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help=(
            "One or more glob patterns under --video-dir. Example: "
            '--pattern "ANU-25-summer-3_*.mp4" "ANU-25-summer-20_*.mp4"'
        ),
    )
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="selected")
    parser.add_argument(
        "--coordinate-preset",
        choices=[COORDINATE_DEFAULT, COORDINATE_AUTO, *sorted(COORDINATE_PRESETS)],
        default=COORDINATE_AUTO,
    )
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--entrance", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frame-pairs", type=int)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--write-frame-csv", action="store_true")
    parser.add_argument("--boundary-band-px", type=int)
    parser.add_argument("--blur-kernel", type=int, choices=[1, 3, 5])
    parser.add_argument("--flow-mag-threshold", type=float)
    parser.add_argument("--normal-flow-threshold", type=float)
    parser.add_argument("--no-persistence-filter", action="store_true")
    parser.add_argument("--persist-decay", type=float)
    parser.add_argument("--persist-threshold", type=float)
    parser.add_argument("--use-component-area-filter", action="store_true")
    parser.add_argument("--no-component-area-filter", action="store_true")
    parser.add_argument("--min-flow-component-area", type=int)
    parser.add_argument("--use-global-flow-compensation", action="store_true")
    parser.add_argument("--use-bidirectional-balance-filter", action="store_true")
    parser.add_argument("--balance-ratio-threshold", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.frame_stride = max(1, args.frame_stride)
    videos = select_videos(args)
    base_config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"videos: {len(videos)}")
    print(f"output: {args.output_dir}")
    print(f"frame_stride: {args.frame_stride}")
    if args.dry_run:
        for video in videos:
            config = resolve_config(video, base_config, args)
            preset = find_coordinate_preset(video) if args.coordinate_preset == COORDINATE_AUTO else args.coordinate_preset
            print(
                f"{video} coords={preset or COORDINATE_DEFAULT} "
                f"roi=({config.roi_x1},{config.roi_y1},{config.roi_x2},{config.roi_y2}) "
                f"entrance=({config.ent_x1},{config.ent_y1},{config.ent_x2},{config.ent_y2})"
            )
        return

    summaries = []
    windows = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video}")
        config = resolve_config(video, base_config, args)
        summary, window_df = extract_video_features(video, args.output_dir, config, args)
        summaries.append(summary)
        if not window_df.empty:
            windows.append(window_df)

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / "video_image_flow_features.csv"
    summary_df.to_csv(summary_path, index=False)

    if windows:
        window_path = args.output_dir / "video_image_flow_features_windows.csv"
        pd.concat(windows, ignore_index=True).to_csv(window_path, index=False)
    else:
        window_path = args.output_dir / "video_image_flow_features_windows.csv"
        pd.DataFrame().to_csv(window_path, index=False)

    dictionary_path = args.output_dir / "video_image_flow_features_dictionary.csv"
    write_data_dictionary(dictionary_path)
    print(f"summary csv   : {summary_path}")
    print(f"window csv    : {window_path}")
    print(f"dictionary csv: {dictionary_path}")


if __name__ == "__main__":
    main()
