import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Config:
    roi_x1: int = 1020
    roi_y1: int = 980
    roi_x2: int = 1420
    roi_y2: int = 1280

    ent_x1: int = 1120
    ent_y1: int = 1080
    ent_x2: int = 1320
    ent_y2: int = 1180

    boundary_band_px: int = 8
    flow_mag_threshold: float = 0.30
    normal_flow_threshold: float = 0.08

    in_bee_flux_unit: float = 100.0
    out_bee_flux_unit: float = 100.0
    window_sec: float = 3.0

    blur_kernel: int = 3
    use_persistence_filter: bool = True
    persist_decay: float = 0.65
    persist_threshold: float = 1.3

    use_component_area_filter: bool = False
    min_flow_component_area: int = 30

    use_global_flow_compensation: bool = False
    use_bidirectional_balance_filter: bool = False
    balance_ratio_threshold: float = 0.70

    preview_stride: int = 1
    preview_panel_width: int = 360
    arrow_step: int = 10
    arrow_scale: float = 5.0

    warn_ratio_between_videos: float = 2.0
    warn_max_window_filtered_traffic_count_est: float = 3.0
    epsilon: float = 1e-6


FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=4,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

TIMING_COLUMNS = [
    "video",
    "video_path",
    "duration_sec",
    "processed_frame_pairs",
    "preprocessing_time_sec",
    "optical_flow_time_sec",
    "preprocessing_time_per_pair_sec",
    "optical_flow_time_per_pair_sec",
]


def save_timing_summary(summaries, output_dir, filename="processing_timing.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_df = pd.DataFrame(summaries)
    available_columns = [column for column in TIMING_COLUMNS if column in timing_df.columns]
    timing_path = output_dir / filename
    timing_df[available_columns].to_csv(timing_path, index=False)
    return timing_path


def load_video_info(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video with OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video frame size: {video_path}")

    return cap, {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def clamp_rect(rect, width, height, name):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid {name} rectangle: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def crop_roi(frame, roi_rect):
    x1, y1, x2, y2 = roi_rect
    return frame[y1:y2, x1:x2]


def build_entrance_mask(roi_shape, entrance_rect):
    roi_h, roi_w = roi_shape[:2]
    rect_x1, rect_y1, rect_x2, rect_y2 = entrance_rect

    if rect_x1 < 0 or rect_y1 < 0 or rect_x2 > roi_w or rect_y2 > roi_h:
        raise ValueError(
            "Entrance rectangle must be fully inside ROI: "
            f"{entrance_rect} inside ROI {(roi_w, roi_h)}"
        )
    if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
        raise ValueError(f"Invalid entrance rectangle: {entrance_rect}")

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask[rect_y1:rect_y2, rect_x1:rect_x2] = 255
    return mask


def build_counting_boundary_band(entrance_mask, entrance_rect, config):
    roi_h, roi_w = entrance_mask.shape
    rect_x1, rect_y1, rect_x2, rect_y2 = entrance_rect
    yy, xx = np.indices(entrance_mask.shape)

    band = int(config.boundary_band_px)
    edge_left = rect_x1
    edge_right = rect_x2 - 1
    edge_top = rect_y1
    edge_bottom = rect_y2 - 1

    expanded_rect = (
        (xx >= max(0, edge_left - band))
        & (xx <= min(roi_w - 1, edge_right + band))
        & (yy >= max(0, edge_top - band))
        & (yy <= min(roi_h - 1, edge_bottom + band))
    )

    distances = np.stack(
        [
            np.abs(yy - edge_top),
            np.abs(yy - edge_bottom),
            np.abs(xx - edge_left),
            np.abs(xx - edge_right),
        ],
        axis=0,
    )
    nearest_edge = np.argmin(distances, axis=0)
    boundary_band = expanded_rect & (np.min(distances, axis=0) <= band)

    normal_x = np.zeros(entrance_mask.shape, dtype=np.float32)
    normal_y = np.zeros(entrance_mask.shape, dtype=np.float32)
    normal_y[boundary_band & (nearest_edge == 0)] = 1.0
    normal_y[boundary_band & (nearest_edge == 1)] = -1.0
    normal_x[boundary_band & (nearest_edge == 2)] = 1.0
    normal_x[boundary_band & (nearest_edge == 3)] = -1.0

    return boundary_band, normal_x, normal_y


def prepare_gray(roi, config):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if config.blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    return gray


def compute_optical_flow(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)


def compensate_global_flow(flow, background_mask):
    compensated = flow.copy()
    if not np.any(background_mask):
        return compensated, 0.0, 0.0

    bg_dx = float(np.median(flow[..., 0][background_mask]))
    bg_dy = float(np.median(flow[..., 1][background_mask]))
    compensated[..., 0] -= bg_dx
    compensated[..., 1] -= bg_dy
    return compensated, bg_dx, bg_dy


def compute_raw_flux(flow, counting_boundary_band, normal_x, normal_y, config):
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    normal_flow = dx * normal_x + dy * normal_y

    candidate = (
        counting_boundary_band
        & (mag > config.flow_mag_threshold)
        & (np.abs(normal_flow) > config.normal_flow_threshold)
    )

    raw_in_flux = float(np.sum(np.clip(normal_flow[candidate], 0, None)))
    raw_out_flux = float(np.sum(np.clip(-normal_flow[candidate], 0, None)))

    return {
        "dx": dx,
        "dy": dy,
        "mag": mag,
        "normal_flow": normal_flow,
        "candidate": candidate,
        "raw_in_flux": raw_in_flux,
        "raw_out_flux": raw_out_flux,
    }


def apply_bidirectional_balance_filter(in_flux, out_flux, config):
    if not config.use_bidirectional_balance_filter:
        return in_flux, out_flux

    larger = max(in_flux, out_flux)
    smaller = min(in_flux, out_flux)
    if larger <= config.epsilon:
        return in_flux, out_flux

    balance_ratio = smaller / larger
    if balance_ratio < config.balance_ratio_threshold:
        return in_flux, out_flux

    residual = larger - smaller
    if in_flux >= out_flux:
        return residual, 0.0
    return 0.0, residual


def update_persistence_filter(candidate, persistence, config):
    if not config.use_persistence_filter:
        return candidate.copy(), persistence

    persistence *= config.persist_decay
    persistence += candidate.astype(np.float32)
    persistent_candidate = persistence > config.persist_threshold
    return candidate & persistent_candidate, persistence


def apply_component_area_filter(candidate, config):
    stats = {
        "raw_component_count": 0,
        "valid_component_count": 0,
        "rejected_component_count": 0,
    }
    if not config.use_component_area_filter:
        return candidate, stats

    mask = candidate.astype(np.uint8)
    num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros(candidate.shape, dtype=bool)
    stats["raw_component_count"] = max(0, num_labels - 1)

    for label in range(1, num_labels):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area >= config.min_flow_component_area:
            filtered[labels == label] = True
            stats["valid_component_count"] += 1
        else:
            stats["rejected_component_count"] += 1

    return filtered, stats


def aggregate_window_counts(frame_df, config):
    columns = [
        "window",
        "start_sec",
        "end_sec",
        "raw_in_flux_sum",
        "raw_out_flux_sum",
        "filtered_in_flux_sum",
        "filtered_out_flux_sum",
        "raw_in_count_est",
        "raw_out_count_est",
        "filtered_in_count_est",
        "filtered_out_count_est",
        "filtered_traffic_count_est",
        "filtered_net_count_est",
        "filtered_in_count_round",
        "filtered_out_count_round",
    ]
    if frame_df.empty:
        return pd.DataFrame(columns=columns)

    df = frame_df.copy()
    df["window"] = (df["time_sec"] // config.window_sec).astype(int)
    summary = (
        df.groupby("window", as_index=False)
        .agg(
            raw_in_flux_sum=("raw_in_flux", "sum"),
            raw_out_flux_sum=("raw_out_flux", "sum"),
            filtered_in_flux_sum=("filtered_in_flux", "sum"),
            filtered_out_flux_sum=("filtered_out_flux", "sum"),
        )
        .sort_values("window")
    )
    summary["start_sec"] = summary["window"] * config.window_sec
    summary["end_sec"] = summary["start_sec"] + config.window_sec
    summary["raw_in_count_est"] = summary["raw_in_flux_sum"] / config.in_bee_flux_unit
    summary["raw_out_count_est"] = summary["raw_out_flux_sum"] / config.out_bee_flux_unit
    summary["filtered_in_count_est"] = (
        summary["filtered_in_flux_sum"] / config.in_bee_flux_unit
    )
    summary["filtered_out_count_est"] = (
        summary["filtered_out_flux_sum"] / config.out_bee_flux_unit
    )
    summary["filtered_traffic_count_est"] = (
        summary["filtered_in_count_est"] + summary["filtered_out_count_est"]
    )
    summary["filtered_net_count_est"] = (
        summary["filtered_in_count_est"] - summary["filtered_out_count_est"]
    )
    summary["filtered_in_count_round"] = summary["filtered_in_count_est"].round().astype(int)
    summary["filtered_out_count_round"] = (
        summary["filtered_out_count_est"].round().astype(int)
    )
    return summary[columns]


def draw_preview(
    roi,
    entrance_rect,
    counting_boundary_band,
    raw_candidate,
    filtered_candidate,
    raw_data,
    frame_row,
    time_sec,
    config,
):
    vis = roi.copy()
    h, w = vis.shape[:2]
    rect_x1, rect_y1, rect_x2, rect_y2 = entrance_rect
    draw_x2 = min(rect_x2 - 1, w - 1)
    draw_y2 = min(rect_y2 - 1, h - 1)

    raw_overlay = vis.copy()
    raw_overlay[raw_candidate] = (0, 220, 220)
    vis = cv2.addWeighted(vis, 0.88, raw_overlay, 0.12, 0)

    band_overlay = np.zeros_like(vis)
    band_overlay[counting_boundary_band] = (255, 255, 0)
    vis = cv2.addWeighted(vis, 0.86, band_overlay, 0.28, 0)

    cv2.rectangle(vis, (rect_x1, rect_y1), (draw_x2, draw_y2), (0, 0, 255), 1)

    dx = raw_data["dx"]
    dy = raw_data["dy"]
    normal_flow = raw_data["normal_flow"]
    for yy in range(0, h, config.arrow_step):
        for xx in range(0, w, config.arrow_step):
            if not filtered_candidate[yy, xx]:
                continue
            end = (
                int(xx + dx[yy, xx] * config.arrow_scale),
                int(yy + dy[yy, xx] * config.arrow_scale),
            )
            color = (0, 255, 0) if normal_flow[yy, xx] > 0 else (255, 0, 0)
            cv2.arrowedLine(vis, (xx, yy), end, color, 1, tipLength=0.3)

    panel_w = max(1, int(config.preview_panel_width))
    panel = np.full((h, panel_w, 3), (32, 34, 36), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, h - 1), (42, 45, 48), 1)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, 34), (20, 23, 26), -1)

    def put_panel_text(text, y, color=(235, 235, 235), scale=0.45, thickness=1):
        cv2.putText(
            panel,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    put_panel_text("Bee entrance preview", 23, (255, 255, 255), 0.52, 1)
    put_panel_text(f"t = {time_sec:.2f}s", 52)
    put_panel_text("Boundary: top / bottom / left / right", 72, (255, 255, 0), 0.42)

    put_panel_text("Raw flux", 101, (0, 220, 220), 0.47)
    put_panel_text(
        f"IN {frame_row['raw_in_flux']:8.1f}   OUT {frame_row['raw_out_flux']:8.1f}",
        121,
        (235, 235, 235),
        0.42,
    )

    put_panel_text("Filtered flux", 150, (0, 255, 0), 0.47)
    put_panel_text(
        (
            f"IN {frame_row['filtered_in_flux']:8.1f}   "
            f"OUT {frame_row['filtered_out_flux']:8.1f}"
        ),
        170,
        (235, 235, 235),
        0.42,
    )

    put_panel_text(
        (
            f"Pixels raw/filter: {frame_row['raw_candidate_pixels']} / "
            f"{frame_row['filtered_candidate_pixels']}"
        ),
        199,
        (235, 235, 235),
        0.42,
    )
    put_panel_text(f"Persistence max: {frame_row['persistence_max']:.2f}", 219, scale=0.42)

    canvas = np.concatenate([vis, panel], axis=1)
    cv2.line(canvas, (w, 0), (w, h - 1), (12, 12, 12), 2)
    return canvas


def process_video(video_path, output_dir, config):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, info = load_video_info(video_path)
    fps = info["fps"]
    roi_rect = clamp_rect(
        (config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2),
        info["width"],
        info["height"],
        "ROI",
    )

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    preprocessing_time_sec = 0.0
    optical_flow_time_sec = 0.0

    preprocess_start = time.perf_counter()
    first_roi = crop_roi(first_frame, roi_rect)
    preprocessing_time_sec += time.perf_counter() - preprocess_start

    roi_h, roi_w = first_roi.shape[:2]
    entrance_rect = (
        config.ent_x1 - roi_rect[0],
        config.ent_y1 - roi_rect[1],
        config.ent_x2 - roi_rect[0],
        config.ent_y2 - roi_rect[1],
    )
    entrance_mask = build_entrance_mask(first_roi.shape, entrance_rect)
    counting_boundary_band, normal_x, normal_y = build_counting_boundary_band(
        entrance_mask,
        entrance_rect,
        config,
    )
    background_mask = entrance_mask == 0

    stem = video_path.stem
    preview_path = output_dir / f"{stem}_preview.mp4"
    frame_csv = output_dir / f"{stem}_frame_flux.csv"
    window_csv = output_dir / f"{stem}_window_3sec.csv"

    preview_fps = max(1.0, fps / max(1, config.preview_stride))
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        preview_fps,
        (roi_w + max(1, int(config.preview_panel_width)), roi_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open preview writer: {preview_path}")

    preprocess_start = time.perf_counter()
    prev_gray = prepare_gray(first_roi, config)
    preprocessing_time_sec += time.perf_counter() - preprocess_start

    persistence = np.zeros((roi_h, roi_w), dtype=np.float32)
    frame_rows = []
    frame_idx = 1
    last_time_sec = 0.0

    while True:
        # For testing, limit to first 2 minutes (assuming 24 fps -> 2880 frames)
        if frame_idx >= 2880: 
            break

        ret, frame = cap.read()
        if not ret:
            break

        preprocess_start = time.perf_counter()
        roi = crop_roi(frame, roi_rect)
        gray = prepare_gray(roi, config)
        preprocessing_time_sec += time.perf_counter() - preprocess_start

        flow_start = time.perf_counter()
        flow = compute_optical_flow(prev_gray, gray)
        optical_flow_time_sec += time.perf_counter() - flow_start

        if config.use_global_flow_compensation:
            flow, _, _ = compensate_global_flow(flow, background_mask)
        raw_data = compute_raw_flux(flow, counting_boundary_band, normal_x, normal_y, config)

        persistent_candidate, persistence = update_persistence_filter(
            raw_data["candidate"],
            persistence,
            config,
        )
        filtered_candidate, component_stats = apply_component_area_filter(
            persistent_candidate,
            config,
        )

        normal_flow = raw_data["normal_flow"]
        filtered_in_flux = float(np.sum(np.clip(normal_flow[filtered_candidate], 0, None)))
        filtered_out_flux = float(np.sum(np.clip(-normal_flow[filtered_candidate], 0, None)))
        filtered_in_flux, filtered_out_flux = apply_bidirectional_balance_filter(
            filtered_in_flux,
            filtered_out_flux,
            config,
        )

        time_sec = frame_idx / fps
        last_time_sec = time_sec
        frame_row = {
            "frame": frame_idx,
            "time_sec": time_sec,
            "raw_in_flux": raw_data["raw_in_flux"],
            "raw_out_flux": raw_data["raw_out_flux"],
            "filtered_in_flux": filtered_in_flux,
            "filtered_out_flux": filtered_out_flux,
            "raw_candidate_pixels": int(np.count_nonzero(raw_data["candidate"])),
            "persistent_candidate_pixels": int(np.count_nonzero(persistent_candidate)),
            "filtered_candidate_pixels": int(np.count_nonzero(filtered_candidate)),
            "persistence_mean": float(np.mean(persistence)),
            "persistence_max": float(np.max(persistence)),
        }
        if config.use_component_area_filter:
            frame_row.update(component_stats)
        frame_rows.append(frame_row)

        if frame_idx % max(1, config.preview_stride) == 0:
            writer.write(
                draw_preview(
                    roi,
                    entrance_rect,
                    counting_boundary_band,
                    raw_data["candidate"],
                    filtered_candidate,
                    raw_data,
                    frame_row,
                    time_sec,
                    config,
                )
            )

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()

    if not frame_rows:
        raise RuntimeError(f"No frame pairs processed: {video_path}")

    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(frame_csv, index=False)

    window_df = aggregate_window_counts(frame_df, config)
    window_df.to_csv(window_csv, index=False)

    total_raw_in_flux = float(frame_df["raw_in_flux"].sum())
    total_raw_out_flux = float(frame_df["raw_out_flux"].sum())
    total_filtered_in_flux = float(frame_df["filtered_in_flux"].sum())
    total_filtered_out_flux = float(frame_df["filtered_out_flux"].sum())
    total_raw_traffic_flux = total_raw_in_flux + total_raw_out_flux
    total_filtered_traffic_flux = total_filtered_in_flux + total_filtered_out_flux
    duration_sec = max(last_time_sec, config.epsilon)

    raw_to_filtered_reduction_ratio = total_raw_traffic_flux / max(
        total_filtered_traffic_flux,
        config.epsilon,
    )
    processed_frame_pairs = len(frame_rows)
    preprocessing_time_per_pair_sec = preprocessing_time_sec / max(
        processed_frame_pairs,
        1,
    )
    optical_flow_time_per_pair_sec = optical_flow_time_sec / max(
        processed_frame_pairs,
        1,
    )

    return {
        "video": video_path.name,
        "video_path": str(video_path),
        "duration_sec": last_time_sec,
        "processed_frame_pairs": processed_frame_pairs,
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
        "boundary_band_px": config.boundary_band_px,
        "preprocessing_time_sec": preprocessing_time_sec,
        "optical_flow_time_sec": optical_flow_time_sec,
        "preprocessing_time_per_pair_sec": preprocessing_time_per_pair_sec,
        "optical_flow_time_per_pair_sec": optical_flow_time_per_pair_sec,
        "blur_kernel": config.blur_kernel,
        "use_persistence_filter": config.use_persistence_filter,
        "persist_decay": config.persist_decay,
        "persist_threshold": config.persist_threshold,
        "use_component_area_filter": config.use_component_area_filter,
        "min_flow_component_area": config.min_flow_component_area,
        "use_global_flow_compensation": config.use_global_flow_compensation,
        "use_bidirectional_balance_filter": config.use_bidirectional_balance_filter,
        "balance_ratio_threshold": config.balance_ratio_threshold,
        "total_raw_in_flux": total_raw_in_flux,
        "total_raw_out_flux": total_raw_out_flux,
        "total_filtered_in_flux": total_filtered_in_flux,
        "total_filtered_out_flux": total_filtered_out_flux,
        "total_raw_traffic_flux": total_raw_traffic_flux,
        "total_filtered_traffic_flux": total_filtered_traffic_flux,
        "mean_raw_traffic_flux_per_sec": total_raw_traffic_flux / duration_sec,
        "mean_filtered_traffic_flux_per_sec": total_filtered_traffic_flux / duration_sec,
        "max_window_filtered_traffic_count_est": (
            float(window_df["filtered_traffic_count_est"].max()) if not window_df.empty else 0.0
        ),
        "mean_window_filtered_traffic_count_est": (
            float(window_df["filtered_traffic_count_est"].mean()) if not window_df.empty else 0.0
        ),
        "total_filtered_in_count_est": total_filtered_in_flux / config.in_bee_flux_unit,
        "total_filtered_out_count_est": total_filtered_out_flux / config.out_bee_flux_unit,
        "total_filtered_traffic_count_est": (
            total_filtered_in_flux / config.in_bee_flux_unit
            + total_filtered_out_flux / config.out_bee_flux_unit
        ),
        "raw_to_filtered_reduction_ratio": raw_to_filtered_reduction_ratio,
        "preview_video": str(preview_path),
        "frame_csv": str(frame_csv),
        "window_csv": str(window_csv),
    }, frame_df, window_df


def compare_videos(video_paths, output_dir, config, config_for_video=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx, video_path in enumerate(video_paths, start=1):
        print(f"[{idx}/{len(video_paths)}] Processing {Path(video_path)}")
        video_config = (
            config_for_video(video_path, config) if config_for_video is not None else config
        )
        result, _, _ = process_video(video_path, output_dir, video_config)
        summaries.append(result)

    summary_columns = [
        "video",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "ent_x1",
        "ent_y1",
        "ent_x2",
        "ent_y2",
        "boundary_band_px",
        "blur_kernel",
        "use_persistence_filter",
        "persist_decay",
        "persist_threshold",
        "use_component_area_filter",
        "min_flow_component_area",
        "use_global_flow_compensation",
        "use_bidirectional_balance_filter",
        "balance_ratio_threshold",
        "total_raw_in_flux",
        "total_raw_out_flux",
        "total_filtered_in_flux",
        "total_filtered_out_flux",
        "total_raw_traffic_flux",
        "total_filtered_traffic_flux",
        "mean_raw_traffic_flux_per_sec",
        "mean_filtered_traffic_flux_per_sec",
        "max_window_filtered_traffic_count_est",
        "mean_window_filtered_traffic_count_est",
        "total_filtered_in_count_est",
        "total_filtered_out_count_est",
        "total_filtered_traffic_count_est",
        "raw_to_filtered_reduction_ratio",
    ]
    summary_df = pd.DataFrame(summaries)
    summary_df[summary_columns].to_csv(output_dir / "comparison_summary.csv", index=False)
    timing_path = save_timing_summary(summaries, output_dir)
    print(f"timing summary: {timing_path}")

    for row in summaries:
        print(
            f"{row['video']}: raw traffic={row['total_raw_traffic_flux']:.1f}, "
            f"filtered traffic={row['total_filtered_traffic_flux']:.1f}, "
            f"reduction ratio={row['raw_to_filtered_reduction_ratio']:.3f}"
        )

    mean_values = [row["mean_filtered_traffic_flux_per_sec"] for row in summaries]
    max_mean = max(mean_values) if mean_values else 0.0
    min_mean = min(mean_values) if mean_values else 0.0
    if max_mean <= config.epsilon:
        ratio_between_videos = 1.0
    else:
        ratio_between_videos = max_mean / max(min_mean, config.epsilon)

    mean_diff = max_mean - min_mean if mean_values else 0.0
    max_window = max(
        (row["max_window_filtered_traffic_count_est"] for row in summaries),
        default=0.0,
    )

    print(f"mean filtered traffic flux/sec difference: {mean_diff:.3f}")
    print(f"ratio between videos: {ratio_between_videos:.3f}")
    print(f"max window filtered traffic count estimate: {max_window:.3f}")

    warnings = []
    if ratio_between_videos > config.warn_ratio_between_videos:
        warnings.append(
            f"ratio_between_videos {ratio_between_videos:.3f} > "
            f"{config.warn_ratio_between_videos:.3f}"
        )
    if max_window > config.warn_max_window_filtered_traffic_count_est:
        warnings.append(
            f"max_window_filtered_traffic_count_est {max_window:.3f} > "
            f"{config.warn_max_window_filtered_traffic_count_est:.3f}"
        )

    if warnings:
        print("WARNING")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("PASS")

    return summary_df, warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bee entrance optical-flow persistence filtering."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Process one video.")
    group.add_argument("--compare", nargs="+", type=Path, help="Compare videos.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bee_count_output") / "persistence_filter",
    )
    parser.add_argument("--blur-kernel", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Full-frame ROI rectangle.",
    )
    parser.add_argument(
        "--entrance",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Full-frame entrance boundary rectangle.",
    )
    parser.add_argument("--boundary-band-px", type=int, default=8)
    parser.add_argument("--no-persistence-filter", action="store_true")
    parser.add_argument("--persist-decay", type=float, default=0.65)
    parser.add_argument("--persist-threshold", type=float, default=1.3)
    parser.add_argument("--use-component-area-filter", action="store_true")
    parser.add_argument("--min-flow-component-area", type=int, default=30)
    parser.add_argument("--use-global-flow-compensation", action="store_true")
    parser.add_argument("--use-bidirectional-balance-filter", action="store_true")
    parser.add_argument("--balance-ratio-threshold", type=float, default=0.70)
    parser.add_argument("--flow-mag-threshold", type=float, default=0.30)
    parser.add_argument("--normal-flow-threshold", type=float, default=0.08)
    parser.add_argument("--preview-stride", type=int, default=1)
    parser.add_argument("--preview-panel-width", type=int, default=360)
    parser.add_argument("--warn-ratio-between-videos", type=float, default=2.0)
    parser.add_argument("--warn-max-window-filtered-traffic-count-est", type=float, default=3.0)
    return parser.parse_args()


def config_from_args(args):
    updates = {
        "blur_kernel": args.blur_kernel,
        "boundary_band_px": args.boundary_band_px,
        "use_persistence_filter": not args.no_persistence_filter,
        "persist_decay": args.persist_decay,
        "persist_threshold": args.persist_threshold,
        "use_component_area_filter": args.use_component_area_filter,
        "min_flow_component_area": args.min_flow_component_area,
        "use_global_flow_compensation": args.use_global_flow_compensation,
        "use_bidirectional_balance_filter": args.use_bidirectional_balance_filter,
        "balance_ratio_threshold": args.balance_ratio_threshold,
        "flow_mag_threshold": args.flow_mag_threshold,
        "normal_flow_threshold": args.normal_flow_threshold,
        "preview_stride": max(1, args.preview_stride),
        "preview_panel_width": max(1, args.preview_panel_width),
        "warn_ratio_between_videos": args.warn_ratio_between_videos,
        "warn_max_window_filtered_traffic_count_est": (
            args.warn_max_window_filtered_traffic_count_est
        ),
    }
    if args.roi is not None:
        updates.update(
            {
                "roi_x1": args.roi[0],
                "roi_y1": args.roi[1],
                "roi_x2": args.roi[2],
                "roi_y2": args.roi[3],
            }
        )
    if args.entrance is not None:
        updates.update(
            {
                "ent_x1": args.entrance[0],
                "ent_y1": args.entrance[1],
                "ent_x2": args.entrance[2],
                "ent_y2": args.entrance[3],
            }
        )
    return Config(**updates)


def main():
    args = parse_args()
    config = config_from_args(args)

    if args.video is not None:
        result, _, _ = process_video(args.video, args.output_dir, config)
        timing_path = save_timing_summary([result], args.output_dir)
        print("Done.")
        print(f"preview video: {result['preview_video']}")
        print(f"frame csv    : {result['frame_csv']}")
        print(f"window csv   : {result['window_csv']}")
        print(f"timing csv   : {timing_path}")
        return

    if len(args.compare) < 2:
        raise ValueError("--compare requires at least two video paths.")
    compare_videos(args.compare, args.output_dir, config)


if __name__ == "__main__":
    main()
