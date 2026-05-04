import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

video_id = "15"
VIDEO_PATH = f"videos/ANU-25-summer-6_20260404_{video_id}0000.mp4"

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1300, 1000, 1640, 1232
ENT_X1, ENT_Y1, ENT_X2, ENT_Y2 = 1400, 1200, 1600, 1232

BOUNDARY_BAND_PX = 8
FLOW_MAG_THRESHOLD = 0.30
NORMAL_FLOW_THRESHOLD = 0.08

IN_EVENT_START_THRESHOLD = 25.0
IN_EVENT_END_THRESHOLD = 8.0
OUT_EVENT_START_THRESHOLD = 25.0
OUT_EVENT_END_THRESHOLD = 8.0

MIN_EVENT_DURATION_SEC = 0.03
MAX_EVENT_DURATION_SEC = 2.0
COOLDOWN_SEC = 0.25

IN_BEE_EVENT_FLUX_UNIT = 100.0
OUT_BEE_EVENT_FLUX_UNIT = 100.0

WINDOW_SEC = 3.0

OUTPUT_DIR = Path("bee_count_output")
OUTPUT_PREVIEW_VIDEO = OUTPUT_DIR / f"entrance_count_preview_{video_id}.mp4"
OUTPUT_FRAME_CSV = OUTPUT_DIR / f"entrance_flux_frame_{video_id}.csv"
OUTPUT_EVENT_CSV = OUTPUT_DIR / f"entrance_events_{video_id}.csv"
OUTPUT_WINDOW_CSV = OUTPUT_DIR / f"entrance_count_3sec_{video_id}.csv"

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=4,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

DRAW_FLOW_ARROWS = True
ARROW_STEP = 12
ARROW_SCALE = 5
MIN_EVENT_COUNT_EST = 0.3


class FluxEventDetector:
    def __init__(
        self,
        start_threshold,
        end_threshold,
        min_duration_sec,
        max_duration_sec,
        cooldown_sec,
        flux_unit,
        direction_name,
    ):
        if end_threshold > start_threshold:
            raise ValueError("end_threshold must be <= start_threshold.")
        if flux_unit <= 0:
            raise ValueError("flux_unit must be positive.")

        self.start_threshold = float(start_threshold)
        self.end_threshold = float(end_threshold)
        self.min_duration_sec = float(min_duration_sec)
        self.max_duration_sec = float(max_duration_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.flux_unit = float(flux_unit)
        self.direction_name = direction_name

        self.state = "idle"
        self.cooldown_until_sec = 0.0
        self.current_event = None

    @property
    def is_active(self):
        return self.state == "active"

    def update(self, time_sec, flux_value):
        time_sec = float(time_sec)
        flux_value = float(flux_value)
        completed = []

        if self.state == "cooldown":
            if time_sec >= self.cooldown_until_sec:
                self.state = "idle"
            else:
                return completed

        if self.state == "idle":
            if flux_value >= self.start_threshold:
                self._start_event(time_sec, flux_value)
            return completed

        if self.state == "active":
            self._accumulate(time_sec, flux_value)
            duration = time_sec - self.current_event["start_sec"]
            should_force_close = duration >= self.max_duration_sec
            should_end = flux_value <= self.end_threshold

            if should_end or should_force_close:
                event = self._close_event(time_sec)
                if event is not None:
                    completed.append(event)
                self.state = "cooldown"
                self.cooldown_until_sec = time_sec + self.cooldown_sec

        return completed

    def finalize(self):
        if self.state != "active":
            return []

        event = self._close_event(self.current_event["last_sec"])
        self.state = "idle"
        return [event] if event is not None else []

    def _start_event(self, time_sec, flux_value):
        self.state = "active"
        self.current_event = {
            "start_sec": time_sec,
            "last_sec": time_sec,
            "flux_sum": flux_value,
            "peak_flux": flux_value,
        }

    def _accumulate(self, time_sec, flux_value):
        self.current_event["last_sec"] = time_sec
        self.current_event["flux_sum"] += flux_value
        self.current_event["peak_flux"] = max(
            self.current_event["peak_flux"],
            flux_value,
        )

    def _close_event(self, end_sec):
        event = self.current_event
        self.current_event = None

        duration_sec = max(0.0, end_sec - event["start_sec"])
        count_est = event["flux_sum"] / self.flux_unit

        if duration_sec < self.min_duration_sec or count_est < MIN_EVENT_COUNT_EST:
            return None

        return {
            "direction": self.direction_name,
            "start_sec": event["start_sec"],
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "flux_sum": event["flux_sum"],
            "peak_flux": event["peak_flux"],
            "count_est": count_est,
            "count_round": max(1, int(round(count_est))),
        }


def clamp_rect(x1, y1, x2, y2, width, height, name):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid {name} coordinates: {(x1, y1, x2, y2)}")

    return x1, y1, x2, y2


def validate_entrance_inside_roi(rect_x1, rect_y1, rect_x2, rect_y2, roi_w, roi_h):
    if rect_x1 < 0 or rect_y1 < 0 or rect_x2 > roi_w or rect_y2 > roi_h:
        raise ValueError(
            "Entrance rectangle must be fully inside ROI after coordinate conversion: "
            f"{(rect_x1, rect_y1, rect_x2, rect_y2)} inside ROI {(roi_w, roi_h)}"
        )
    if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
        raise ValueError(
            "Invalid entrance rectangle inside ROI: "
            f"{(rect_x1, rect_y1, rect_x2, rect_y2)}"
        )


def make_entrance_mask(roi_h, roi_w, rect_x1, rect_y1, rect_x2, rect_y2):
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask[rect_y1:rect_y2, rect_x1:rect_x2] = 255
    return mask


def build_boundary_normal_maps(
    entrance_mask,
    boundary_band_px,
    rect_x1,
    rect_y1,
    rect_x2,
    rect_y2,
):
    inside_dist = cv2.distanceTransform(entrance_mask, cv2.DIST_L2, 5)
    outside_dist = cv2.distanceTransform(255 - entrance_mask, cv2.DIST_L2, 5)
    signed_dist = inside_dist - outside_dist

    boundary_band = np.abs(signed_dist) <= boundary_band_px

    yy, xx = np.indices(entrance_mask.shape)
    bottom_exclude = (
        (yy >= rect_y2 - boundary_band_px - 2)
        & (xx >= rect_x1 - boundary_band_px)
        & (xx <= rect_x2 + boundary_band_px)
    )
    boundary_band[bottom_exclude] = False

    grad_x = cv2.Sobel(signed_dist, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed_dist, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6

    normal_x = grad_x / norm
    normal_y = grad_y / norm

    return boundary_band, normal_x, normal_y


def compute_boundary_flux(dx, dy, boundary_band, normal_x, normal_y):
    mag = np.sqrt(dx * dx + dy * dy)
    normal_flow = dx * normal_x + dy * normal_y
    valid = (
        boundary_band
        & (mag > FLOW_MAG_THRESHOLD)
        & (np.abs(normal_flow) > NORMAL_FLOW_THRESHOLD)
    )

    in_flux = float(np.sum(np.clip(normal_flow[valid], 0, None)))
    out_flux = float(np.sum(np.clip(-normal_flow[valid], 0, None)))

    return mag, normal_flow, valid, in_flux, out_flux


def draw_preview(
    roi,
    entrance_mask,
    boundary_band,
    rect,
    dx,
    dy,
    mag,
    normal_flow,
    valid,
    in_flux,
    out_flux,
    in_detector,
    out_detector,
    time_sec,
    total_in_round,
    total_out_round,
):
    vis = roi.copy()
    h, w = vis.shape[:2]
    rect_x1, rect_y1, rect_x2, rect_y2 = rect
    draw_x2 = min(rect_x2, w - 1)
    draw_y2 = min(rect_y2, h - 1)

    entrance_overlay = vis.copy()
    entrance_overlay[entrance_mask > 0] = (0, 80, 0)
    vis = cv2.addWeighted(vis, 0.78, entrance_overlay, 0.22, 0)

    band_overlay = np.zeros_like(vis)
    band_overlay[boundary_band] = (255, 255, 0)
    vis = cv2.addWeighted(vis, 0.86, band_overlay, 0.30, 0)

    cv2.rectangle(vis, (rect_x1, rect_y1), (draw_x2, draw_y2), (0, 0, 255), 1)
    cv2.line(vis, (rect_x1, draw_y2), (draw_x2, draw_y2), (150, 150, 150), 2)

    if DRAW_FLOW_ARROWS:
        for yy in range(0, h, ARROW_STEP):
            for xx in range(0, w, ARROW_STEP):
                if not valid[yy, xx]:
                    continue

                start = (xx, yy)
                end = (
                    int(xx + dx[yy, xx] * ARROW_SCALE),
                    int(yy + dy[yy, xx] * ARROW_SCALE),
                )
                color = (0, 255, 0) if normal_flow[yy, xx] > 0 else (255, 0, 0)
                cv2.arrowedLine(vis, start, end, color, 1, tipLength=0.3)

    status_color = (255, 255, 255)
    lines = [
        f"t={time_sec:.2f}s",
        f"IN flux={in_flux:.1f} state={in_detector.state}",
        f"OUT flux={out_flux:.1f} state={out_detector.state}",
        f"events IN={total_in_round} OUT={total_out_round} NET={total_in_round - total_out_round}",
        "counting band: top/left/right only",
    ]
    colors = [
        status_color,
        (0, 255, 0),
        (255, 0, 0),
        status_color,
        (255, 255, 0),
    ]

    for idx, (text, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            vis,
            text,
            (10, 24 + idx * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def summarize_events(events, total_duration_sec):
    columns = [
        "window",
        "start_sec",
        "end_sec",
        "in_event_count",
        "out_event_count",
        "in_count_est_sum",
        "out_count_est_sum",
        "in_count_round_sum",
        "out_count_round_sum",
        "net_count_round",
    ]

    if total_duration_sec <= 0:
        return pd.DataFrame(columns=columns)

    num_windows = max(1, int(math.ceil(total_duration_sec / WINDOW_SEC)))
    rows = []

    for window in range(num_windows):
        start_sec = window * WINDOW_SEC
        end_sec = start_sec + WINDOW_SEC
        rows.append(
            {
                "window": window,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "in_event_count": 0,
                "out_event_count": 0,
                "in_count_est_sum": 0.0,
                "out_count_est_sum": 0.0,
                "in_count_round_sum": 0,
                "out_count_round_sum": 0,
                "net_count_round": 0,
            }
        )

    for event in events:
        event_mid_sec = (event["start_sec"] + event["end_sec"]) / 2.0
        window = int(event_mid_sec // WINDOW_SEC)
        if window < 0:
            continue
        while window >= len(rows):
            start_sec = len(rows) * WINDOW_SEC
            rows.append(
                {
                    "window": len(rows),
                    "start_sec": start_sec,
                    "end_sec": start_sec + WINDOW_SEC,
                    "in_event_count": 0,
                    "out_event_count": 0,
                    "in_count_est_sum": 0.0,
                    "out_count_est_sum": 0.0,
                    "in_count_round_sum": 0,
                    "out_count_round_sum": 0,
                    "net_count_round": 0,
                }
            )

        row = rows[window]
        if event["direction"] == "IN":
            row["in_event_count"] += 1
            row["in_count_est_sum"] += event["count_est"]
            row["in_count_round_sum"] += event["count_round"]
        else:
            row["out_event_count"] += 1
            row["out_count_est_sum"] += event["count_est"]
            row["out_count_round_sum"] += event["count_round"]

        row["net_count_round"] = row["in_count_round_sum"] - row["out_count_round_sum"]

    return pd.DataFrame(rows, columns=columns)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}. "
            "Place the input video at this path or edit VIDEO_PATH."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Cannot read video frame size.")

    roi_x1, roi_y1, roi_x2, roi_y2 = clamp_rect(
        ROI_X1,
        ROI_Y1,
        ROI_X2,
        ROI_Y2,
        frame_w,
        frame_h,
        "ROI",
    )

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from video: {VIDEO_PATH}")

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]

    rect_x1 = ENT_X1 - roi_x1
    rect_y1 = ENT_Y1 - roi_y1
    rect_x2 = ENT_X2 - roi_x1
    rect_y2 = ENT_Y2 - roi_y1
    validate_entrance_inside_roi(rect_x1, rect_y1, rect_x2, rect_y2, roi_w, roi_h)

    entrance_mask = make_entrance_mask(roi_h, roi_w, rect_x1, rect_y1, rect_x2, rect_y2)
    boundary_band, normal_x, normal_y = build_boundary_normal_maps(
        entrance_mask,
        BOUNDARY_BAND_PX,
        rect_x1,
        rect_y1,
        rect_x2,
        rect_y2,
    )

    prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PREVIEW_VIDEO), fourcc, fps, (roi_w, roi_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {OUTPUT_PREVIEW_VIDEO}")

    in_detector = FluxEventDetector(
        IN_EVENT_START_THRESHOLD,
        IN_EVENT_END_THRESHOLD,
        MIN_EVENT_DURATION_SEC,
        MAX_EVENT_DURATION_SEC,
        COOLDOWN_SEC,
        IN_BEE_EVENT_FLUX_UNIT,
        "IN",
    )
    out_detector = FluxEventDetector(
        OUT_EVENT_START_THRESHOLD,
        OUT_EVENT_END_THRESHOLD,
        MIN_EVENT_DURATION_SEC,
        MAX_EVENT_DURATION_SEC,
        COOLDOWN_SEC,
        OUT_BEE_EVENT_FLUX_UNIT,
        "OUT",
    )

    frame_rows = []
    events = []
    total_in_round = 0
    total_out_round = 0
    frame_idx = 1
    last_time_sec = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag, normal_flow, valid, in_flux, out_flux = compute_boundary_flux(
            dx,
            dy,
            boundary_band,
            normal_x,
            normal_y,
        )

        time_sec = frame_idx / fps
        last_time_sec = time_sec

        completed_events = []
        completed_events.extend(in_detector.update(time_sec, in_flux))
        completed_events.extend(out_detector.update(time_sec, out_flux))

        for event in completed_events:
            events.append(event)
            if event["direction"] == "IN":
                total_in_round += event["count_round"]
            else:
                total_out_round += event["count_round"]

        frame_rows.append(
            {
                "frame": frame_idx,
                "time_sec": time_sec,
                "in_flux": in_flux,
                "out_flux": out_flux,
                "net_flux": in_flux - out_flux,
                "in_event_active": in_detector.is_active,
                "out_event_active": out_detector.is_active,
            }
        )

        preview = draw_preview(
            roi=roi,
            entrance_mask=entrance_mask,
            boundary_band=boundary_band,
            rect=(rect_x1, rect_y1, rect_x2, rect_y2),
            dx=dx,
            dy=dy,
            mag=mag,
            normal_flow=normal_flow,
            valid=valid,
            in_flux=in_flux,
            out_flux=out_flux,
            in_detector=in_detector,
            out_detector=out_detector,
            time_sec=time_sec,
            total_in_round=total_in_round,
            total_out_round=total_out_round,
        )
        writer.write(preview)

        prev_gray = gray
        frame_idx += 1

    events.extend(in_detector.finalize())
    events.extend(out_detector.finalize())

    cap.release()
    writer.release()

    if not frame_rows:
        raise RuntimeError("No frame pairs processed.")

    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(OUTPUT_FRAME_CSV, index=False)

    event_columns = [
        "direction",
        "start_sec",
        "end_sec",
        "duration_sec",
        "flux_sum",
        "peak_flux",
        "count_est",
        "count_round",
    ]
    event_df = pd.DataFrame(events, columns=event_columns)
    event_df.to_csv(OUTPUT_EVENT_CSV, index=False)

    summary_df = summarize_events(events, last_time_sec)
    summary_df.to_csv(OUTPUT_WINDOW_CSV, index=False)

    print("Done.")
    print(f"preview video: {OUTPUT_PREVIEW_VIDEO}")
    print(f"frame csv    : {OUTPUT_FRAME_CSV}")
    print(f"event csv    : {OUTPUT_EVENT_CSV}")
    print(f"window csv   : {OUTPUT_WINDOW_CSV}")
    print(f"ROI size     : {roi_w} x {roi_h}")
    print(f"Entrance ROI : {(rect_x1, rect_y1, rect_x2, rect_y2)}")
    print(f"FPS          : {fps:.3f}")


if __name__ == "__main__":
    main()
