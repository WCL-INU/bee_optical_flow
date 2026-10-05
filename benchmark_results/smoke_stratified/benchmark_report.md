# Raspberry Pi optical-flow real-time benchmark

Generated: 2026-09-06T06:26:41.388131+00:00

## Experimental design

- Input: 1 recorded Raspberry Pi camera videos
- Modes: realtime
- Temporal segments per video/mode: 3
- Frames requested per segment: 30
- Segment starts distributed across each video: True
- Warm-up frame pairs excluded per run: 3
- OpenCV worker threads: 2
- Optical-flow preset: `selected`
- Preview encoding: False
- Queue policy in realtime mode: latest frame, capacity 1
- Timing clock: Python `perf_counter_ns` (monotonic high-resolution clock)
- Confidence intervals: deterministic run-level non-parametric bootstrap, 10,000 resamples

The basic criterion is zero dropped frames and p95 end-to-end latency no greater than the recorded frame period. The recommended criterion reserves 20% of the frame period for camera/OS integration.

## Results

| Mode | Video | Runs | FPS mean ± SD | p95 latency mean ± SD (ms) | Deadline misses (%) | Drops (%) | Max temp (°C) | Basic | 20% margin |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| realtime | ANU-25-summer-1_20260702_120000.mp4 | 3 | 4.00 ± 0.14 | 377.15 ± 96.58 | 100.00 | 80.00 | 37.5 | FAIL | FAIL |

## Interpretation

At least one tested workload failed the nominal 24 FPS criterion. The current implementation cannot be described as reliably real-time across the evaluated workloads without optimization or frame-rate/ROI reduction.

## Validity and limitations

The videos preserve representative image content, resolution, motion, and compression complexity. Offline mode includes H.264 file decoding, whereas a CSI camera normally delivers ISP-processed YUV/RGB frames through libcamera; the source overheads are therefore not identical. Realtime mode reproduces arrival cadence, a bounded latest-frame queue, processing backlog, and dropping behavior, but it cannot measure sensor exposure, ISP, CSI transfer, or libcamera latency. A short Picamera2 validation using the same processor remains necessary before claiming complete sensor-to-result latency.

Runs share one physical device and are not independent Raspberry Pi specimens. When stratified starts are enabled, confidence intervals quantify variation among temporal scene segments on this device, not pure rerun noise or population-level hardware variation.

## Reproducibility artifacts

- `frame_timings.csv`: raw per-frame observations
- `run_summary.csv`: per-run statistics and resource measurements
- `aggregate_summary.csv`: repeated-run statistics and bootstrap intervals
- `environment.json`: software, hardware, Git, command, and input hashes
- `figures/`: publication-resolution timing figures

Total measured frame pairs: 6
