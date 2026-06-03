# Bee Entrance Event Counting

## 1. Problem definition

This project estimates bee entrance activity from video using optical flow around a known hive entrance rectangle. The task is not bee detection, YOLO tracking, or maintaining stable IDs for individual bees.

The goal is to estimate crossing events at the entrance boundary. A useful output is a short-window summary such as "0-3 seconds: IN 2, OUT 0" or "3-6 seconds: IN 0, OUT 1". Exact absolute counts can be imperfect as long as the estimate is proportional to entrance activity.

## 2. Limitations of frame-level flux accumulation

The previous approach accumulated frame-level boundary flux directly into IN/OUT counts. That can over-count when a bee moves slowly across the boundary, stays on top of the boundary, or moves back and forth near the entrance edge. In those cases, many consecutive frames can contain flow from the same physical crossing, so summing every frame as count inflates the result.

## 3. New algorithm overview

1. Crop the full video frame to the configured ROI.
2. Convert the entrance rectangle from full-frame coordinates into ROI coordinates.
3. Create an entrance rectangle mask.
4. Create boundary bands for the four rectangle edges.
5. Create inward normal maps for the top, bottom, left, and right edges.
6. Create a boundary band around the entrance rectangle.
7. Use all four entrance boundaries for counting.
8. Compute Farneback optical flow between consecutive ROI frames.
9. Project flow onto the boundary normal direction.
10. Compute frame-level IN and OUT flux on the top, bottom, left, and right boundary bands.
11. Feed IN and OUT flux time series into separate hysteresis event detectors.
12. Convert each completed event into an estimated bee count.
13. Aggregate event counts into 3-second windows.

## 4. IN/OUT definition

The boundary normal points from outside the entrance rectangle toward the inside. A positive projected normal flow means motion from outside the entrance rectangle toward the inside, and is defined as IN.

A negative projected normal flow means motion from inside the entrance rectangle toward the outside, and is defined as OUT.

If visual inspection shows that the biological interpretation is reversed for a specific camera setup, the result can be interpreted by swapping the IN and OUT labels.

## 5. Four-boundary accumulation

The configured entrance rectangle is:

```python
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1300, 1000, 1640, 1232
ENT_X1, ENT_Y1, ENT_X2, ENT_Y2 = 1400, 1200, 1600, 1232
```

Here `ENT_Y2` equals `ROI_Y2`, so the entrance bottom edge touches the bottom of the ROI. The current counting policy still includes this bottom edge and accumulates optical-flow flux on all four rectangle boundaries. This makes the preview and CSV summaries match the current 4-direction accumulation rule.

## 6. Event detector design

IN and OUT flux are processed by separate detectors. Each detector has three states: `idle`, `active`, and `cooldown`.

The detector uses hysteresis:

- In `idle`, an event starts when flux is greater than or equal to the start threshold.
- In `active`, flux is accumulated into the current event.
- In `active`, the event ends when flux falls to or below the end threshold.
- The start threshold is higher than the end threshold, so brief noise near the threshold is less likely to split or start events.
- Consecutive same-direction flux above the end threshold is grouped into one event.
- Events shorter than the minimum duration are discarded.
- Events longer than the maximum duration are force-closed as one long event.
- After an event ends, the detector enters `cooldown`; same-direction restarts are ignored until cooldown expires.
- The default cooldown behavior is to ignore flux during cooldown, not attach it back to the previous event.

Large events can represent multiple bees. Each event computes:

```python
count_est = event_flux_sum / BEE_EVENT_FLUX_UNIT
count_round = max(1, round(count_est))
```

Very small events can be discarded when `count_est < 0.3`.

## 7. Main parameters

- `BOUNDARY_BAND_PX`: Width in pixels of the counting band around the entrance boundary.
- `FLOW_MAG_THRESHOLD`: Minimum optical-flow magnitude for a pixel to contribute to flux.
- `NORMAL_FLOW_THRESHOLD`: Minimum projected normal-flow magnitude for a pixel to contribute.
- `IN_EVENT_START_THRESHOLD`: IN event start threshold.
- `IN_EVENT_END_THRESHOLD`: IN event end threshold.
- `OUT_EVENT_START_THRESHOLD`: OUT event start threshold.
- `OUT_EVENT_END_THRESHOLD`: OUT event end threshold.
- `MIN_EVENT_DURATION_SEC`: Minimum duration for a completed event to be valid.
- `MAX_EVENT_DURATION_SEC`: Maximum duration before an active event is force-closed.
- `COOLDOWN_SEC`: Time after an event during which same-direction restarts are ignored.
- `IN_BEE_EVENT_FLUX_UNIT`: Flux sum corresponding to one estimated IN bee.
- `OUT_BEE_EVENT_FLUX_UNIT`: Flux sum corresponding to one estimated OUT bee.
- `WINDOW_SEC`: Window size for summary aggregation. The default is 3 seconds.

## 8. Output files

Outputs are written to `bee_count_output/`.

- `entrance_count_preview.mp4`: ROI preview video with the entrance rectangle, four-edge counting boundary band, flow arrows, and a separated text panel for flux values, detector states, and cumulative event counts.
- `entrance_flux_frame.csv`: Frame-level flux time series with detector active flags.
- `entrance_events.csv`: Event-level rows with direction, start/end time, duration, flux sum, peak flux, and estimated count.
- `entrance_count_3sec.csv`: 3-second event-based summary CSV.

The 3-second summary is based on event midpoint time, not direct frame-level flux accumulation.

## 9. Validation and calibration

Manually count IN/OUT crossings for a short segment of about 30 seconds to 1 minute. Compare the manual count with event rows and their `event_flux_sum` values.

Tune `IN_BEE_EVENT_FLUX_UNIT` and `OUT_BEE_EVENT_FLUX_UNIT` so that the calibrated segment matches the manual count reasonably well. Then check another segment to confirm that the estimated counts remain proportional to observed activity.

The first preview points to inspect are:

- The cyan counting band appears on the top, bottom, left, and right edges.
- The text panel is separated from the ROI so that flux values do not cover the entrance view.
- Green arrows correspond to IN flux and blue arrows correspond to OUT flux.
- Slow or sustained motion near the same boundary is grouped into one event rather than repeatedly counted frame by frame.

## 10. Running the script

Run the default two-video wrapper from the project root:

Run:

```powershell
uv run python -m src.optical_count
```

For a quick syntax check without processing video:

```powershell
uv run python -m py_compile src/optical_count.py
```
