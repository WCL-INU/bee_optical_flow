# Bee Entrance Persistence Filter

## 1. Purpose

The event detector approach has been rolled back. This version does not use crossing events, hysteresis, cooldown, object tracking, YOLO, or event CSV files.

The goal is simpler: reduce optical-flow noise first, then keep the existing 3-second window-based flux/count summary. In quiet videos, filtered flux should stay small and stable.

## 2. Gaussian Blur

Grayscale frames are optionally blurred before Farneback optical flow.

- `BLUR_KERNEL = 1`: no blur
- `BLUR_KERNEL = 3`: default `GaussianBlur((3, 3), 0)`
- `BLUR_KERNEL = 5`: stronger blur for noisy video

Kernels larger than 5 are intentionally excluded. A 7x7 or larger blur can remove small bee motion along with noise.

## 3. Temporal Persistence Filter

The raw candidate mask is built from the counting boundary band, flow magnitude threshold, and normal-flow threshold:

```python
candidate = (
    counting_boundary_band
    & (mag > FLOW_MAG_THRESHOLD)
    & (abs(normal_flow) > NORMAL_FLOW_THRESHOLD)
)
```

Instead of using `candidate` directly for filtered flux, the filter keeps a float32 persistence map:

```python
persistence = PERSIST_DECAY * persistence + candidate.astype(np.float32)
persistent_candidate = persistence > PERSIST_THRESHOLD
filtered_candidate = candidate & persistent_candidate
```

A one-frame candidate spike usually disappears. Motion that persists for multiple frames can pass through.

## 4. Persistence Parameters

- `PERSIST_DECAY`: Higher values keep history longer.
- `PERSIST_THRESHOLD`: Higher values require the candidate to persist longer before passing.

Defaults:

```python
PERSIST_DECAY = 0.65
PERSIST_THRESHOLD = 1.3
```

With these defaults, a pixel generally needs to remain active for more than one frame before it contributes to filtered flux.

## 5. Optional Component Area Filter

`USE_COMPONENT_AREA_FILTER` is off by default.

When enabled, the already persistence-filtered candidate mask is split with `connectedComponentsWithStats`, and only components with area greater than or equal to `MIN_FLOW_COMPONENT_AREA` are kept.

No coherence, mean magnitude, aspect-ratio, bounding-box, morphology, or event filters are used in this version. Area filtering is only a final optional cleanup step.

## 6. Bottom Boundary Exclusion

The entrance bottom edge touches the bottom of the ROI, so it is excluded from counting. Only the top, left, and right entrance boundaries are used.

The preview shows:

- full entrance rectangle as a thin red outline
- actual counting boundary band on top/left/right in cyan
- bottom edge as gray or visually excluded from the cyan band

## 7. Test Method

Compare:

- `videos/ANU-25-summer-6_20260405_060000.mp4`
- `videos/ANU-25-summer-6_20260405_070000.mp4`

Both are treated as noise-dominant or low-activity segments. Expected behavior:

- raw traffic flux should be reduced by filtered traffic flux
- both filtered traffic values should be similarly small
- if one video remains much larger than the other, filtering is not yet good enough

## 8. Tuning Order

Tune in this order:

1. Compare `--blur-kernel 3` and `--blur-kernel 5`.
2. Increase `--persist-threshold` if one-frame or short-lived noise remains.
3. Adjust `--persist-decay` to control how long candidate history remains.
4. Enable `--use-component-area-filter` only after persistence behavior is understood.

## 9. Warning

Do not tune only on quiet/noise videos. A setting that suppresses noise can also suppress real bee motion. After quiet-video results look stable, test a positive sample with clear bee crossings and verify that real flow still appears in the preview and contributes to filtered flux.

## 10. Continued Optional Checks

If persistence and blur are not enough, two simple optional checks can be tested:

- `--use-global-flow-compensation`: subtracts the median background ROI flow vector before boundary flux is computed. This targets camera/background drift and compression-wide motion. Risk: if the background estimate is contaminated by many bees, real motion can be reduced.
- `--use-bidirectional-balance-filter`: when frame-level IN and OUT filtered flux are very balanced, treats the balanced portion as vibration-like noise and keeps only the directional residual. Risk: true simultaneous IN and OUT activity can be under-counted.

Both are off by default because they are stronger assumptions than blur and persistence.

## 11. Commands

Single video:

```powershell
python -m src.bee_entrance_count --video videos/ANU-25-summer-6_20260405_060000.mp4
```

Default comparison:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

Blur 5 comparison:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --blur-kernel 5
```

Disable persistence:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --no-persistence-filter
```

Stronger persistence:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --persist-decay 0.75 --persist-threshold 1.8
```

Optional area filter:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --use-component-area-filter --min-flow-component-area 30
```

Optional global flow compensation:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --use-global-flow-compensation
```

Optional balanced bidirectional suppression:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --use-bidirectional-balance-filter
```
