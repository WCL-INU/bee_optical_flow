# Bee Entrance Noise Filtering

## 1. Why the event detector is rolled back

The event detector grouped flux bursts into IN/OUT crossing events, but it made the pipeline harder to tune before the flux signal itself was stable. When background, compression, illumination, or camera noise produces dense optical-flow artifacts, event logic can still convert those artifacts into counts.

For now, counting is rolled back to simple window-based flux accumulation. The current priority is to make the per-frame flux signal cleaner.

## 2. Why flux noise removal comes first

Optical flow reacts to real bee movement, but it can also react to codec blocks, brightness shifts, small background texture changes, and sensor noise. If these weak flow pixels are allowed into the entrance boundary flux, a quiet video can look active.

The first goal is therefore not better counting logic. The first goal is to remove small isolated flow components and keep only compact, dense flow regions that are more likely to be caused by bees.

## 3. Raw flux versus component-filtered flux

Baseline raw flux uses only:

- the counting boundary band
- `FLOW_MAG_THRESHOLD`
- `NORMAL_FLOW_THRESHOLD`

Component-filtered flux starts from the same raw candidate mask, then runs `connectedComponentsWithStats`. Components smaller than `MIN_FLOW_COMPONENT_AREA` are removed before IN/OUT flux is summed.

The output CSV keeps both raw and filtered flux so each filter step can be evaluated directly.

## 4. Why start with area filtering

Area filtering is the simplest and least interpretive filter. It removes tiny speckles and isolated codec or illumination artifacts without assuming much about bee direction, shape, or velocity.

The default filter stack therefore enables only:

```python
MIN_FLOW_COMPONENT_AREA = 30
```

More aggressive filters should be added only after checking whether area filtering is insufficient.

## 5. Optional filters and risks

### Mean magnitude filter

`USE_MEAN_MAG_FILTER` removes components whose average optical-flow magnitude is below `MIN_COMPONENT_MEAN_MAG`.

Risk: a real slow-moving bee can be removed.

### Normal-flow strength filter

`USE_MEAN_NORMAL_FLOW_FILTER` removes components whose average absolute boundary-normal flow is below `MIN_COMPONENT_MEAN_NORMAL_FLOW`.

Risk: a bee moving diagonally or partly parallel to the boundary can be removed.

### Coherence filter

`USE_COHERENCE_FILTER` measures flow direction consistency inside each component. A component with mixed directions has lower coherence.

Risk: real bee motion can be locally messy because legs, wings, shadows, and body texture move differently.

### Morphology

`USE_MORPHOLOGY` can apply optional open/close operations to the candidate mask.

Risk: opening can erase real small bee flow; closing can merge unrelated noise into a larger accepted component. It is off by default.

## 6. Two-video comparison criteria

The first comparison targets:

- `videos/ANU-25-summer-6_20260405_060000.mp4`
- `videos/ANU-25-summer-6_20260405_070000.mp4`

These are treated as low-activity or noise-dominant segments. Expected behavior:

- both videos should have small filtered traffic flux
- filtered traffic flux should be lower than raw traffic flux
- one video should not be abnormally larger than the other

Temporary warning criteria:

- `ratio_between_videos > 2.0` prints `WARNING`
- `max_window_filtered_traffic_count_est > 3.0` prints `WARNING`
- otherwise the comparison prints `PASS`

These thresholds are configurable in code and CLI.

## 7. Expected result for the two test videos

The expected result is not necessarily zero. Small residual flux is acceptable. The important condition is that both videos produce similarly small filtered traffic values.

If one of the two videos has much larger filtered traffic than the other, the filter is probably still accepting background or compression noise.

## 8. Tuning order

Tune in this order:

1. Increase or decrease `MIN_FLOW_COMPONENT_AREA`.
2. Adjust `FLOW_MAG_THRESHOLD` and `NORMAL_FLOW_THRESHOLD`.
3. Enable `USE_MEAN_MAG_FILTER` only if low-magnitude components remain.
4. Enable `USE_MEAN_NORMAL_FLOW_FILTER` only if weak normal-flow components remain.
5. Enable `USE_COHERENCE_FILTER` only after the simpler filters are understood.
6. Enable morphology only if candidate regions are fragmented in a way that area filtering cannot handle.

## 9. Positive sample warning

Noise videos are not enough for final tuning. A filter that works on quiet segments can be too aggressive on videos with many real bees.

After the two quiet videos pass, check a high-activity positive sample and confirm that real bee flow is still visible in the preview and still contributes to filtered flux.

## 10. Commands

Run one video:

```powershell
python bee_entrance_count.py --video videos/ANU-25-summer-6_20260405_060000.mp4
```

Compare the two noise-focused test videos:

```powershell
python bee_entrance_count.py --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

Try a stricter area filter:

```powershell
python bee_entrance_count.py --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --min-flow-component-area 50
```

Enable optional mean magnitude filtering:

```powershell
python bee_entrance_count.py --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4 --use-mean-mag-filter
```
