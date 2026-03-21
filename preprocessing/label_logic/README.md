# Label Logic

*Created: 2026-03-21 | Updated: 2026-03-21*

## Overview

Classifies a 2D joystick signal into 5 movement directions (up/down/left/right/noise) using a two-pass **segment-then-classify** approach.

## Pipeline

```
  Pass 1: Activity Detection (per axis)
  ──────────────────────────────────────
  position ──► |position| ──► smooth ──► threshold ──► cleanup ──► segments
                                              │
                                     activity_threshold_percent
                                     (% of max |position|)

  Pass 2: Classification (per axis)
  ──────────────────────────────────
  segments + position ──► peak displacement per segment ──► direction (1 or 2)

  Pass 3: Dual-axis merge
  ────────────────────────
  X labels + Y labels ──► amplitude voting (v²) ──► 5-class labels
```

### Pass 1 — Activity detection (`detect_movements`)

Finds where the joystick is deflected from center using smoothed `|position|`:

1. Compute `|position|`, smooth with moving average
2. Threshold: `smoothed |position| > activity_threshold_percent% × max`
3. Extract contiguous active regions
4. Merge regions closer than `merge_gap` samples
5. Remove regions shorter than `min_duration` samples

Position-based detection avoids the velocity zero-crossing problem — velocity always crosses zero at the turning point, but position stays deflected throughout the entire push-and-return.

### Pass 2 — Classification (`label_movements`)

For each detected segment, determines direction from peak displacement:
- Find sample where `|position|` is maximum within the segment
- `direction = 1` (positive) if peak > 0, else `2` (negative)

### Pass 3 — Dual-axis merge (`label_movements_xy`)

Runs Pass 1+2 on each axis independently, then merges per-sample:

```
  x_lbl    y_lbl    Resolution                 Output
  ─────    ─────    ────────────────────────    ──────
  0        0        Both noise                  0 (noise)
  0        1/2      Only Y active               1 (up) or 2 (down)
  1/2      0        Only X active, remap        4 (right) or 3 (left)
  1/2      1/2      OVERLAP → compare v²        dominant axis wins
```

### Label mapping

| Label | Direction |
|-------|-----------|
| 0     | Noise     |
| 1     | Up (Y+)   |
| 2     | Down (Y−) |
| 3     | Left (X−) |
| 4     | Right (X+)|

## Configuration

Parameters in `label_config.yaml` under `segment_classify`:

| Parameter                    | Description                                      |
|------------------------------|--------------------------------------------------|
| `activity_threshold_percent` | % of max `\|position\|` to count as "active"     |
| `smooth_window`              | Moving average window for `\|position\|` (samples)|
| `min_duration`               | Discard segments shorter than this (samples)      |
| `merge_gap`                  | Merge segments with gap smaller than this (samples)|

## Visualization

- `preprocessing/visualization/visualize_labels.py` — 2-row plot (X/Y) with label shading and segment markers
- `preprocessing/visualization/visualize_labels_debug.py` — 6-row debug plot showing position, velocity, and activity signal with raw/merged/final segments
- `preprocessing/visualization/visualize_labels_raster.py` — matplotlib grid of label heatmaps across multiple experiments

## Files

```
  label_logic/
  ├── label_logic.py          ← current implementation (segment-classify)
  ├── label_config.yaml       ← current config
  ├── label_logic_old.py      ← previous implementation (velocity state machine)
  ├── label_config_old.yaml   ← previous config
  └── README.md
```
