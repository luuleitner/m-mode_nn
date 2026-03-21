"""
Segment-then-Classify Label Logic

Two-pass approach for labeling joystick movements:
  Pass 1 (detect_movements): position-based activity detection
    - Smooth |position| with moving average
    - Threshold to find active regions (joystick deflected from center)
    - Cleanup: remove short segments, merge close ones

  Pass 2 (label_movements): peak-displacement classification
    - For each segment, find peak |position|
    - Direction = sign of peak position

For visualization, use: preprocessing/visualization/visualize_labels.py
"""

import numpy as np
import os
import yaml


# Load label config for defaults
_script_dir = os.path.dirname(os.path.abspath(__file__))
_label_config_path = os.path.join(_script_dir, "label_config.yaml")

try:
    with open(_label_config_path, 'r') as f:
        _label_config = yaml.safe_load(f)
    _sc_config = _label_config.get('segment_classify', {})
except FileNotFoundError:
    _sc_config = {}

DEFAULT_ACTIVITY_THRESH = _sc_config.get('activity_threshold_percent', 5.0)
DEFAULT_SMOOTH_WINDOW = _sc_config.get('smooth_window', 15)
DEFAULT_MIN_DURATION = _sc_config.get('min_duration', 10)
DEFAULT_MERGE_GAP = _sc_config.get('merge_gap', 10)


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def smooth_activity(position, smooth_window):
    """Compute smoothed |position| as activity signal."""
    raw_activity = np.abs(position)
    kernel = np.ones(smooth_window) / smooth_window
    return np.convolve(raw_activity, kernel, mode='same')


def segment_activity(activity, threshold, min_duration, merge_gap):
    """
    Convert activity signal into clean movement segments.

    1. Binary mask: activity > threshold
    2. Extract contiguous active regions
    3. Merge regions closer than merge_gap
    4. Remove regions shorter than min_duration

    Returns:
        segments: list of (start, stop) tuples
        debug: dict with intermediate results
    """
    active = activity > threshold
    diff = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]

    if len(starts) == 0:
        return [], {'active_mask': active, 'raw_segments': [], 'merged_segments': []}

    raw_segments = list(zip(starts.tolist(), stops.tolist()))

    # Merge close segments
    merged_starts, merged_stops = [starts[0]], [stops[0]]
    for i in range(1, len(starts)):
        if starts[i] - merged_stops[-1] < merge_gap:
            merged_stops[-1] = stops[i]
        else:
            merged_starts.append(starts[i])
            merged_stops.append(stops[i])

    merged_segments = list(zip(merged_starts, merged_stops))

    # Remove short segments
    segments = [
        (s, e) for s, e in merged_segments
        if e - s >= min_duration
    ]

    debug = {
        'active_mask': active,
        'raw_segments': raw_segments,
        'merged_segments': merged_segments,
    }

    return segments, debug


def classify_segments(segments, position):
    """
    Classify each segment's direction from peak displacement.

    For each segment, finds the sample where |position| is maximum.
    Direction = 1 (positive) if peak > 0, else 2 (negative).

    Returns list of dicts: {start, stop, peak_idx, peak_val, direction}
    """
    classified = []
    for start, stop in segments:
        segment_pos = position[start:stop]
        peak_local = np.argmax(np.abs(segment_pos))
        peak_idx = start + peak_local
        peak_val = position[peak_idx]
        direction = 1 if peak_val > 0 else 2

        classified.append({
            'start': start,
            'stop': stop,
            'peak_idx': peak_idx,
            'peak_val': peak_val,
            'direction': direction,
        })

    return classified


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def detect_movements(position, config=None):
    """
    Pass 1: Find active movement regions from position signal.

    Uses smoothed |position| to detect when the joystick is deflected
    from center. No velocity zero-crossing problem.

    Args:
        position: filtered position signal [n]
        config: dict with segment_classify parameters (optional)

    Returns:
        segments: list of (start, stop) tuples
        params: dict with computed thresholds, activity signal, and debug info
    """
    if config is None:
        config = {}

    activity_thresh_pct = config.get('activity_threshold_percent', DEFAULT_ACTIVITY_THRESH)
    smooth_win = config.get('smooth_window', DEFAULT_SMOOTH_WINDOW)
    min_dur = config.get('min_duration', DEFAULT_MIN_DURATION)
    gap = config.get('merge_gap', DEFAULT_MERGE_GAP)

    activity = smooth_activity(position, smooth_win)
    threshold = activity_thresh_pct / 100.0 * np.max(activity)
    segments, debug = segment_activity(activity, threshold, min_dur, gap)

    params = {
        'activity_threshold': threshold,
        'activity_threshold_percent': activity_thresh_pct,
        'activity_max': np.max(activity),
        'smooth_window': smooth_win,
        'min_duration': min_dur,
        'merge_gap': gap,
        'activity': activity,
        'active_mask': debug['active_mask'],
        'raw_segments': debug['raw_segments'],
        'merged_segments': debug['merged_segments'],
    }
    return segments, params


def label_movements(position, velocity, config=None):
    """
    Single-axis labeling: detect movements then classify by peak displacement.

    Args:
        position: filtered position signal [n]
        velocity: filtered velocity signal [n] (kept for API compatibility with callers)
        config: dict with segment_classify parameters (optional)

    Returns:
        labels: array [n], 0=noise, 1=positive, 2=negative
        segments: list of segment dicts {start, stop, peak_idx, peak_val, direction}
        params: dict with thresholds and settings
    """
    raw_segments, params = detect_movements(position, config)
    classified = classify_segments(raw_segments, position)

    labels = np.zeros(len(position), dtype=np.int8)
    for seg in classified:
        labels[seg['start']:seg['stop']] = seg['direction']

    return labels, classified, params


def label_movements_xy(x_pos, y_pos, x_vel, y_vel, config=None):
    """
    Dual-axis 5-class labeling.

    Runs label_movements on each axis independently,
    then merges with per-sample amplitude voting (v²).

    Label mapping:
        0: Noise    (no movement on either axis)
        1: Up       (Y+ dominant)
        2: Down     (Y- dominant)
        3: Left     (X- dominant)
        4: Right    (X+ dominant)

    Args:
        x_pos, y_pos: filtered position signals [n]
        x_vel, y_vel: filtered velocity signals [n]
        config: dict with segment_classify parameters (optional)

    Returns:
        labels: array [n] with 5-class labels
        segments: {'x': [...], 'y': [...]}
        params: {'x': {...}, 'y': {...}}
    """
    x_labels, x_segs, x_params = label_movements(x_pos, x_vel, config)
    y_labels, y_segs, y_params = label_movements(y_pos, y_vel, config)

    n = len(x_pos)
    labels = np.zeros(n, dtype=np.int8)
    x_energy = x_vel ** 2
    y_energy = y_vel ** 2

    for i in range(n):
        x_lbl = x_labels[i]
        y_lbl = y_labels[i]

        if x_lbl == 0 and y_lbl == 0:
            labels[i] = 0
        elif x_lbl == 0:
            labels[i] = y_lbl                          # 1=up, 2=down
        elif y_lbl == 0:
            labels[i] = 4 if x_lbl == 1 else 3         # 4=right, 3=left
        else:
            # overlap: amplitude voting using energy (v²)
            if y_energy[i] >= x_energy[i]:
                labels[i] = y_lbl
            else:
                labels[i] = 4 if x_lbl == 1 else 3

    return labels, {'x': x_segs, 'y': y_segs}, {'x': x_params, 'y': y_params}
