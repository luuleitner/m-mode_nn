"""
Position Peak Label Logic

Labels movement periods using a three-phase state machine:
  Phase 1 (IDLE → RISING): |velocity| > threshold AND |position| < center threshold
  Phase 2 (RISING → PEAKED): position reverses (starts returning to center)
  Phase 3 (PEAKED → IDLE): |velocity| < threshold

This explicitly tracks the position peak (turning point) for robust detection.

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
    _position_peak_config = _label_config.get('position_peak', {})
except FileNotFoundError:
    _position_peak_config = {}

# Default thresholds from config
DEFAULT_DERIV_THRESH = _position_peak_config.get('deriv_threshold_percent', 10.0)
DEFAULT_POS_THRESH = _position_peak_config.get('pos_threshold_percent', 5.0)
DEFAULT_PEAK_WINDOW = _position_peak_config.get('peak_window', 3)
DEFAULT_TIMEOUT = _position_peak_config.get('timeout_samples', 500)

# State constants
STATE_IDLE = 0
STATE_RISING = 1
STATE_PEAKED = 2


def create_position_peak_labels(
    position,
    velocity,
    deriv_threshold_percent=None,
    pos_threshold_percent=None,
    peak_window=None,
    timeout_samples=None
):
    """
    Label movements using three-phase position peak detection.

    Phase 1 (IDLE → RISING): |velocity| > threshold AND |position| < center threshold
    Phase 2 (RISING → PEAKED): position reverses direction
    Phase 3 (PEAKED → IDLE): |velocity| < threshold

    Args:
        position: Filtered joystick position signal [n]
        velocity: Filtered derivative of position [n]
        deriv_threshold_percent: Threshold for |velocity| as % of range
        pos_threshold_percent: Position must be within ±this% of range from center
        peak_window: Number of samples to confirm position reversal (noise filter)
        timeout_samples: Max samples in any phase before forcing transition

    Returns:
        labels: [n] array, 0=noise, 1=positive movement, 2=negative movement
        thresholds: dict with threshold values
        markers: dict with start/peak/stop/rejected markers
    """
    # Use config defaults if not specified
    if deriv_threshold_percent is None:
        deriv_threshold_percent = DEFAULT_DERIV_THRESH
    if pos_threshold_percent is None:
        pos_threshold_percent = DEFAULT_POS_THRESH
    if peak_window is None:
        peak_window = DEFAULT_PEAK_WINDOW
    if timeout_samples is None:
        timeout_samples = DEFAULT_TIMEOUT

    n = len(position)
    labels = np.zeros(n, dtype=np.int64)

    # Compute thresholds
    vel_range = np.max(np.abs(velocity))
    pos_range = max(abs(position.max()), abs(position.min()))

    deriv_threshold = deriv_threshold_percent / 100.0 * vel_range
    pos_threshold = pos_threshold_percent / 100.0 * pos_range

    # Track markers
    start_markers = []
    peak_markers = []
    stop_markers = []
    rejected_markers = []
    timeout_markers = []

    # State tracking
    state = STATE_IDLE
    start_idx = 0
    direction = 0
    peak_idx = 0
    max_position = 0  # Track extreme position for peak detection
    samples_in_state = 0

    i = 0
    while i < n:
        vel = velocity[i]
        pos = position[i]
        abs_vel = abs(vel)
        abs_pos = abs(pos)
        samples_in_state += 1

        if state == STATE_IDLE:
            # Phase 1: Check for movement start
            if abs_vel > deriv_threshold:
                # Validate: position must be near center
                if abs_pos > pos_threshold:
                    rejected_markers.append(i)
                    i += 1
                    continue

                # Valid start - transition to RISING
                state = STATE_RISING
                start_idx = i
                direction = 1 if vel > 0 else 2  # 1=positive, 2=negative
                max_position = pos
                samples_in_state = 0
                start_markers.append(i)

        elif state == STATE_RISING:
            # Phase 2: Monitor position for peak (reversal)

            # Update extreme position
            if direction == 1:  # positive movement (going up/right)
                if pos > max_position:
                    max_position = pos
                    peak_idx = i
            else:  # negative movement (going down/left)
                if pos < max_position:
                    max_position = pos
                    peak_idx = i

            # Check for position reversal (peak detected)
            # Use window to filter noise: position must be reversing for peak_window samples
            reversal_detected = False
            if i >= peak_window:
                if direction == 1:
                    # Positive: peak when position consistently decreasing
                    reversal_detected = all(
                        position[i - j] < position[i - j - 1]
                        for j in range(peak_window)
                    )
                else:
                    # Negative: peak when position consistently increasing
                    reversal_detected = all(
                        position[i - j] > position[i - j - 1]
                        for j in range(peak_window)
                    )

            if reversal_detected:
                # Transition to PEAKED
                state = STATE_PEAKED
                peak_markers.append(peak_idx)
                samples_in_state = 0

            elif samples_in_state >= timeout_samples:
                # Timeout in RISING phase - force end
                labels[start_idx:i] = direction
                stop_markers.append(i)
                timeout_markers.append(i)
                state = STATE_IDLE
                samples_in_state = 0

        elif state == STATE_PEAKED:
            # Phase 3: Wait for velocity to settle below threshold
            if abs_vel < deriv_threshold:
                # Movement complete - apply labels
                labels[start_idx:i] = direction
                stop_markers.append(i)
                state = STATE_IDLE
                samples_in_state = 0

            elif samples_in_state >= timeout_samples:
                # Timeout in PEAKED phase - force end
                labels[start_idx:i] = direction
                stop_markers.append(i)
                timeout_markers.append(i)
                state = STATE_IDLE
                samples_in_state = 0

        i += 1

    # Handle case where movement extends to end of signal
    if state != STATE_IDLE:
        labels[start_idx:n] = direction
        stop_markers.append(n - 1)
        timeout_markers.append(n - 1)

    thresholds = {
        'deriv': deriv_threshold,
        'pos': pos_threshold,
        'deriv_percent': deriv_threshold_percent,
        'pos_percent': pos_threshold_percent,
        'peak_window': peak_window,
        'timeout': timeout_samples
    }

    markers = {
        'start': np.array(start_markers, dtype=int),
        'peak': np.array(peak_markers, dtype=int),
        'stop': np.array(stop_markers, dtype=int),
        'rejected': np.array(rejected_markers, dtype=int),
        'timeout': np.array(timeout_markers, dtype=int)
    }

    return labels, thresholds, markers


def create_5class_position_peak_labels(
    x_position, y_position, x_velocity, y_velocity,
    deriv_threshold_percent=None, pos_threshold_percent=None,
    peak_window=None, timeout_samples=None
):
    """
    Create 5-class labels using position_peak method on both axes.

    Pipeline:
        1. Apply position_peak to X-axis → labels_x (0=noise, 1=positive, 2=negative)
        2. Apply position_peak to Y-axis → labels_y (0=noise, 1=positive, 2=negative)
        3. Merge labels with amplitude voting:
           - Both noise → 0 (noise)
           - Only X has label → remap to 3=left, 4=right
           - Only Y has label → use as 1=up, 2=down
           - Both have labels (overlap) → compare amplitudes, pick dominant

    Label mapping:
        0: Noise    (no significant movement on either axis)
        1: Up       (Y+ dominant)
        2: Down     (Y- dominant)
        3: Left     (X- dominant)
        4: Right    (X+ dominant)

    Returns:
        labels: Array of 5-class labels
        thresholds: Dict with threshold values for both axes
        markers: Dict with markers from both axes
    """
    n = len(x_position)
    labels = np.zeros(n, dtype=np.int8)

    # Apply position_peak to both axes independently
    x_labels, x_thresh, x_markers = create_position_peak_labels(
        x_position, x_velocity,
        deriv_threshold_percent, pos_threshold_percent,
        peak_window, timeout_samples
    )
    y_labels, y_thresh, y_markers = create_position_peak_labels(
        y_position, y_velocity,
        deriv_threshold_percent, pos_threshold_percent,
        peak_window, timeout_samples
    )

    # Compute per-sample amplitudes for overlap resolution
    x_amp = np.abs(x_velocity)
    y_amp = np.abs(y_velocity)

    # Merge labels with amplitude voting at overlaps
    for i in range(n):
        x_lbl = x_labels[i]
        y_lbl = y_labels[i]

        if x_lbl == 0 and y_lbl == 0:
            # Both noise
            labels[i] = 0
        elif x_lbl == 0:
            # Only Y has label: 1=up, 2=down
            labels[i] = y_lbl
        elif y_lbl == 0:
            # Only X has label: remap 1=positive→4 (right), 2=negative→3 (left)
            labels[i] = 4 if x_lbl == 1 else 3
        else:
            # OVERLAP: both axes have labels → amplitude voting
            if y_amp[i] >= x_amp[i]:
                # Y dominant: 1=up, 2=down
                labels[i] = y_lbl
            else:
                # X dominant: remap 1=positive→4 (right), 2=negative→3 (left)
                labels[i] = 4 if x_lbl == 1 else 3

    thresholds = {
        'x_deriv': x_thresh['deriv'], 'x_pos': x_thresh['pos'],
        'y_deriv': y_thresh['deriv'], 'y_pos': y_thresh['pos'],
        'deriv_percent': x_thresh['deriv_percent'],
        'pos_percent': x_thresh['pos_percent'],
        'peak_window': x_thresh['peak_window'],
        'timeout': x_thresh['timeout']
    }
    markers = {'x': x_markers, 'y': y_markers}

    return labels, thresholds, markers
