import numpy as np
import glob
import os
import yaml
from preprocessing.signal_utils import apply_joystick_filters

# Paths - relative to this file's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
label_config_path = os.path.join(script_dir, "label_config.yaml")
main_config_path = os.path.join(project_dir, "config", "config.yaml")

# Load label config
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)

# Load main config for paths
with open(main_config_path, 'r') as f:
    main_config = yaml.safe_load(f)

# Paths from main config
base_data_path = main_config.get('global_setting', {}).get('paths', {}).get('base_data_path', '')
base_path = os.path.join(base_data_path, 'raw')
output_base_path = os.path.join(base_data_path, 'processed')

# File names
joystick_file = '_joystick.npy'
labels_file = '_labels.npy'

# Label settings from label_config
label_axis = label_config.get('axis', 'x')
joystick_column = 1 if label_axis == 'x' else 2
filters_config = label_config.get('filters', {})
threshold_percent = label_config.get('threshold_percent', 5.0)
label_method = label_config.get('method', 'derivative')


def get_excluded_experiments(session_name, max_exp=20):
    """
    Returns a set of experiment numbers to exclude.

    Note: Exclusion logic has been moved to main config's selection_file strategy.
    This function now returns empty set - use config/preprocessing_selection.csv instead.
    """
    return set()



def create_derivative_labels(derivative, threshold_percent=5.0):
    """
    Create labels from derivative signal based on percentage of max value.
    Labels array: 0 = neutral (noise), 1 = positive direction, 2 = negative direction

    Note: Direction meaning depends on axis used:
        - X-axis: 1 = right, 2 = left
        - Y-axis: 1 = up, 2 = down
    """
    # Calculate threshold as percentage of max absolute value
    max_val = np.max(np.abs(derivative))
    threshold = max_val * (threshold_percent / 100.0)

    # Create labels
    labels = np.zeros(len(derivative), dtype=np.int8)
    labels[derivative > threshold] = 1   # Positive direction
    labels[derivative < -threshold] = 2  # Negative direction
    # Labels remain 0 for neutral/noise zone

    return labels, threshold


def create_edge_to_peak_labels(filtered_data, derivative, threshold_percent=5.0):
    """
    Create labels based on edge detection on raw position and peak finding on derivative.

    Algorithm:
    1. Detect rising edge on filtered position crossing upper threshold (start of positive movement)
    2. Detect falling edge on filtered position crossing lower threshold (start of negative movement)
    3. Find first derivative peak/trough after each edge (end of movement)
    4. Label region between edge and peak

    This approach avoids labeling spring-back movements by detecting actual position
    displacement rather than velocity spikes.

    Args:
        filtered_data: Filtered joystick position data
        derivative: Derivative of the filtered data
        threshold_percent: Percentage of max value for threshold detection

    Returns:
        labels: Array of labels (0=noise, 1=positive, 2=negative)
        threshold: Position threshold value used for edge detection
        markers: Dict with 'edges' and 'peaks' indices for visualization

    Note: Direction meaning depends on axis used:
        - X-axis: 1 = right, 2 = left
        - Y-axis: 1 = up, 2 = down
    """
    n = len(derivative)
    labels = np.zeros(n, dtype=np.int8)

    # Calculate position threshold for edge detection
    pos_max = np.max(np.abs(filtered_data))
    threshold = pos_max * (threshold_percent / 100.0)

    # Store markers for visualization
    edge_up_indices = []
    edge_down_indices = []
    peak_up_indices = []
    peak_down_indices = []

    # Find rising edges on POSITION data (crossing upper threshold from below)
    above_upper = filtered_data > threshold
    rising_edges = np.where(np.diff(above_upper.astype(int)) == 1)[0] + 1

    # Find falling edges on POSITION data (crossing lower threshold from above)
    below_lower = filtered_data < -threshold
    falling_edges = np.where(np.diff(below_lower.astype(int)) == 1)[0] + 1

    # Process rising edges (upward movement)
    for edge_idx in rising_edges:
        edge_up_indices.append(edge_idx)

        # Find first local maximum on DERIVATIVE after edge
        peak_idx = _find_next_peak(derivative, edge_idx, direction='max')
        if peak_idx is not None:
            peak_up_indices.append(peak_idx)
            # Label region between edge and peak as upward (1)
            labels[edge_idx:peak_idx + 1] = 1

    # Process falling edges (downward movement)
    for edge_idx in falling_edges:
        edge_down_indices.append(edge_idx)

        # Find first local minimum on DERIVATIVE after edge
        peak_idx = _find_next_peak(derivative, edge_idx, direction='min')
        if peak_idx is not None:
            peak_down_indices.append(peak_idx)
            # Label region between edge and peak as downward (2)
            labels[edge_idx:peak_idx + 1] = 2

    markers = {
        'edge_up': np.array(edge_up_indices),
        'edge_down': np.array(edge_down_indices),
        'peak_up': np.array(peak_up_indices),
        'peak_down': np.array(peak_down_indices),
    }

    return labels, threshold, markers


def create_edge_to_derivative_labels(filtered_data, derivative, threshold_percent=5.0):
    """
    Create labels based on edge detection on position and derivative threshold crossing.

    Algorithm:
    1. Detect rising edge on filtered position crossing upper threshold (start of positive movement)
    2. Detect falling edge on filtered position crossing lower threshold (start of negative movement)
    3. Find when derivative crosses back through derivative threshold (end of movement)
    4. Label region between edge and derivative threshold crossing

    This hybrid approach uses position displacement to detect movement start (avoiding
    spring-back false positives) and derivative threshold to detect movement end.

    Args:
        filtered_data: Filtered joystick position data
        derivative: Derivative of the filtered data
        threshold_percent: Percentage of max value for threshold detection (applied to both signals)

    Returns:
        labels: Array of labels (0=noise, 1=positive, 2=negative)
        threshold: Position threshold value used for edge detection
        markers: Dict with 'edge_up', 'edge_down', 'deriv_cross_up', 'deriv_cross_down' indices

    Note: Direction meaning depends on axis used:
        - X-axis: 1 = right, 2 = left
        - Y-axis: 1 = up, 2 = down
    """
    n = len(derivative)
    labels = np.zeros(n, dtype=np.int8)

    # Calculate position threshold for edge detection
    pos_max = np.max(np.abs(filtered_data))
    pos_threshold = pos_max * (threshold_percent / 100.0)

    # Calculate derivative threshold for end detection
    deriv_max = np.max(np.abs(derivative))
    deriv_threshold = deriv_max * (threshold_percent / 100.0)

    # Store markers for visualization
    edge_up_indices = []
    edge_down_indices = []
    deriv_cross_up_indices = []
    deriv_cross_down_indices = []

    # Find rising edges on POSITION data (crossing upper threshold from below)
    above_upper = filtered_data > pos_threshold
    rising_edges = np.where(np.diff(above_upper.astype(int)) == 1)[0] + 1

    # Find falling edges on POSITION data (crossing lower threshold from above)
    below_lower = filtered_data < -pos_threshold
    falling_edges = np.where(np.diff(below_lower.astype(int)) == 1)[0] + 1

    # Process rising edges (upward movement)
    for edge_idx in rising_edges:
        edge_up_indices.append(edge_idx)

        # Find when derivative drops back below threshold (movement ending)
        cross_idx = _find_derivative_threshold_crossing(
            derivative, edge_idx, deriv_threshold, direction='down'
        )
        if cross_idx is not None:
            deriv_cross_up_indices.append(cross_idx)
            # Label region between edge and derivative crossing as upward (1)
            labels[edge_idx:cross_idx + 1] = 1

    # Process falling edges (downward movement)
    for edge_idx in falling_edges:
        edge_down_indices.append(edge_idx)

        # Find when derivative rises back above -threshold (movement ending)
        cross_idx = _find_derivative_threshold_crossing(
            derivative, edge_idx, deriv_threshold, direction='up'
        )
        if cross_idx is not None:
            deriv_cross_down_indices.append(cross_idx)
            # Label region between edge and derivative crossing as downward (2)
            labels[edge_idx:cross_idx + 1] = 2

    markers = {
        'edge_up': np.array(edge_up_indices),
        'edge_down': np.array(edge_down_indices),
        'deriv_cross_up': np.array(deriv_cross_up_indices),
        'deriv_cross_down': np.array(deriv_cross_down_indices),
    }

    return labels, pos_threshold, deriv_threshold, markers


def create_5class_labels_dual_axis(x_filtered, y_filtered, x_derivative, y_derivative, threshold_percent=5.0):
    """
    Create 5-class labels using edge_to_derivative method on both axes.

    Pipeline:
        1. Apply edge_to_derivative to X-axis → labels_x (0=noise, 1=right, 2=left)
        2. Apply edge_to_derivative to Y-axis → labels_y (0=noise, 1=up, 2=down)
        3. Merge labels:
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

    Args:
        x_filtered: Filtered X-axis position data
        y_filtered: Filtered Y-axis position data
        x_derivative: Derivative of filtered X-axis data
        y_derivative: Derivative of filtered Y-axis data
        threshold_percent: Percentage of max value for threshold detection

    Returns:
        labels: Array of 5-class labels
        thresholds: Dict with 'x' and 'y' threshold values
        markers: Dict with markers from both axes for visualization
    """
    n = len(x_derivative)
    labels = np.zeros(n, dtype=np.int8)

    # Step 1: Apply edge_to_derivative to both axes independently
    x_labels, x_pos_thresh, x_deriv_thresh, x_markers = create_edge_to_derivative_labels(
        x_filtered, x_derivative, threshold_percent
    )
    y_labels, y_pos_thresh, y_deriv_thresh, y_markers = create_edge_to_derivative_labels(
        y_filtered, y_derivative, threshold_percent
    )

    # Compute per-sample amplitudes for overlap resolution
    x_amp = np.abs(x_derivative)
    y_amp = np.abs(y_derivative)

    # Step 2: Merge labels with amplitude voting at overlaps
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
            # Only X has label: remap 1=right→4, 2=left→3
            labels[i] = 4 if x_lbl == 1 else 3
        else:
            # OVERLAP: both axes have labels → amplitude voting
            if y_amp[i] >= x_amp[i]:
                # Y dominant: 1=up, 2=down
                labels[i] = y_lbl
            else:
                # X dominant: remap 1=right→4, 2=left→3
                labels[i] = 4 if x_lbl == 1 else 3

    thresholds = {
        'x_pos': x_pos_thresh, 'x_deriv': x_deriv_thresh,
        'y_pos': y_pos_thresh, 'y_deriv': y_deriv_thresh
    }
    markers = {'x': x_markers, 'y': y_markers}

    return labels, thresholds, markers


def _find_derivative_threshold_crossing(derivative, start_idx, threshold, direction='down'):
    """
    Find when derivative crosses back through threshold after start_idx.

    Args:
        derivative: The derivative signal array
        start_idx: Index to start searching from
        threshold: Absolute threshold value
        direction: 'down' = find where derivative drops below +threshold (for upward movements)
                   'up' = find where derivative rises above -threshold (for downward movements)

    Returns:
        Index of the crossing, or None if not found
    """
    n = len(derivative)

    if start_idx >= n - 1:
        return None

    if direction == 'down':
        # For upward movement: derivative starts high (positive), find where it drops below threshold
        for i in range(start_idx + 1, n):
            if derivative[i] < threshold:
                return i
    else:  # direction == 'up'
        # For downward movement: derivative starts low (negative), find where it rises above -threshold
        for i in range(start_idx + 1, n):
            if derivative[i] > -threshold:
                return i

    return None


def _find_next_peak(signal, start_idx, direction='max', max_search=None):
    """
    Find the next local peak (max or min) after start_idx.

    Args:
        signal: The signal array
        start_idx: Index to start searching from
        direction: 'max' for local maximum, 'min' for local minimum
        max_search: Maximum number of samples to search (None = search to end)

    Returns:
        Index of the peak, or None if not found
    """
    n = len(signal)
    if max_search is None:
        end_idx = n
    else:
        end_idx = min(start_idx + max_search, n)

    if start_idx >= n - 1:
        return None

    # Search for peak
    for i in range(start_idx + 1, end_idx - 1):
        if direction == 'max':
            # Local maximum: higher than both neighbors
            if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
                # Also check it's actually decreasing after (not plateau)
                if signal[i] > signal[i + 1]:
                    return i
        else:  # direction == 'min'
            # Local minimum: lower than both neighbors
            if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
                if signal[i] < signal[i + 1]:
                    return i

    return None




def process_session(session_name):
    """Process all experiments in a session and create labels."""
    session_path = os.path.join(base_path, session_name)
    excluded = get_excluded_experiments(session_name)

    files = glob.glob(os.path.join(session_path, "*", joystick_file))
    files = sorted(files, key=lambda f: int(os.path.basename(os.path.dirname(f))))

    # Filter out excluded experiments
    files = [f for f in files if int(os.path.basename(os.path.dirname(f))) not in excluded]

    for file in files:
        # Load joystick data
        joystick_data = np.load(file, allow_pickle=True)
        raw_data = joystick_data[:, joystick_column]

        # Apply filters to position data
        data = apply_joystick_filters(raw_data.copy(), filters_config, 'position')

        # Compute derivative
        derivative = np.gradient(data)

        # Apply filters to derivative
        derivative = apply_joystick_filters(derivative, filters_config, 'derivative')

        # Create labels from derivative
        labels, _ = create_derivative_labels(derivative, threshold_percent)

        # Create output path maintaining folder structure
        relative_path = os.path.relpath(file, base_path)
        output_dir = os.path.join(output_base_path, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        # Save labels using config filename
        save_path = os.path.join(output_dir, labels_file)
        np.save(save_path, labels)
        print(f"Saved labels to: {save_path} (shape: {labels.shape})")


if __name__ == "__main__":
    # Get all sessions
    sessions = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    for session in sessions:
        print(f"\nProcessing session: {session}")
        process_session(session)

    print("\nDone!")
