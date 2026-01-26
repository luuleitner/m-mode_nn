import numpy as np
import glob
import os
import yaml
from preprocessing.signal_utils import apply_joystick_filters

# Paths - relative to this file's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
config_path = os.path.join(script_dir, "config.yaml")

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths from config
paths_config = config.get('paths', {})
base_path = os.path.join(project_dir, paths_config.get('raw_data', 'Data/raw'))
output_base_path = os.path.join(project_dir, paths_config.get('processed_data', 'Data/processed'))

# File names from config
files_config = config.get('files', {})
joystick_file = files_config.get('joystick', '_joystick.npy')
labels_file = files_config.get('labels', '_labels.npy')

joystick_column = config.get('joystick_column', 1)
filters_config = config.get('filters', {})
labels_config = config.get('labels', {})
threshold_percent = labels_config.get('threshold_percent', 5.0)


def get_excluded_experiments(session_name, max_exp=20):
    """
    Returns a set of experiment numbers to exclude based on config.
    """
    if 'exclude' not in config or session_name not in config['exclude']:
        return set()

    session_config = config['exclude'][session_name]
    excluded = set()

    if 'pattern' in session_config:
        pattern = session_config['pattern']
        double_exclude = session_config.get('double_exclude', [])

        if pattern == 'odd':
            exp = 1
            while exp <= max_exp:
                excluded.add(exp)
                if exp in double_exclude:
                    excluded.add(exp + 1)
                    exp += 3
                else:
                    exp += 2
        elif pattern == 'even':
            exp = 0
            while exp <= max_exp:
                excluded.add(exp)
                if exp in double_exclude:
                    excluded.add(exp + 1)
                    exp += 3
                else:
                    exp += 2

    if 'additional' in session_config:
        excluded.update(session_config['additional'])

    return excluded



def create_derivative_labels(derivative, threshold_percent=5.0):
    """
    Create labels from derivative signal based on percentage of max value.
    Labels array: 0 = neutral (noise), Movement Intention: 1 = upward, 2 = downward
    """
    # Calculate threshold as percentage of max absolute value
    max_val = np.max(np.abs(derivative))
    threshold = max_val * (threshold_percent / 100.0)

    # Create labels
    labels = np.zeros(len(derivative), dtype=np.int8)
    labels[derivative > threshold] = 1   # Upward intention
    labels[derivative < -threshold] = 2  # Downward intention
    # Labels remain 0 for neutral/noise zone

    return labels, threshold


def create_edge_to_peak_labels(filtered_data, derivative, threshold_percent=5.0):
    """
    Create labels based on edge detection on raw position and peak finding on derivative.

    Algorithm:
    1. Detect rising edge on filtered position crossing upper threshold (start of upward movement)
    2. Detect falling edge on filtered position crossing lower threshold (start of downward movement)
    3. Find first derivative peak/trough after each edge (end of movement)
    4. Label region between edge and peak

    This approach avoids labeling spring-back movements by detecting actual position
    displacement rather than velocity spikes.

    Args:
        filtered_data: Filtered joystick position data
        derivative: Derivative of the filtered data
        threshold_percent: Percentage of max value for threshold detection

    Returns:
        labels: Array of labels (0=noise, 1=upward, 2=downward)
        threshold: Position threshold value used for edge detection
        markers: Dict with 'edges' and 'peaks' indices for visualization
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

        # Apply filters to raw data
        data = apply_joystick_filters(raw_data.copy(), filters_config, 'raw')

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
