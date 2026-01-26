import numpy as np
import glob
import os
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocessing.signal_utils import apply_joystick_filters
from preprocessing.label_logic.labeling import create_derivative_labels, create_edge_to_peak_labels

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

# File names from config
files_config = config.get('files', {})
joystick_file = files_config.get('joystick', '_joystick.npy')

joystick_column = config.get('joystick_column', 1)

# Display settings
display_config = config.get('display', {})
show_raw_trace = display_config.get('show_raw_trace', False)

# Filter settings
filters_config = config.get('filters', {})

# Labels settings
labels_config = config.get('labels', {})
label_method = labels_config.get('method', 'derivative')  # "derivative" or "edge_to_peak"
threshold_percent = labels_config.get('threshold_percent', 5.0)

# Include settings (sessions to process with their exclusion rules)
include_config = config.get('include', {})


def get_sessions():
    """Get sessions from include config."""
    if not include_config:
        # If no include config, return all sessions
        return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    # Return only sessions listed in include config
    return list(include_config.keys())


sessions = get_sessions()


def find_label_regions(labels, label_values=None):
    """
    Find contiguous regions for each label value.

    Args:
        labels: 1D array of label values
        label_values: List of label values to find regions for.
                      If None, uses [1, 2] (upward, downward).

    Returns:
        dict: {label_value: [(start, end), ...]} mapping each label to its regions
    """
    if label_values is None:
        label_values = [1, 2]

    regions = {}
    for label_val in label_values:
        mask = labels == label_val
        # Find transitions: 0->1 (start) and 1->0 (end)
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        regions[label_val] = list(zip(starts, ends))

    return regions


def get_excluded_experiments(session_name, max_exp=20):
    """
    Returns a set of experiment numbers to exclude based on config.
    """
    if session_name not in include_config:
        return set()

    session_config = include_config[session_name].get('exclude', {})
    if not session_config:
        return set()

    excluded = set()

    if 'pattern' in session_config:
        pattern = session_config['pattern']
        double_exclude = session_config.get('double_exclude', [])

        if pattern == 'odd':
            # Build exclusion list accounting for double excludes
            exp = 1  # Start at first odd
            while exp <= max_exp:
                excluded.add(exp)
                if exp in double_exclude:
                    # Also exclude the next one and shift pattern
                    excluded.add(exp + 1)
                    exp += 3  # Skip to next odd after double exclusion
                else:
                    exp += 2  # Normal odd pattern
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


def plot_joystick_stacked(session_path, session_name):
    """
    Creates a stacked plot of joystick_data[:,1] for all experiments in a session.
    Derivatives are plotted on a secondary y-axis.
    Uses Plotly for interactive visualization.
    """
    excluded = get_excluded_experiments(session_name)

    files = glob.glob(os.path.join(session_path, "*", joystick_file))
    files = sorted(files, key=lambda f: int(os.path.basename(os.path.dirname(f))))

    # Filter out excluded experiments
    files = [f for f in files if int(os.path.basename(os.path.dirname(f))) not in excluded]

    num_files = len(files)
    if num_files == 0:
        return

    col_name = "X" if joystick_column == 1 else "Y"

    # Create subplots with secondary y-axis for each row
    fig = make_subplots(
        rows=num_files, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=[[{"secondary_y": True}] for _ in range(num_files)],
        subplot_titles=[f"Exp {os.path.basename(os.path.dirname(f))}" for f in files]
    )

    for idx, file in enumerate(files):
        row = idx + 1
        joystick_data = np.load(file, allow_pickle=True)
        raw_data = joystick_data[:, joystick_column]
        sync_signal = joystick_data[:, 3]

        # Apply filters to data
        data = apply_joystick_filters(raw_data.copy(), filters_config, 'raw')

        derivative = np.gradient(data)

        # Apply filters to derivative
        derivative = apply_joystick_filters(derivative, filters_config, 'derivative')

        exp_name = os.path.basename(os.path.dirname(file))
        x_vals = np.arange(len(data))

        # Compute labels using configured method
        markers = None
        threshold = None
        if label_method == 'edge_to_peak':
            labels, threshold, markers = create_edge_to_peak_labels(data, derivative, threshold_percent)
        else:  # default: 'derivative'
            labels, threshold = create_derivative_labels(derivative, threshold_percent)

        # Shade regions based on labels (add shapes)
        label_regions = find_label_regions(labels)
        label_colors = {1: 'rgba(144, 238, 144, 0.3)', 2: 'rgba(240, 128, 128, 0.3)'}  # lightgreen, lightcoral with alpha
        for label_val, regions in label_regions.items():
            color = label_colors.get(label_val, 'rgba(128, 128, 128, 0.3)')
            for start, end in regions:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )

        # Plot raw trace in background if enabled
        if show_raw_trace:
            fig.add_trace(
                go.Scatter(x=x_vals, y=raw_data, mode='lines',
                          line=dict(color='green', dash='dot', width=1),
                          opacity=0.7, name='Raw', legendgroup='raw',
                          showlegend=(idx == 0)),
                row=row, col=1, secondary_y=False
            )

        # Plot filtered data on primary y-axis
        fig.add_trace(
            go.Scatter(x=x_vals, y=data, mode='lines',
                      line=dict(color='blue', width=1),
                      name='Filtered', legendgroup='filtered',
                      showlegend=(idx == 0)),
            row=row, col=1, secondary_y=False
        )

        # Plot derivative on secondary y-axis
        fig.add_trace(
            go.Scatter(x=x_vals, y=derivative, mode='lines',
                      line=dict(color='red', width=1),
                      opacity=0.7, name='Derivative', legendgroup='derivative',
                      showlegend=(idx == 0)),
            row=row, col=1, secondary_y=True
        )

        # Plot sync signal (scaled to fit on derivative axis)
        sync_scaled = sync_signal / np.max(sync_signal) * np.max(np.abs(derivative)) if np.max(sync_signal) > 0 else sync_signal
        fig.add_trace(
            go.Scatter(x=x_vals, y=sync_scaled, mode='lines',
                      line=dict(color='purple', width=1),
                      opacity=0.5, name='Sync', legendgroup='sync',
                      showlegend=(idx == 0)),
            row=row, col=1, secondary_y=True
        )

        # Add horizontal dashed lines for threshold
        if threshold is not None:
            # For edge_to_peak: threshold is on position (primary y-axis)
            # For derivative: threshold is on derivative (secondary y-axis)
            is_position_threshold = (label_method == 'edge_to_peak')
            fig.add_hline(y=threshold, line=dict(color='blue' if is_position_threshold else 'gray', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=not is_position_threshold)
            fig.add_hline(y=-threshold, line=dict(color='blue' if is_position_threshold else 'gray', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=not is_position_threshold)

        # Plot edge and peak markers for edge_to_peak method
        if markers is not None:
            # Edge up markers on POSITION trace (primary y-axis) - where edges are detected
            if len(markers['edge_up']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['edge_up'], y=data[markers['edge_up']],
                              mode='markers',
                              marker=dict(symbol='triangle-up', size=12, color='green',
                                         line=dict(color='white', width=1)),
                              opacity=0.7, name='Edge Up (pos)', legendgroup='edge_up',
                              showlegend=(idx == 0)),
                    row=row, col=1, secondary_y=False
                )

            # Edge down markers on POSITION trace (primary y-axis)
            if len(markers['edge_down']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['edge_down'], y=data[markers['edge_down']],
                              mode='markers',
                              marker=dict(symbol='triangle-down', size=12, color='red',
                                         line=dict(color='white', width=1)),
                              opacity=0.7, name='Edge Down (pos)', legendgroup='edge_down',
                              showlegend=(idx == 0)),
                    row=row, col=1, secondary_y=False
                )

            # Peak up markers on DERIVATIVE trace (secondary y-axis)
            if len(markers['peak_up']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['peak_up'], y=derivative[markers['peak_up']],
                              mode='markers',
                              marker=dict(symbol='circle', size=10, color='darkgreen'),
                              name='Peak Up (deriv)', legendgroup='peak_up',
                              showlegend=(idx == 0)),
                    row=row, col=1, secondary_y=True
                )

            # Peak down markers on DERIVATIVE trace (secondary y-axis)
            if len(markers['peak_down']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['peak_down'], y=derivative[markers['peak_down']],
                              mode='markers',
                              marker=dict(symbol='circle', size=10, color='darkred'),
                              name='Peak Down (deriv)', legendgroup='peak_down',
                              showlegend=(idx == 0)),
                    row=row, col=1, secondary_y=True
                )

        # Update y-axis labels
        fig.update_yaxes(title_text=f"{col_name} Direction", secondary_y=False, row=row, col=1,
                        title_font=dict(color='blue'), tickfont=dict(color='blue'))
        fig.update_yaxes(title_text="Deriv", secondary_y=True, row=row, col=1,
                        title_font=dict(color='red'), tickfont=dict(color='red'))

    # Update layout
    fig.update_xaxes(title_text="Sample", row=num_files, col=1)
    fig.update_layout(
        title=dict(text=f"Joystick Data ({col_name}) & Derivative - {session_name}",
                  font=dict(size=16)),
        height=200 * num_files,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.show()


if __name__ == "__main__":
    for session in sessions:
        session_path = os.path.join(base_path, session)
        plot_joystick_stacked(session_path, session)
