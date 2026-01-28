"""
Visualize Label Data for a Single Experiment

Displays joystick position, derivative, and label regions for both X and Y axes.
Same visualization style as visualize_session.py but for one experiment only.

Usage:
    python preprocessing/label_logic/visualize_exp.py
    python preprocessing/label_logic/visualize_exp.py --seed 42
    python preprocessing/label_logic/visualize_exp.py --exp-path /vol/data/.../session14_W_001/10
"""

import numpy as np
import glob
import os
import sys
import argparse
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

from preprocessing.signal_utils import apply_joystick_filters
from preprocessing.label_logic.labeling import (
    create_derivative_labels,
    create_edge_to_peak_labels,
    create_edge_to_derivative_labels
)

# Load label config
label_config_path = os.path.join(script_dir, "label_config.yaml")
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)

# Load main config for paths
main_config_path = os.path.join(project_dir, "config", "config.yaml")
with open(main_config_path, 'r') as f:
    main_config = yaml.safe_load(f)

# Paths from main config
base_data_path = main_config.get('global_setting', {}).get('paths', {}).get('base_data_path', '')
base_path = os.path.join(base_data_path, 'raw')

# Label settings from label_config
filters_config = label_config.get('filters', {})
label_method = label_config.get('method', 'derivative')
threshold_percent = label_config.get('threshold_percent', 5.0)
label_axis = label_config.get('axis', 'x')

# Display settings
display_config = label_config.get('display', {})
show_raw_trace = display_config.get('show_raw_trace', False)


def find_all_experiments():
    """Find all experiment paths in the raw data directory."""
    experiments = []
    session_dirs = glob.glob(os.path.join(base_path, "session*"))

    for session_dir in session_dirs:
        exp_dirs = glob.glob(os.path.join(session_dir, "*"))
        for exp_dir in exp_dirs:
            if os.path.isdir(exp_dir) and os.path.basename(exp_dir).isdigit():
                joystick_file = os.path.join(exp_dir, "_joystick.npy")
                if os.path.exists(joystick_file):
                    experiments.append(exp_dir)

    return sorted(experiments)


def select_random_experiment(seed=None):
    """Select a random experiment from available paths."""
    experiments = find_all_experiments()
    if not experiments:
        raise ValueError(f"No experiments found in {base_path}")

    rng = np.random.default_rng(seed)
    return rng.choice(experiments)


def find_label_regions(labels, label_values=None):
    """Find contiguous regions for each label value."""
    if label_values is None:
        label_values = [1, 2]

    regions = {}
    for label_val in label_values:
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        regions[label_val] = list(zip(starts, ends))

    return regions


def calculate_window_statistics(label_regions):
    """Calculate mean and median window widths for labeled regions."""
    up_widths = [end - start for start, end in label_regions.get(1, [])]
    down_widths = [end - start for start, end in label_regions.get(2, [])]
    all_widths = up_widths + down_widths

    def compute_stats(widths):
        if len(widths) == 0:
            return {'mean': 0.0, 'median': 0.0, 'count': 0}
        return {
            'mean': np.mean(widths),
            'median': np.median(widths),
            'count': len(widths)
        }

    return {
        'overall': compute_stats(all_widths),
        'up': compute_stats(up_widths),
        'down': compute_stats(down_widths)
    }


def plot_single_experiment(exp_path):
    """
    Create a visualization for a single experiment showing both X and Y axes.

    Layout:
    - Row 1: X axis (position + derivative + label regions)
    - Row 2: Y axis (position + derivative + label regions)
    """
    joystick_file = os.path.join(exp_path, "_joystick.npy")
    if not os.path.exists(joystick_file):
        raise FileNotFoundError(f"Joystick file not found: {joystick_file}")

    joystick_data = np.load(joystick_file, allow_pickle=True)
    sync_signal = joystick_data[:, 3]

    # Define coordinates to plot: (column_index, name)
    coordinates = [(1, "X"), (2, "Y")]
    num_rows = len(coordinates)

    # Get experiment info
    session_name = os.path.basename(os.path.dirname(exp_path))
    exp_num = os.path.basename(exp_path)
    exp_name = f"{session_name}/{exp_num}"

    # Build subplot titles
    subplot_titles = [f"{exp_name} - Joystick {coord_name}" for _, coord_name in coordinates]

    # Create subplots with secondary y-axis for each row
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}] for _ in range(num_rows)],
        subplot_titles=subplot_titles
    )

    # Collect statistics for printing
    all_stats = {}

    for coord_idx, (col, col_name) in enumerate(coordinates):
        row = coord_idx + 1
        is_first_row = (row == 1)

        raw_data = joystick_data[:, col]

        # Apply filters to position data
        data = apply_joystick_filters(raw_data.copy(), filters_config, 'position')

        derivative = np.gradient(data)

        # Apply filters to derivative
        derivative = apply_joystick_filters(derivative, filters_config, 'derivative')

        x_vals = np.arange(len(data))

        # Compute labels using configured method
        markers = None
        threshold = None
        deriv_threshold = None
        if label_method == 'edge_to_peak':
            labels, threshold, markers = create_edge_to_peak_labels(data, derivative, threshold_percent)
        elif label_method == 'edge_to_derivative':
            labels, threshold, deriv_threshold, markers = create_edge_to_derivative_labels(data, derivative, threshold_percent)
        else:  # default: 'derivative'
            labels, threshold = create_derivative_labels(derivative, threshold_percent)

        # Shade regions based on labels
        label_regions = find_label_regions(labels)

        # Store statistics
        all_stats[col_name] = calculate_window_statistics(label_regions)

        label_colors = {1: 'rgba(144, 238, 144, 0.3)', 2: 'rgba(240, 128, 128, 0.3)'}
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
                          showlegend=is_first_row),
                row=row, col=1, secondary_y=False
            )

        # Plot filtered data on primary y-axis
        fig.add_trace(
            go.Scatter(x=x_vals, y=data, mode='lines',
                      line=dict(color='blue', width=1),
                      name='Filtered', legendgroup='filtered',
                      showlegend=is_first_row),
            row=row, col=1, secondary_y=False
        )

        # Plot derivative on secondary y-axis
        fig.add_trace(
            go.Scatter(x=x_vals, y=derivative, mode='lines',
                      line=dict(color='red', width=1),
                      opacity=0.7, name='Derivative', legendgroup='derivative',
                      showlegend=is_first_row),
            row=row, col=1, secondary_y=True
        )

        # Plot sync signal (scaled to fit on derivative axis)
        if np.max(sync_signal) > 0:
            sync_scaled = sync_signal / np.max(sync_signal) * np.max(np.abs(derivative))
        else:
            sync_scaled = sync_signal
        fig.add_trace(
            go.Scatter(x=x_vals, y=sync_scaled, mode='lines',
                      line=dict(color='purple', width=1),
                      opacity=0.5, name='Sync', legendgroup='sync',
                      showlegend=is_first_row),
            row=row, col=1, secondary_y=True
        )

        # Add horizontal dashed lines for threshold
        if threshold is not None:
            is_position_threshold = (label_method in ['edge_to_peak', 'edge_to_derivative'])
            fig.add_hline(y=threshold, line=dict(color='blue' if is_position_threshold else 'gray', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=not is_position_threshold)
            fig.add_hline(y=-threshold, line=dict(color='blue' if is_position_threshold else 'gray', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=not is_position_threshold)

        # Add derivative threshold lines for edge_to_derivative method
        if deriv_threshold is not None:
            fig.add_hline(y=deriv_threshold, line=dict(color='orange', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=True)
            fig.add_hline(y=-deriv_threshold, line=dict(color='orange', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=True)

        # Add markers for edge-based methods
        if markers is not None:
            # Edge up markers on POSITION trace
            if len(markers.get('edge_up', [])) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['edge_up'], y=data[markers['edge_up']],
                              mode='markers',
                              marker=dict(symbol='triangle-up', size=12, color='green',
                                         line=dict(color='white', width=1)),
                              opacity=0.7, name='Edge Up', legendgroup='edge_up',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=False
                )

            # Edge down markers on POSITION trace
            if len(markers.get('edge_down', [])) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['edge_down'], y=data[markers['edge_down']],
                              mode='markers',
                              marker=dict(symbol='triangle-down', size=12, color='red',
                                         line=dict(color='white', width=1)),
                              opacity=0.7, name='Edge Down', legendgroup='edge_down',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=False
                )

            # Peak markers for edge_to_peak method
            if 'peak_up' in markers and len(markers['peak_up']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['peak_up'], y=derivative[markers['peak_up']],
                              mode='markers',
                              marker=dict(symbol='circle', size=10, color='darkgreen'),
                              name='Peak Up', legendgroup='peak_up',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=True
                )

            if 'peak_down' in markers and len(markers['peak_down']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['peak_down'], y=derivative[markers['peak_down']],
                              mode='markers',
                              marker=dict(symbol='circle', size=10, color='darkred'),
                              name='Peak Down', legendgroup='peak_down',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=True
                )

            # Derivative crossing markers for edge_to_derivative method
            if 'deriv_cross_up' in markers and len(markers['deriv_cross_up']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['deriv_cross_up'], y=derivative[markers['deriv_cross_up']],
                              mode='markers',
                              marker=dict(symbol='square', size=10, color='darkgreen',
                                         line=dict(color='white', width=1)),
                              name='Deriv Cross Up', legendgroup='deriv_cross_up',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=True
                )

            if 'deriv_cross_down' in markers and len(markers['deriv_cross_down']) > 0:
                fig.add_trace(
                    go.Scatter(x=markers['deriv_cross_down'], y=derivative[markers['deriv_cross_down']],
                              mode='markers',
                              marker=dict(symbol='square', size=10, color='darkred',
                                         line=dict(color='white', width=1)),
                              name='Deriv Cross Down', legendgroup='deriv_cross_down',
                              showlegend=is_first_row),
                    row=row, col=1, secondary_y=True
                )

        # Update y-axis labels
        fig.update_yaxes(title_text=f"{col_name}", secondary_y=False, row=row, col=1,
                        title_font=dict(color='blue'), tickfont=dict(color='blue'))
        fig.update_yaxes(title_text="Deriv", secondary_y=True, row=row, col=1,
                        title_font=dict(color='red'), tickfont=dict(color='red'))

    # Update layout
    fig.update_xaxes(title_text="Sample", row=num_rows, col=1)
    fig.update_layout(
        title=dict(
            text=f"Joystick & Labels - {exp_name} (method: {label_method}, thresh: {threshold_percent}%)",
            font=dict(size=16)
        ),
        height=400 * num_rows,
        width=1400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Method: {label_method}, Threshold: {threshold_percent}%")
    print(f"Label axis in config: {label_axis}")
    print(f"{'='*60}")
    print(f"\nWindow Width Statistics (samples):\n")

    for coord_name in ['X', 'Y']:
        stats = all_stats[coord_name]
        print(f"  {coord_name} Coordinate:")
        print(f"    Overall: mean={stats['overall']['mean']:.1f}, median={stats['overall']['median']:.1f}, count={stats['overall']['count']}")
        print(f"    Up:      mean={stats['up']['mean']:.1f}, median={stats['up']['median']:.1f}, count={stats['up']['count']}")
        print(f"    Down:    mean={stats['down']['mean']:.1f}, median={stats['down']['median']:.1f}, count={stats['down']['count']}")
        print()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize label data for a single experiment')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', '-e', type=str, default=None,
                        help='Specific experiment path (overrides random selection)')
    args = parser.parse_args()

    # Select experiment
    if args.exp_path:
        exp_path = args.exp_path
    else:
        exp_path = select_random_experiment(seed=args.seed)

    print(f"Selected experiment: {exp_path}")

    # Create and show visualization
    fig = plot_single_experiment(exp_path)
    fig.show()

    return 0


if __name__ == '__main__':
    exit(main())
