"""
Visualize Position Peak Labels for Experiments

Displays joystick position, velocity, and 5-class label regions for both X and Y axes.

Usage:
    python preprocessing/visualization/visualize_labels.py
    python preprocessing/visualization/visualize_labels.py --seed 42
    python preprocessing/visualization/visualize_labels.py --exp-path /vol/data/.../session002/exp000
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
from preprocessing.label_logic.label_logic import create_5class_position_peak_labels

# Load configs
label_config_path = os.path.join(project_dir, "preprocessing", "label_logic", "label_config.yaml")
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)

main_config_path = os.path.join(project_dir, "config", "config.yaml")
with open(main_config_path, 'r') as f:
    main_config = yaml.safe_load(f)

# Paths and settings
base_data_path = main_config.get('global_setting', {}).get('paths', {}).get('base_data_path', '')
base_path = os.path.join(base_data_path, 'raw')
filters_config = label_config.get('filters', {})

# Position peak settings from config
position_peak_config = label_config.get('position_peak', {})
DEFAULT_DERIV_THRESH = position_peak_config.get('deriv_threshold_percent', 10.0)
DEFAULT_POS_THRESH = position_peak_config.get('pos_threshold_percent', 5.0)
DEFAULT_PEAK_WINDOW = position_peak_config.get('peak_window', 3)
DEFAULT_TIMEOUT = position_peak_config.get('timeout_samples', 500)


def find_all_experiments():
    """Find all experiment paths recursively."""
    pattern = os.path.join(base_path, "**", "_joystick.npy")
    joystick_files = glob.glob(pattern, recursive=True)
    return sorted([os.path.dirname(f) for f in joystick_files])


def select_random_experiment(seed=None):
    """Select a random experiment."""
    experiments = find_all_experiments()
    if not experiments:
        raise ValueError(f"No experiments found in {base_path}")
    rng = np.random.default_rng(seed)
    return rng.choice(experiments)


def visualize_labels(exp_path, deriv_threshold_percent=None, pos_threshold_percent=None,
                     peak_window=None, timeout_samples=None):
    """
    Visualize the position peak labeling for an experiment.

    Layout:
    - Row 1: Joystick X with 5-class labels
    - Row 2: Joystick Y with 5-class labels

    Args:
        exp_path: Path to experiment directory containing _joystick.npy
        deriv_threshold_percent: Velocity threshold as % of range (default from config)
        pos_threshold_percent: Position center threshold as % of range (default from config)
        peak_window: Samples to confirm position reversal (default from config)
        timeout_samples: Timeout samples per phase (default from config)

    Returns:
        plotly Figure object
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

    # Load joystick data
    joystick_file = os.path.join(exp_path, "_joystick.npy")
    joystick_data = np.load(joystick_file, allow_pickle=True)

    # Get experiment info
    session_name = os.path.basename(os.path.dirname(exp_path))
    exp_num = os.path.basename(exp_path)
    exp_name = f"{session_name}/{exp_num}"

    # Get and filter data for both axes
    x_raw = joystick_data[:, 1]
    y_raw = joystick_data[:, 2]
    x_position = apply_joystick_filters(x_raw.copy(), filters_config, 'position')
    y_position = apply_joystick_filters(y_raw.copy(), filters_config, 'position')
    x_velocity = apply_joystick_filters(np.gradient(x_position), filters_config, 'derivative')
    y_velocity = apply_joystick_filters(np.gradient(y_position), filters_config, 'derivative')

    # Compute 5-class labels
    labels_5class, thresholds, markers = create_5class_position_peak_labels(
        x_position, y_position, x_velocity, y_velocity,
        deriv_threshold_percent, pos_threshold_percent,
        peak_window, timeout_samples
    )

    # Process both axes for per-axis visualization
    axes_data = [
        (1, 'X', 'LEFT/RIGHT', x_position, x_velocity, markers['x']),
        (2, 'Y', 'UP/DOWN', y_position, y_velocity, markers['y'])
    ]

    subplot_titles = [
        f"{exp_name} - Joystick X (LEFT/RIGHT)",
        f"{exp_name} - Joystick Y (UP/DOWN)"
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=subplot_titles
    )

    # 5-class colors from config
    classes_config = label_config.get('classes', {})
    class_colors = classes_config.get('colors', {})
    label_colors = {int(k): v for k, v in class_colors.items()} if class_colors else {
        0: 'rgba(128, 128, 128, 0.3)',   # Noise - gray
        1: 'rgba(0, 200, 0, 0.4)',       # Up - green
        2: 'rgba(200, 0, 0, 0.4)',       # Down - red
        3: 'rgba(0, 100, 255, 0.4)',     # Left - blue
        4: 'rgba(255, 165, 0, 0.4)',     # Right - orange
    }
    class_names = classes_config.get('names', {})
    label_names = {int(k): v for k, v in class_names.items()} if class_names else {
        0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'
    }

    # Find 5-class label regions
    def find_label_regions(labels, label_values):
        regions = {}
        for label_val in label_values:
            mask = labels == label_val
            diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            regions[label_val] = list(zip(starts, ends))
        return regions

    label_regions = find_label_regions(labels_5class, [1, 2, 3, 4])

    # Statistics for 5-class labels
    unique, counts = np.unique(labels_5class, return_counts=True)
    stats_5class = {
        'distribution': dict(zip(unique.tolist(), counts.tolist())),
        'regions': {label_names.get(k, f'Class {k}'): len(v) for k, v in label_regions.items()},
        'thresholds': thresholds
    }

    # Add 5-class label regions as shaded areas to BOTH rows
    for row in [1, 2]:
        xref = "x" if row == 1 else "x2"
        yref = "y domain" if row == 1 else "y3 domain"
        for label_val, regions in label_regions.items():
            color = label_colors.get(label_val, 'rgba(128, 128, 128, 0.3)')
            for start, end in regions:
                fig.add_shape(
                    type="rect",
                    x0=start, x1=end,
                    y0=0, y1=1,
                    xref=xref, yref=yref,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                )

    # Add legend entries for 5-class labels (dummy traces)
    for label_val in [1, 2, 3, 4]:
        label_name = label_names.get(label_val, f'Class {label_val}')
        color = label_colors.get(label_val, 'rgba(128, 128, 128, 0.4)')
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color, symbol='square'),
                name=f'{label_name}',
                legendgroup=f'label_{label_val}',
                showlegend=True
            ),
            row=1, col=1
        )

    # Per-axis statistics
    all_stats = {'5class': stats_5class}

    for row, (col_idx, axis_name, direction_name, position, velocity, axis_markers) in enumerate(axes_data, 1):
        is_first = (row == 1)
        x_vals = np.arange(len(position))

        # Per-axis statistics from markers
        all_stats[axis_name] = {
            'n_movements': len(axis_markers['start']),
            'n_peaks': len(axis_markers['peak']),
            'n_rejected': len(axis_markers['rejected']),
            'n_timeout': len(axis_markers['timeout']),
        }

        # Get per-axis thresholds
        axis_prefix = 'x' if axis_name == 'X' else 'y'
        pos_thresh = thresholds[f'{axis_prefix}_pos']
        deriv_thresh = thresholds[f'{axis_prefix}_deriv']

        # Plot position (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=position,
                mode='lines',
                line=dict(color='blue', width=1.5),
                name='Position',
                legendgroup='position',
                showlegend=is_first
            ),
            row=row, col=1, secondary_y=False
        )

        # Plot position threshold lines (center zone)
        fig.add_hline(
            y=pos_thresh, line=dict(color='blue', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=False
        )
        fig.add_hline(
            y=-pos_thresh, line=dict(color='blue', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=False
        )

        # Plot velocity (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=velocity,
                mode='lines',
                line=dict(color='orange', width=1),
                opacity=0.7,
                name='Velocity',
                legendgroup='velocity',
                showlegend=is_first
            ),
            row=row, col=1, secondary_y=True
        )

        # Plot velocity threshold lines
        fig.add_hline(
            y=deriv_thresh, line=dict(color='orange', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=True
        )
        fig.add_hline(
            y=-deriv_thresh, line=dict(color='orange', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=True
        )

        # Plot start markers
        if len(axis_markers['start']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=axis_markers['start'],
                    y=position[axis_markers['start']],
                    mode='markers',
                    marker=dict(symbol='triangle-right', size=12, color='green',
                               line=dict(color='white', width=1)),
                    name='Start',
                    legendgroup='start',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot peak markers (on position trace)
        if len(axis_markers['peak']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=axis_markers['peak'],
                    y=position[axis_markers['peak']],
                    mode='markers',
                    marker=dict(symbol='star', size=14, color='purple',
                               line=dict(color='white', width=1)),
                    name='Peak',
                    legendgroup='peak',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot stop markers
        if len(axis_markers['stop']) > 0:
            # Separate normal stops from timeout stops
            timeout_set = set(axis_markers['timeout'].tolist())
            normal_stops = [s for s in axis_markers['stop'] if s not in timeout_set]

            if len(normal_stops) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=normal_stops,
                        y=position[normal_stops],
                        mode='markers',
                        marker=dict(symbol='square', size=10, color='red',
                                   line=dict(color='white', width=1)),
                        name='Stop',
                        legendgroup='stop',
                        showlegend=is_first
                    ),
                    row=row, col=1, secondary_y=False
                )

        # Plot timeout markers
        if len(axis_markers['timeout']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=axis_markers['timeout'],
                    y=position[axis_markers['timeout']],
                    mode='markers',
                    marker=dict(symbol='square', size=10, color='orange',
                               line=dict(color='white', width=1)),
                    name='Timeout',
                    legendgroup='timeout',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot rejected markers
        if len(axis_markers['rejected']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=axis_markers['rejected'],
                    y=position[axis_markers['rejected']],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='gray',
                               line=dict(color='darkgray', width=2)),
                    name='Rejected',
                    legendgroup='rejected',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Axis labels
        fig.update_yaxes(
            title_text=f"Position {axis_name}",
            secondary_y=False, row=row, col=1,
            title_font=dict(color='blue'),
            tickfont=dict(color='blue')
        )
        fig.update_yaxes(
            title_text="Velocity",
            secondary_y=True, row=row, col=1,
            title_font=dict(color='orange'),
            tickfont=dict(color='orange')
        )

    # Layout
    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"Position Peak Labels - {exp_name} (deriv={deriv_threshold_percent}%, pos={pos_threshold_percent}%, window={peak_window})",
            font=dict(size=14)
        ),
        height=800,
        width=1400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Thresholds: deriv={deriv_threshold_percent}%, pos={pos_threshold_percent}%")
    print(f"Peak window: {peak_window} samples, Timeout: {timeout_samples} samples")
    print(f"{'='*60}")

    # 5-class statistics
    stats_5c = all_stats['5class']
    print(f"\n5-Class Labels (position_peak):")
    print(f"  Distribution: {stats_5c['distribution']}")
    print(f"  Regions: {stats_5c['regions']}")

    # Per-axis statistics
    for axis_name in ['X', 'Y']:
        stats = all_stats[axis_name]
        print(f"\n{axis_name} Axis:")
        print(f"  Movements detected: {stats['n_movements']}")
        print(f"  Peaks detected: {stats['n_peaks']}")
        print(f"  Rejected (not from center): {stats['n_rejected']}")
        print(f"  Ended by timeout: {stats['n_timeout']}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize position peak labeling')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', '-e', type=str, default=None,
                        help='Specific experiment path')
    parser.add_argument('--deriv-thresh', type=float, default=None,
                        help=f'Velocity threshold as %% of range (config: {DEFAULT_DERIV_THRESH})')
    parser.add_argument('--pos-thresh', type=float, default=None,
                        help=f'Position center threshold as %% of range (config: {DEFAULT_POS_THRESH})')
    parser.add_argument('--peak-window', type=int, default=None,
                        help=f'Samples to confirm position reversal (config: {DEFAULT_PEAK_WINDOW})')
    parser.add_argument('--timeout', type=int, default=None,
                        help=f'Timeout samples per phase (config: {DEFAULT_TIMEOUT})')
    parser.add_argument('--save', type=str, default=None,
                        help='Save to HTML file instead of showing')
    args = parser.parse_args()

    # Select experiment
    if args.exp_path:
        exp_path = args.exp_path
    else:
        exp_path = select_random_experiment(seed=args.seed)

    print(f"Selected experiment: {exp_path}")

    # Create visualization
    fig = visualize_labels(
        exp_path,
        deriv_threshold_percent=args.deriv_thresh,
        pos_threshold_percent=args.pos_thresh,
        peak_window=args.peak_window,
        timeout_samples=args.timeout
    )

    # Show or save
    if args.save:
        fig.write_html(args.save)
        print(f"\nSaved to: {args.save}")
    else:
        fig.show()

    return 0


if __name__ == '__main__':
    exit(main())
