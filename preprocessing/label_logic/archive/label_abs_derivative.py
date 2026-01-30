"""
Absolute Derivative Label Logic

Labels movement periods using absolute derivative threshold crossing:
- Start: abs(derivative) rises above threshold AND position near center
- Stop: abs(derivative) falls below threshold
- Direction: sign(derivative) at start marker

Usage:
    python preprocessing/label_logic/label_abs_derivative.py
    python preprocessing/label_logic/label_abs_derivative.py --seed 42
    python preprocessing/label_logic/label_abs_derivative.py --deriv-thresh 10 --pos-thresh 5
"""

import numpy as np
import os
import sys
import argparse
import glob
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


# Load configs
label_config_path = os.path.join(script_dir, "label_config.yaml")
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)

main_config_path = os.path.join(project_dir, "config", "config.yaml")
with open(main_config_path, 'r') as f:
    main_config = yaml.safe_load(f)

# Paths and settings
base_data_path = main_config.get('global_setting', {}).get('paths', {}).get('base_data_path', '')
base_path = os.path.join(base_data_path, 'raw')
filters_config = label_config.get('filters', {})


def create_abs_derivative_labels(
    position,
    derivative,
    deriv_threshold_percent=10.0,
    pos_threshold_percent=5.0
):
    """
    Label movements using absolute derivative threshold crossing.

    Start condition: abs(derivative) rises above deriv_threshold
                     AND abs(position) < pos_threshold (near center)
    Stop condition:  abs(derivative) falls below deriv_threshold
    Direction:       sign(derivative) at start → 1=positive, 2=negative

    Args:
        position: Filtered joystick position signal [n]
        derivative: Filtered derivative of position [n]
        deriv_threshold_percent: Threshold for abs(derivative) as % of range
        pos_threshold_percent: Position must be within ±this% of range from center

    Returns:
        labels: [n] array, 0=noise, 1=positive movement, 2=negative movement
        thresholds: {'deriv': float, 'pos': float}
        markers: {'start': [...], 'stop': [...], 'rejected': [...]}
    """
    n = len(position)
    labels = np.zeros(n, dtype=np.int64)

    # Compute thresholds
    deriv_range = np.max(np.abs(derivative))
    pos_range = max(abs(position.max()), abs(position.min()))

    deriv_threshold = deriv_threshold_percent / 100.0 * deriv_range
    pos_threshold = pos_threshold_percent / 100.0 * pos_range

    # Compute absolute derivative
    abs_deriv = np.abs(derivative)

    # Find threshold crossings on abs(derivative)
    above_threshold = abs_deriv > deriv_threshold

    # Rising edges: transition from below to above threshold
    transitions = np.diff(above_threshold.astype(int))
    rising_edges = np.where(transitions == 1)[0] + 1
    falling_edges = np.where(transitions == -1)[0] + 1

    # Track markers
    start_markers = []
    stop_markers = []
    rejected_markers = []

    for start_idx in rising_edges:
        # Validate: position must be near center
        if np.abs(position[start_idx]) > pos_threshold:
            rejected_markers.append(start_idx)
            continue

        # Find corresponding stop (next falling edge after start)
        stops_after = falling_edges[falling_edges > start_idx]
        if len(stops_after) == 0:
            # No stop found, movement continues to end
            stop_idx = n
        else:
            stop_idx = stops_after[0]

        # Determine direction from derivative sign at start
        direction = 1 if derivative[start_idx] > 0 else 2

        # Apply labels
        labels[start_idx:stop_idx] = direction

        start_markers.append(start_idx)
        stop_markers.append(min(stop_idx, n - 1))

    thresholds = {
        'deriv': deriv_threshold,
        'pos': pos_threshold,
        'deriv_percent': deriv_threshold_percent,
        'pos_percent': pos_threshold_percent
    }

    markers = {
        'start': np.array(start_markers, dtype=int),
        'stop': np.array(stop_markers, dtype=int),
        'rejected': np.array(rejected_markers, dtype=int)
    }

    return labels, thresholds, markers


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


def visualize_labels(exp_path, deriv_threshold_percent=10.0, pos_threshold_percent=5.0):
    """
    Visualize the absolute derivative labeling for an experiment.

    Layout:
    - Row 1: Joystick X with labels
    - Row 2: Joystick Y with labels

    Each row shows: position, abs(derivative), thresholds, markers, labeled regions
    """
    # Load joystick data
    joystick_file = os.path.join(exp_path, "_joystick.npy")
    joystick_data = np.load(joystick_file, allow_pickle=True)

    # Get experiment info
    session_name = os.path.basename(os.path.dirname(exp_path))
    exp_num = os.path.basename(exp_path)
    exp_name = f"{session_name}/{exp_num}"

    # Process both axes
    axes_data = [
        (1, 'X', 'LEFT/RIGHT'),
        (2, 'Y', 'UP/DOWN')
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

    # Colors for labels
    label_colors = {
        0: 'rgba(128, 128, 128, 0.2)',
        1: 'rgba(0, 200, 0, 0.3)',
        2: 'rgba(200, 0, 0, 0.3)'
    }
    label_names = {0: 'Noise', 1: 'Positive', 2: 'Negative'}

    all_stats = {}

    for row, (col_idx, axis_name, direction_name) in enumerate(axes_data, 1):
        is_first = (row == 1)

        # Get and filter data
        raw_pos = joystick_data[:, col_idx]
        position = apply_joystick_filters(raw_pos.copy(), filters_config, 'position')
        derivative = apply_joystick_filters(np.gradient(position), filters_config, 'derivative')
        abs_deriv = np.abs(derivative)
        x_vals = np.arange(len(position))

        # Create labels
        labels, thresholds, markers = create_abs_derivative_labels(
            position, derivative, deriv_threshold_percent, pos_threshold_percent
        )

        # Statistics
        unique, counts = np.unique(labels, return_counts=True)
        all_stats[axis_name] = {
            'distribution': dict(zip(unique.tolist(), counts.tolist())),
            'n_valid_starts': len(markers['start']),
            'n_rejected': len(markers['rejected']),
            'thresholds': thresholds
        }

        # Add label regions as colored rectangles
        for label_val in [1, 2]:
            mask = labels == label_val
            diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor=label_colors[label_val],
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )

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

        # Plot position threshold lines
        fig.add_hline(
            y=thresholds['pos'], line=dict(color='blue', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=False
        )
        fig.add_hline(
            y=-thresholds['pos'], line=dict(color='blue', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=False
        )

        # Plot abs(derivative) (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=abs_deriv,
                mode='lines',
                line=dict(color='orange', width=1),
                opacity=0.7,
                name='|Derivative|',
                legendgroup='abs_deriv',
                showlegend=is_first
            ),
            row=row, col=1, secondary_y=True
        )

        # Plot derivative threshold line
        fig.add_hline(
            y=thresholds['deriv'], line=dict(color='orange', dash='dash', width=1),
            opacity=0.7, row=row, col=1, secondary_y=True
        )

        # Plot start markers (valid)
        if len(markers['start']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['start'],
                    y=position[markers['start']],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green',
                               line=dict(color='white', width=1)),
                    name='Start (valid)',
                    legendgroup='start',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot stop markers
        if len(markers['stop']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['stop'],
                    y=position[markers['stop']],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red',
                               line=dict(color='white', width=1)),
                    name='Stop',
                    legendgroup='stop',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot rejected markers (not from center)
        if len(markers['rejected']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['rejected'],
                    y=position[markers['rejected']],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='gray',
                               line=dict(color='darkgray', width=2)),
                    name='Rejected (not center)',
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
            title_text="|Derivative|",
            secondary_y=True, row=row, col=1,
            title_font=dict(color='orange'),
            tickfont=dict(color='orange')
        )

    # Layout
    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"Abs Derivative Labels - {exp_name} (deriv_thresh={deriv_threshold_percent}%, pos_thresh={pos_threshold_percent}%)",
            font=dict(size=16)
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
    print(f"{'='*60}")

    for axis_name, stats in all_stats.items():
        print(f"\n{axis_name} Axis:")
        print(f"  Thresholds: deriv={stats['thresholds']['deriv']:.4f}, pos={stats['thresholds']['pos']:.4f}")
        print(f"  Valid starts: {stats['n_valid_starts']}")
        print(f"  Rejected (not from center): {stats['n_rejected']}")
        print(f"  Label distribution: {stats['distribution']}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize absolute derivative labeling')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', '-e', type=str, default=None,
                        help='Specific experiment path')
    parser.add_argument('--deriv-thresh', type=float, default=10.0,
                        help='Derivative threshold as %% of range (default: 10)')
    parser.add_argument('--pos-thresh', type=float, default=5.0,
                        help='Position threshold as %% of range (default: 5)')
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
        pos_threshold_percent=args.pos_thresh
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
