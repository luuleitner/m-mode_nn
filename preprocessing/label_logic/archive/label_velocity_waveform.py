"""
Velocity Waveform Label Logic

Labels movement periods by tracking the full velocity waveform:
- Start: |velocity| rises above threshold AND position near center
- End: velocity sign has changed AND |velocity| falls below threshold
- Fallback: timeout if movement doesn't complete

This captures the full movement cycle (out and back) as a single labeled region.

Usage:
    python preprocessing/label_logic/label_velocity_waveform.py
    python preprocessing/label_logic/label_velocity_waveform.py --seed 42
    python preprocessing/label_logic/label_velocity_waveform.py --deriv-thresh 10 --pos-thresh 5
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


def create_velocity_waveform_labels(
    position,
    velocity,
    deriv_threshold_percent=10.0,
    pos_threshold_percent=5.0,
    timeout_samples=500
):
    """
    Label movements by tracking the full velocity waveform cycle.

    Start: |velocity| > deriv_threshold AND |position| < pos_threshold
    End: velocity sign changed at least once AND |velocity| < deriv_threshold
    Fallback: end after timeout_samples if sign never changes

    Args:
        position: Filtered joystick position signal [n]
        velocity: Filtered derivative of position [n]
        deriv_threshold_percent: Threshold for |velocity| as % of range
        pos_threshold_percent: Position must be within ±this% of range from center
        timeout_samples: Max samples before forcing end (held position fallback)

    Returns:
        labels: [n] array, 0=noise, 1=positive movement, 2=negative movement
        thresholds: {'deriv': float, 'pos': float}
        markers: {'start': [...], 'stop': [...], 'sign_change': [...], 'rejected': [...], 'timeout': [...]}
    """
    n = len(position)
    labels = np.zeros(n, dtype=np.int64)

    # Compute thresholds
    vel_range = np.max(np.abs(velocity))
    pos_range = max(abs(position.max()), abs(position.min()))

    deriv_threshold = deriv_threshold_percent / 100.0 * vel_range
    pos_threshold = pos_threshold_percent / 100.0 * pos_range

    # Track markers
    start_markers = []
    stop_markers = []
    sign_change_markers = []
    rejected_markers = []
    timeout_markers = []

    # State tracking
    in_movement = False
    sign_changed = False
    start_idx = 0
    start_sign = 0
    direction = 0

    i = 0
    while i < n:
        vel = velocity[i]
        pos = position[i]
        abs_vel = abs(vel)
        abs_pos = abs(pos)

        if not in_movement:
            # Check for movement start
            if abs_vel > deriv_threshold:
                # Validate: position must be near center
                if abs_pos > pos_threshold:
                    rejected_markers.append(i)
                    i += 1
                    continue

                # Valid start
                in_movement = True
                sign_changed = False
                start_idx = i
                start_sign = 1 if vel > 0 else -1
                direction = 1 if vel > 0 else 2  # 1=positive, 2=negative
                start_markers.append(i)

        else:
            # In movement - check for sign change
            current_sign = 1 if vel > 0 else (-1 if vel < 0 else 0)

            if not sign_changed and current_sign != 0 and current_sign != start_sign:
                sign_changed = True
                sign_change_markers.append(i)

            # Check for movement end
            samples_elapsed = i - start_idx

            if sign_changed and abs_vel < deriv_threshold:
                # Normal end: sign changed and velocity below threshold
                labels[start_idx:i] = direction
                stop_markers.append(i)
                in_movement = False

            elif samples_elapsed >= timeout_samples:
                # Timeout: movement held without returning
                labels[start_idx:i] = direction
                stop_markers.append(i)
                timeout_markers.append(i)
                in_movement = False

        i += 1

    # Handle case where movement extends to end of signal
    if in_movement:
        labels[start_idx:n] = direction
        stop_markers.append(n - 1)
        if not sign_changed:
            timeout_markers.append(n - 1)

    thresholds = {
        'deriv': deriv_threshold,
        'pos': pos_threshold,
        'deriv_percent': deriv_threshold_percent,
        'pos_percent': pos_threshold_percent,
        'timeout': timeout_samples
    }

    markers = {
        'start': np.array(start_markers, dtype=int),
        'stop': np.array(stop_markers, dtype=int),
        'sign_change': np.array(sign_change_markers, dtype=int),
        'rejected': np.array(rejected_markers, dtype=int),
        'timeout': np.array(timeout_markers, dtype=int)
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


def visualize_labels(exp_path, deriv_threshold_percent=10.0, pos_threshold_percent=5.0, timeout_samples=500):
    """
    Visualize the velocity waveform labeling for an experiment.

    Layout:
    - Row 1: Joystick X with labels
    - Row 2: Joystick Y with labels

    Each row shows: position, velocity, thresholds, markers, labeled regions
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

    all_stats = {}

    for row, (col_idx, axis_name, direction_name) in enumerate(axes_data, 1):
        is_first = (row == 1)

        # Get and filter data
        raw_pos = joystick_data[:, col_idx]
        position = apply_joystick_filters(raw_pos.copy(), filters_config, 'position')
        velocity = apply_joystick_filters(np.gradient(position), filters_config, 'derivative')
        x_vals = np.arange(len(position))

        # Create labels
        labels, thresholds, markers = create_velocity_waveform_labels(
            position, velocity, deriv_threshold_percent, pos_threshold_percent, timeout_samples
        )

        # Statistics
        unique, counts = np.unique(labels, return_counts=True)
        all_stats[axis_name] = {
            'distribution': dict(zip(unique.tolist(), counts.tolist())),
            'n_movements': len(markers['start']),
            'n_rejected': len(markers['rejected']),
            'n_timeout': len(markers['timeout']),
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

        # Plot velocity (secondary y-axis) - NOT abs, show full waveform
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

        # Plot velocity threshold lines (both + and -)
        fig.add_hline(
            y=thresholds['deriv'], line=dict(color='orange', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=True
        )
        fig.add_hline(
            y=-thresholds['deriv'], line=dict(color='orange', dash='dash', width=1),
            opacity=0.5, row=row, col=1, secondary_y=True
        )
        # Zero line for velocity
        fig.add_hline(
            y=0, line=dict(color='gray', dash='dot', width=1),
            opacity=0.3, row=row, col=1, secondary_y=True
        )

        # Plot start markers (valid)
        if len(markers['start']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['start'],
                    y=position[markers['start']],
                    mode='markers',
                    marker=dict(symbol='triangle-right', size=12, color='green',
                               line=dict(color='white', width=1)),
                    name='Start',
                    legendgroup='start',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=False
            )

        # Plot sign change markers
        if len(markers['sign_change']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['sign_change'],
                    y=velocity[markers['sign_change']],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, color='purple',
                               line=dict(color='white', width=1)),
                    name='Sign Change',
                    legendgroup='sign_change',
                    showlegend=is_first
                ),
                row=row, col=1, secondary_y=True
            )

        # Plot stop markers
        if len(markers['stop']) > 0:
            # Separate normal stops from timeout stops
            timeout_set = set(markers['timeout'].tolist())
            normal_stops = [s for s in markers['stop'] if s not in timeout_set]

            if len(normal_stops) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=normal_stops,
                        y=position[normal_stops],
                        mode='markers',
                        marker=dict(symbol='square', size=10, color='red',
                                   line=dict(color='white', width=1)),
                        name='Stop (normal)',
                        legendgroup='stop',
                        showlegend=is_first
                    ),
                    row=row, col=1, secondary_y=False
                )

        # Plot timeout markers
        if len(markers['timeout']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=markers['timeout'],
                    y=position[markers['timeout']],
                    mode='markers',
                    marker=dict(symbol='square', size=10, color='orange',
                               line=dict(color='white', width=1)),
                    name='Stop (timeout)',
                    legendgroup='timeout',
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
            text=f"Velocity Waveform Labels - {exp_name} (deriv={deriv_threshold_percent}%, pos={pos_threshold_percent}%, timeout={timeout_samples})",
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
    print(f"Timeout: {timeout_samples} samples")
    print(f"{'='*60}")

    for axis_name, stats in all_stats.items():
        print(f"\n{axis_name} Axis:")
        print(f"  Thresholds: deriv={stats['thresholds']['deriv']:.4f}, pos={stats['thresholds']['pos']:.4f}")
        print(f"  Movements detected: {stats['n_movements']}")
        print(f"  Rejected (not from center): {stats['n_rejected']}")
        print(f"  Ended by timeout: {stats['n_timeout']}")
        print(f"  Label distribution: {stats['distribution']}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize velocity waveform labeling')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', '-e', type=str, default=None,
                        help='Specific experiment path')
    parser.add_argument('--deriv-thresh', type=float, default=10.0,
                        help='Velocity threshold as %% of range (default: 10)')
    parser.add_argument('--pos-thresh', type=float, default=5.0,
                        help='Position threshold as %% of range (default: 5)')
    parser.add_argument('--timeout', type=int, default=500,
                        help='Timeout samples for held positions (default: 500)')
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
