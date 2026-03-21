"""
Visualize Segment-Classify Labels

Usage:
    python preprocessing/visualization/visualize_labels.py
    python preprocessing/visualization/visualize_labels.py --seed 42
    python preprocessing/visualization/visualize_labels.py --exp-path /path/to/exp
    python preprocessing/visualization/visualize_labels.py --energy-thresh 15 --merge-gap 80
"""

import numpy as np
import os
import sys
import argparse
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

from preprocessing.signal_utils import apply_joystick_filters
from preprocessing.label_logic.label_logic import label_movements_xy
from preprocessing.processor import DataProcessor

# Load label config
with open(os.path.join(project_dir, "preprocessing", "label_logic", "label_config.yaml")) as f:
    label_config = yaml.safe_load(f)

FILTERS = label_config.get('filters', {})
SC_DEFAULTS = label_config.get('segment_classify', {})

LABEL_COLORS = {
    0: 'rgba(128,128,128,0.3)', 1: 'rgba(0,200,0,0.4)',
    2: 'rgba(200,0,0,0.4)', 3: 'rgba(0,100,255,0.4)', 4: 'rgba(255,165,0,0.4)'
}
LABEL_NAMES = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}

classes_cfg = label_config.get('classes', {})
if classes_cfg.get('colors'):
    LABEL_COLORS = {int(k): v for k, v in classes_cfg['colors'].items()}
if classes_cfg.get('names'):
    LABEL_NAMES = {int(k): v for k, v in classes_cfg['names'].items()}


def visualize_labels(exp_path, config=None):
    if config is None:
        config = SC_DEFAULTS

    # Load and filter signals
    joystick = np.load(os.path.join(exp_path, "_joystick.npy"), allow_pickle=True)
    session = os.path.basename(os.path.dirname(exp_path))
    exp_name = f"{session}/{os.path.basename(exp_path)}"

    x_pos = apply_joystick_filters(joystick[:, 1].copy(), FILTERS, 'position')
    y_pos = apply_joystick_filters(joystick[:, 2].copy(), FILTERS, 'position')
    x_vel = apply_joystick_filters(np.gradient(x_pos), FILTERS, 'derivative')
    y_vel = apply_joystick_filters(np.gradient(y_pos), FILTERS, 'derivative')

    # Label
    labels, segments, params = label_movements_xy(x_pos, y_pos, x_vel, y_vel, config)

    # Build figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=[f"{exp_name} - X (LEFT/RIGHT)", f"{exp_name} - Y (UP/DOWN)"]
    )

    # Shaded label regions on both rows
    for label_val in [1, 2, 3, 4]:
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        color = LABEL_COLORS.get(label_val, 'rgba(128,128,128,0.3)')

        for s, e in zip(starts, ends):
            for row in [1, 2]:
                fig.add_shape(
                    type="rect", x0=s, x1=e, y0=0, y1=1,
                    xref="x" if row == 1 else "x2",
                    yref="y domain" if row == 1 else "y3 domain",
                    fillcolor=color, layer="below", line_width=0,
                )

        # Legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=15, color=color, symbol='square'),
            name=LABEL_NAMES.get(label_val, f'Class {label_val}'),
            showlegend=True
        ), row=1, col=1)

    # Plot each axis
    axes = [
        (1, 'X', x_pos, x_vel, segments['x'], params['x']),
        (2, 'Y', y_pos, y_vel, segments['y'], params['y']),
    ]
    for row, name, pos, vel, segs, prms in axes:
        is_first = row == 1
        x_vals = np.arange(len(pos))

        # Position + velocity traces
        fig.add_trace(go.Scatter(
            x=x_vals, y=pos, mode='lines', line=dict(color='blue', width=1.5),
            name='Position', legendgroup='pos', showlegend=is_first
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=x_vals, y=vel, mode='lines', line=dict(color='orange', width=1), opacity=0.7,
            name='Velocity', legendgroup='vel', showlegend=is_first
        ), row=row, col=1, secondary_y=True)

        # Activity threshold as position line
        pos_thresh = prms['activity_threshold']
        for y in [pos_thresh, -pos_thresh]:
            fig.add_hline(y=y, line=dict(color='blue', dash='dash', width=1),
                         opacity=0.5, row=row, col=1, secondary_y=False)

        # Segment markers: start, peak, stop
        if segs:
            s_idx = [s['start'] for s in segs]
            p_idx = [s['peak_idx'] for s in segs]
            e_idx = [s['stop'] - 1 for s in segs]

            fig.add_trace(go.Scatter(
                x=s_idx, y=pos[s_idx], mode='markers',
                marker=dict(symbol='triangle-right', size=12, color='green', line=dict(color='white', width=1)),
                name='Start', legendgroup='start', showlegend=is_first
            ), row=row, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                x=p_idx, y=pos[p_idx], mode='markers',
                marker=dict(symbol='star', size=14, color='purple', line=dict(color='white', width=1)),
                name='Peak', legendgroup='peak', showlegend=is_first
            ), row=row, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                x=e_idx, y=pos[e_idx], mode='markers',
                marker=dict(symbol='square', size=10, color='red', line=dict(color='white', width=1)),
                name='Stop', legendgroup='stop', showlegend=is_first
            ), row=row, col=1, secondary_y=False)

        # Axis labels
        fig.update_yaxes(title_text=f"Position {name}", secondary_y=False, row=row, col=1,
                        title_font=dict(color='blue'), tickfont=dict(color='blue'))
        fig.update_yaxes(title_text="Velocity", secondary_y=True, row=row, col=1,
                        title_font=dict(color='orange'), tickfont=dict(color='orange'))

    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_layout(
        title=f"Segment-Classify Labels - {exp_name} | {config}",
        height=800, width=1400, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Print stats
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Config: {config}")
    print(f"Distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    for ax in ['X', 'Y']:
        p = params[ax.lower()]
        print(f"  {ax}-axis: {len(segments[ax.lower()])} final segments, "
              f"{len(p['raw_segments'])} raw, {len(p['merged_segments'])} after merge")
    print(f"{'='*60}")

    return fig


def select_random_experiment(processor, seed=None):
    paths = processor.get_experiment_paths()
    if not paths:
        raise ValueError("No experiments found")
    return np.random.default_rng(seed).choice(paths)


def main():
    parser = argparse.ArgumentParser(description='Visualize segment-classify labeling')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--exp-path', '-e', type=str, default=None)
    parser.add_argument('--activity-thresh', type=float, default=None)
    parser.add_argument('--smooth-window', type=int, default=None)
    parser.add_argument('--min-duration', type=int, default=None)
    parser.add_argument('--merge-gap', type=int, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    config = dict(SC_DEFAULTS)
    if args.activity_thresh is not None:
        config['activity_threshold_percent'] = args.activity_thresh
    if args.smooth_window is not None:
        config['smooth_window'] = args.smooth_window
    if args.min_duration is not None:
        config['min_duration'] = args.min_duration
    if args.merge_gap is not None:
        config['merge_gap'] = args.merge_gap

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_dir, config_path)

    processor = DataProcessor(config_file=config_path, auto_run=False)
    exp_path = args.exp_path or select_random_experiment(processor, seed=args.seed)
    print(f"Selected experiment: {exp_path}")

    fig = visualize_labels(exp_path, config)

    if args.save:
        fig.write_html(args.save)
        print(f"\nSaved to: {args.save}")
    else:
        fig.show()


if __name__ == '__main__':
    exit(main())
