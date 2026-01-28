"""
Data Visualization - Verify config settings by visualizing raw vs processed data.

Uses Plotly for interactive zooming. Displays:
- M-mode heatmaps with joystick overlay and label regions
- Raw vs Processed comparison for all 3 US channels

Usage:
    python visualization/data_visualization.py
    python visualization/data_visualization.py --seed 42
    python visualization/data_visualization.py --exp-path /path/to/session/experiment
"""

import os
import sys
import argparse
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from preprocessing.processor import DataProcessor


def select_random_experiment(processor, seed=None):
    """Select a random experiment from available paths."""
    paths = processor.get_experiment_paths()
    if not paths:
        raise ValueError("No experiments found with current config")

    rng = np.random.default_rng(seed)
    selected = rng.choice(paths)
    return selected


def find_label_regions(labels):
    """Find start and end indices for each labeled region."""
    regions = {1: [], 2: []}  # {label_val: [(start, end), ...]}

    for label_val in [1, 2]:
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1  # -1 to get last index of region

        for s, e in zip(starts, ends):
            regions[label_val].append((s, e))

    return regions


def add_label_regions(fig, labels, row, col, x_range=None):
    """Add colored background regions for labeled areas."""
    label_colors = {
        1: 'rgba(0, 255, 0, 0.15)',   # Green for up
        2: 'rgba(255, 0, 0, 0.15)'    # Red for down
    }

    x_offset = x_range[0] if x_range is not None else 0
    regions = find_label_regions(labels)

    for label_val in [1, 2]:
        for s, e in regions[label_val]:
            fig.add_vrect(
                x0=s + x_offset, x1=e + 1 + x_offset,  # +1 because vrect is exclusive
                fillcolor=label_colors[label_val],
                layer="below", line_width=0,
                row=row, col=col
            )




def create_visualization(exp_path, data):
    """
    Create interactive Plotly visualization.

    Layout: 3 US channels × 2 rows (raw/processed) = 6 rows
    Each row: M-mode heatmap with joystick X/Y overlay + label regions
    """
    raw_us = data['raw_us']
    processed_us = data['processed_us']
    joystick = data['joystick']
    labels = data['labels']
    markers = data['markers']
    config_info = data['config_info']

    n_channels = raw_us.shape[0]

    # Extract joystick positions and trigger
    x_pos = joystick[1, :]
    y_pos = joystick[2, :]
    trigger = joystick[3, :]

    # Normalize joystick to fit on M-mode (scale to depth range)
    raw_depth = raw_us.shape[1]
    proc_depth = processed_us.shape[1]

    def normalize_joystick(pos, depth):
        """Normalize joystick position to M-mode depth range."""
        pos_min, pos_max = pos.min(), pos.max()
        if pos_max - pos_min > 0:
            return depth * 0.1 + (pos - pos_min) / (pos_max - pos_min) * (depth * 0.3)
        return np.full_like(pos, depth * 0.2)

    # Get decimation factor
    dec_factor = config_info['decimation_factor'] or 1

    # Build subplot titles with actual dimensions
    subplot_titles = []
    for ch in range(n_channels):
        subplot_titles.append(f'US Ch{ch} - RAW [{raw_depth} depth × {raw_us.shape[2]} pulses]')
        subplot_titles.append(f'US Ch{ch} - PROCESSED [{proc_depth} depth × {processed_us.shape[2]} pulses] (depth decimated ÷{dec_factor})')

    # Create figure: n_channels * 2 rows, 1 column
    fig = make_subplots(
        rows=n_channels * 2, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        row_heights=[1] * (n_channels * 2)
    )

    # Build title
    exp_name = os.path.basename(os.path.dirname(exp_path)) + "/" + os.path.basename(exp_path)
    processing_parts = []
    if config_info['bandpass']: processing_parts.append('BP')
    if config_info['tgc']: processing_parts.append('TGC')
    if config_info['clip']: processing_parts.append('Clip')
    if config_info['envelope']: processing_parts.append('Env')
    if config_info['logcompression']: processing_parts.append('Log')
    if config_info['normalization']: processing_parts.append(f"Norm:{config_info['normalization']}")
    if dec_factor > 1: processing_parts.append(f"Dec:÷{dec_factor}")

    title_text = (
        f"<b>{exp_name}</b>"
    )

    for ch in range(n_channels):
        row_raw = ch * 2 + 1
        row_proc = ch * 2 + 2

        # === RAW ROW ===
        # M-mode heatmap
        fig.add_trace(
            go.Heatmap(
                z=raw_us[ch],
                colorscale='gray',
                showscale=False,
                name=f'US Ch{ch} Raw'
            ),
            row=row_raw, col=1
        )

        # Add label regions
        add_label_regions(fig, labels, row_raw, 1)

        # Joystick X overlay (scaled to depth)
        joy_x_scaled = normalize_joystick(x_pos, raw_depth)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(x_pos)),
                y=joy_x_scaled,
                mode='lines',
                line=dict(color='cyan', width=1.5),
                name='Joystick X',
                legendgroup='joy_x',
                showlegend=(ch == 0)
            ),
            row=row_raw, col=1
        )

        # Joystick Y overlay
        joy_y_scaled = normalize_joystick(y_pos, raw_depth)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(y_pos)),
                y=joy_y_scaled,
                mode='lines',
                line=dict(color='yellow', width=1.5),
                name='Joystick Y',
                legendgroup='joy_y',
                showlegend=(ch == 0)
            ),
            row=row_raw, col=1
        )

        # Trigger overlay (scaled to bottom portion of depth)
        trig_scaled = normalize_joystick(trigger, raw_depth) + raw_depth * 0.4
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(trigger)),
                y=trig_scaled,
                mode='lines',
                line=dict(color='magenta', width=1.5),
                name='Trigger',
                legendgroup='trigger',
                showlegend=(ch == 0)
            ),
            row=row_raw, col=1
        )

        fig.update_yaxes(autorange='reversed', title_text='Depth', row=row_raw, col=1)
        fig.update_xaxes(title_text='Pulses', row=row_raw, col=1)

        # === PROCESSED ROW ===
        # M-mode heatmap
        fig.add_trace(
            go.Heatmap(
                z=processed_us[ch],
                colorscale='gray',
                showscale=(ch == n_channels - 1),
                colorbar=dict(title='Amp', x=1.02) if ch == n_channels - 1 else None,
                name=f'US Ch{ch} Proc'
            ),
            row=row_proc, col=1
        )

        # NOTE: Decimation is on DEPTH axis, not PULSES axis
        # So joystick and labels have the same pulse count as processed US
        add_label_regions(fig, labels, row_proc, 1)

        # Joystick overlays (same pulse count, scaled to processed depth)
        joy_x_proc = normalize_joystick(x_pos, proc_depth)
        joy_y_proc = normalize_joystick(y_pos, proc_depth)

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(x_pos)),
                y=joy_x_proc,
                mode='lines',
                line=dict(color='cyan', width=1.5),
                name='Joystick X (proc)',
                legendgroup='joy_x',
                showlegend=False
            ),
            row=row_proc, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(y_pos)),
                y=joy_y_proc,
                mode='lines',
                line=dict(color='yellow', width=1.5),
                name='Joystick Y (proc)',
                legendgroup='joy_y',
                showlegend=False
            ),
            row=row_proc, col=1
        )

        # Trigger overlay (scaled to bottom portion)
        trig_proc = normalize_joystick(trigger, proc_depth) + proc_depth * 0.4
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(trigger)),
                y=trig_proc,
                mode='lines',
                line=dict(color='magenta', width=1.5),
                name='Trigger (proc)',
                legendgroup='trigger',
                showlegend=False
            ),
            row=row_proc, col=1
        )

        fig.update_yaxes(autorange='reversed', title_text=f'Depth (÷{dec_factor})' if dec_factor > 1 else 'Depth', row=row_proc, col=1)
        fig.update_xaxes(title_text='Pulses', row=row_proc, col=1)

    # Layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        height=300 * n_channels * 2,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.12,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(t=150, b=50)
    )

    # Add info annotation box at the top
    info_lines = [
        f"<b>Processing:</b> {' → '.join(processing_parts)}",
        f"<b>Labels:</b> {config_info['label_method']} (axis={config_info['label_axis']}, thresh={config_info['label_threshold']}%)",
        f"<b>Shapes:</b> Raw {raw_us.shape} → Proc {processed_us.shape} | Joy {joystick.shape} | Labels {labels.shape}"
    ]
    fig.add_annotation(
        text="<br>".join(info_lines),
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=11),
        align="center",
        bgcolor="rgba(240,240,240,0.9)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=8
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessing config verification')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', type=str, default=None,
                        help='Specific experiment path (overrides random selection)')
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    print(f"Loading config: {config_path}")

    # Initialize processor without running full pipeline
    processor = DataProcessor(config_file=config_path, auto_run=False)

    # Select experiment
    if args.exp_path:
        exp_path = args.exp_path
    else:
        exp_path = select_random_experiment(processor, seed=args.seed)

    print(f"Selected experiment: {exp_path}")

    # Process single experiment
    print("Processing experiment...")
    data = processor.process_single_experiment(exp_path)

    raw_shape = data['raw_us'].shape
    proc_shape = data['processed_us'].shape
    dec_factor = data['config_info']['decimation_factor'] or 1

    print(f"\nData shapes:")
    print(f"  Raw US:       {raw_shape} [C, depth, pulses]")
    print(f"  Processed US: {proc_shape} [C, depth, pulses]")
    print(f"  Joystick:     {data['joystick'].shape} [channels, pulses]")
    print(f"  Labels:       {data['labels'].shape}")
    print(f"\nDecimation (on DEPTH axis): factor={dec_factor}")
    print(f"  Depth: {raw_shape[1]} → {proc_shape[1]} samples")
    print(f"  Pulses: {raw_shape[2]} → {proc_shape[2]} (unchanged)")
    print(f"\nLabel distribution: 0={np.sum(data['labels']==0)}, 1={np.sum(data['labels']==1)}, 2={np.sum(data['labels']==2)}")

    # Create and show visualization
    fig = create_visualization(exp_path, data)
    fig.show()

    print("Visualization opened in browser")


if __name__ == '__main__':
    main()
