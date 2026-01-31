"""
Processed Data Visualization - DIFFERENTIATED Signal (Temporal Gradient)

Shows the temporal derivative (diff along pulse axis) of processed US channels.
This reveals CHANGES in the signal over time, which may be more discriminative
for movement classification than absolute values.

Layout: 6 rows (1 column) - For each of 3 US channels: X-axis row + Y-axis row

Each row contains:
- US channel DERIVATIVE heatmap (dUS/dt along pulse axis)
- Joystick position trace (secondary y-axis)
- Joystick derivative trace (secondary y-axis)
- 5-class label regions (colored rectangles)

Usage:
    python visualization/visualize_processed_diff.py
    python visualization/visualize_processed_diff.py --seed 42
    python visualization/visualize_processed_diff.py --exp-path /path/to/experiment
    python visualization/visualize_processed_diff.py --config config/config.yaml
"""

import os
import sys
import argparse
import numpy as np
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from preprocessing.processor import DataProcessor
from preprocessing.signal_utils import apply_joystick_filters

# Load label config for joystick filters
label_config_path = os.path.join(project_root, "preprocessing", "label_logic", "label_config.yaml")
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)
filters_config = label_config.get('filters', {})


# 5-class label colors (fill, border) - low alpha for transparency
LABEL_COLORS = {
    0: ('rgba(255, 0, 255, 0.15)', 'rgba(255, 0, 255, 0.5)'),     # Noise - magenta
    1: ('rgba(0, 255, 0, 0.15)', 'rgba(0, 255, 0, 0.5)'),         # Up - green
    2: ('rgba(255, 0, 0, 0.15)', 'rgba(255, 0, 0, 0.5)'),         # Down - red
    3: ('rgba(0, 0, 255, 0.15)', 'rgba(0, 0, 255, 0.5)'),         # Left - blue
    4: ('rgba(255, 165, 0, 0.15)', 'rgba(255, 165, 0, 0.5)')      # Right - orange
}

LABEL_NAMES = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}


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
    regions = {i: [] for i in range(5)}  # 5 classes: 0-4

    for label_val in range(5):
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        for s, e in zip(starts, ends):
            regions[label_val].append((s, e))

    return regions


def add_label_regions(fig, labels, row, col, y_max, show_noise=True):
    """Add colored rectangles for 5-class label regions."""
    regions = find_label_regions(labels)

    for label_val in range(5):
        if label_val == 0 and not show_noise:
            continue  # Skip noise regions unless explicitly requested

        fill_color, border_color = LABEL_COLORS[label_val]

        for s, e in regions[label_val]:
            fig.add_shape(
                type="rect",
                x0=s, x1=e + 1,
                y0=0, y1=y_max,
                fillcolor=fill_color,
                line=dict(color=border_color, width=1),
                layer="above",
                row=row, col=col
            )


def create_visualization(exp_path, data):
    """
    Create visualization with 6 rows (single column):
    - For each US channel (0, 1, 2):
      - Row: US DERIVATIVE heatmap + Joystick X + X derivative + labels
      - Row: US DERIVATIVE heatmap + Joystick Y + Y derivative + labels
    """
    processed_us = data['processed_us']  # (C, Depth, Pulses)
    joystick = data['joystick']  # (4, Pulses) - [trigger, x, y, button]
    labels = data['labels']
    config_info = data['config_info']

    n_channels = processed_us.shape[0]
    depth = processed_us.shape[1]
    n_pulses = processed_us.shape[2]

    # ========================================
    # COMPUTE TEMPORAL DERIVATIVE OF US SIGNAL
    # ========================================
    # Gradient along pulse axis (axis=2) shows temporal changes
    us_diff = np.gradient(processed_us, axis=2)

    # Also compute absolute derivative for comparison
    us_diff_abs = np.abs(us_diff)

    print(f"US diff shape: {us_diff.shape}")
    print(f"US diff range: [{us_diff.min():.4f}, {us_diff.max():.4f}]")
    print(f"US diff abs mean per class:")
    for cls in range(5):
        mask = labels == cls
        if mask.sum() > 0:
            cls_mean = us_diff_abs[:, :, mask].mean()
            print(f"  {LABEL_NAMES[cls]}: {cls_mean:.4f}")

    # Extract and filter joystick data (same as visualize_labels.py)
    x_raw = joystick[1, :]  # X axis
    y_raw = joystick[2, :]  # Y axis

    # Apply position filters from label_config.yaml
    joy_x = apply_joystick_filters(x_raw.copy(), filters_config, 'position')
    joy_y = apply_joystick_filters(y_raw.copy(), filters_config, 'position')

    # Compute and filter derivatives (same as visualize_labels.py)
    deriv_x = apply_joystick_filters(np.gradient(joy_x), filters_config, 'derivative')
    deriv_y = apply_joystick_filters(np.gradient(joy_y), filters_config, 'derivative')

    # Normalize derivatives for visualization (scale to similar range as position)
    deriv_scale = max(np.abs(deriv_x).max(), np.abs(deriv_y).max())
    if deriv_scale > 0:
        deriv_x_scaled = deriv_x / deriv_scale * 50  # Scale to +/-50 range
        deriv_y_scaled = deriv_y / deriv_scale * 50
    else:
        deriv_x_scaled = deriv_x
        deriv_y_scaled = deriv_y

    # Build subplot titles
    subplot_titles = []
    for ch in range(n_channels):
        subplot_titles.append(f'dUS/dt Ch{ch} + Joystick X [{depth}x{n_pulses}]')
        subplot_titles.append(f'dUS/dt Ch{ch} + Joystick Y [{depth}x{n_pulses}]')

    # Create figure: 6 rows (2 per channel), 1 column, with secondary y-axis
    n_rows = n_channels * 2
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        row_heights=[1] * n_rows,
        specs=[[{"secondary_y": True}] for _ in range(n_rows)]
    )

    # Get session/experiment info for title
    session_name = os.path.basename(os.path.dirname(exp_path))
    exp_num = os.path.basename(exp_path)

    # Use diverging colorscale for derivative (negative=blue, zero=white, positive=red)
    diff_colorscale = 'RdBu_r'  # Red-Blue reversed (red=positive, blue=negative)

    # Compute symmetric color range for derivative
    diff_max = np.percentile(np.abs(us_diff), 99)  # Use 99th percentile to avoid outliers

    for ch in range(n_channels):
        row_x = ch * 2 + 1  # X-axis row
        row_y = ch * 2 + 2  # Y-axis row

        # === X-AXIS ROW ===
        # US DERIVATIVE heatmap (temporal gradient)
        fig.add_trace(
            go.Heatmap(
                z=us_diff[ch],
                colorscale=diff_colorscale,
                zmid=0,  # Center colorscale at zero
                zmin=-diff_max,
                zmax=diff_max,
                showscale=False,
                name=f'dUS/dt Ch{ch}'
            ),
            row=row_x, col=1
        )

        # Add 5-class label regions
        add_label_regions(fig, labels, row_x, 1, depth)

        # Joystick X position (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_pulses),
                y=joy_x,
                mode='lines',
                line=dict(color='cyan', width=2),
                name='Joy X',
                legendgroup='joyx',
                showlegend=(ch == 0)
            ),
            row=row_x, col=1, secondary_y=True
        )

        # X derivative (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_pulses),
                y=deriv_x_scaled,
                mode='lines',
                line=dict(color='yellow', width=1.5, dash='dot'),
                name='dX/dt',
                legendgroup='derivx',
                showlegend=(ch == 0)
            ),
            row=row_x, col=1, secondary_y=True
        )

        # Configure axes for X row
        fig.update_yaxes(autorange='reversed', title_text='Depth', row=row_x, col=1, secondary_y=False)
        fig.update_yaxes(title_text='Joy X', row=row_x, col=1, secondary_y=True)
        fig.update_xaxes(title_text='Pulses', range=[0, n_pulses], autorange=False, row=row_x, col=1)

        # === Y-AXIS ROW ===
        # US DERIVATIVE heatmap
        fig.add_trace(
            go.Heatmap(
                z=us_diff[ch],
                colorscale=diff_colorscale,
                zmid=0,
                zmin=-diff_max,
                zmax=diff_max,
                showscale=(ch == n_channels - 1),
                colorbar=dict(title='dUS/dt', x=1.02) if ch == n_channels - 1 else None,
                name=f'dUS/dt Ch{ch} (Y)'
            ),
            row=row_y, col=1
        )

        # Add 5-class label regions
        add_label_regions(fig, labels, row_y, 1, depth)

        # Joystick Y position (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_pulses),
                y=joy_y,
                mode='lines',
                line=dict(color='lime', width=2),
                name='Joy Y',
                legendgroup='joyy',
                showlegend=(ch == 0)
            ),
            row=row_y, col=1, secondary_y=True
        )

        # Y derivative (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_pulses),
                y=deriv_y_scaled,
                mode='lines',
                line=dict(color='orange', width=1.5, dash='dot'),
                name='dY/dt',
                legendgroup='derivy',
                showlegend=(ch == 0)
            ),
            row=row_y, col=1, secondary_y=True
        )

        # Configure axes for Y row
        fig.update_yaxes(autorange='reversed', title_text='Depth', row=row_y, col=1, secondary_y=False)
        fig.update_yaxes(title_text='Joy Y', row=row_y, col=1, secondary_y=True)
        fig.update_xaxes(title_text='Pulses', range=[0, n_pulses], autorange=False, row=row_y, col=1)

    # Build processing info
    dec_factor = config_info['decimation_factor'] or 1
    processing_parts = []
    if config_info['bandpass']: processing_parts.append('BP')
    if config_info['tgc']: processing_parts.append('TGC')
    if config_info['clip']: processing_parts.append('Clip')
    if config_info['envelope']: processing_parts.append('Env')
    if config_info['logcompression']: processing_parts.append('Log')
    if config_info['normalization']: processing_parts.append(f"Norm:{config_info['normalization']}")
    if dec_factor > 1: processing_parts.append(f"Dec:{dec_factor}")
    processing_parts.append('DIFF')  # Indicate differentiation

    # Layout
    fig.update_layout(
        height=250 * n_rows,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.06,
            xanchor='left',
            x=0.0,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(t=160, b=50)
    )

    # Label distribution
    label_counts = {i: np.sum(labels == i) for i in range(5)}
    label_dist = " | ".join([f"{LABEL_NAMES[i]}: {label_counts[i]}" for i in range(5)])

    # Info annotation
    info_lines = [
        f"<b>Session:</b> {session_name}  |  <b>Exp:</b> {exp_num}  |  <b style='color:red'>DIFFERENTIATED SIGNAL (dUS/dt)</b>",
        f"<b>Processing:</b> {' -> '.join(processing_parts)}",
        f"<b>Labels:</b> {config_info['label_method']} (pos={config_info.get('pp_pos_thresh', 'N/A')}%, deriv={config_info.get('pp_deriv_thresh', 'N/A')}%)",
        f"<b>Distribution:</b> {label_dist}",
        f"<b>Legend:</b> <span style='color:magenta'>Noise</span> <span style='color:green'>Up</span> <span style='color:red'>Down</span> <span style='color:blue'>Left</span> <span style='color:orange'>Right</span>",
        f"<b>Colorscale:</b> Blue=negative change, White=no change, Red=positive change"
    ]
    fig.add_annotation(
        text="<br>".join(info_lines),
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=11),
        align="center",
        bgcolor="rgba(255,240,240,0.9)",  # Light red background to indicate diff view
        bordercolor="red",
        borderwidth=1,
        borderpad=8
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize DIFFERENTIATED processed data with 5-class labels')
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

    # Initialize processor
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

    proc_shape = data['processed_us'].shape
    print(f"\nProcessed US shape: {proc_shape} [C, depth, pulses]")
    print(f"Joystick shape:     {data['joystick'].shape} [channels, pulses]")
    print(f"Labels shape:       {data['labels'].shape}")

    # Label distribution
    print(f"\n5-Class Label Distribution:")
    for i in range(5):
        count = np.sum(data['labels'] == i)
        pct = 100.0 * count / len(data['labels'])
        print(f"  {LABEL_NAMES[i]:6s}: {count:5d} ({pct:5.1f}%)")

    # Create and show visualization
    print("\nCreating DIFFERENTIATED signal visualization...")
    fig = create_visualization(exp_path, data)
    fig.show()

    print("\nVisualization opened in browser")
    print("NOTE: This shows dUS/dt (temporal derivative)")
    print("      Blue = signal decreasing, Red = signal increasing, White = stable")


if __name__ == '__main__':
    main()
