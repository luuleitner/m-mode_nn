"""
Processed Data Visualization with 5-Class Labels

Shows processed US channels with joystick position, derivative, and 5-class labels.
Layout: 6 rows (1 column) - For each of 3 US channels: X-axis row + Y-axis row

Each row contains:
- US channel heatmap (primary)
- Joystick position trace (secondary y-axis)
- Joystick derivative trace (secondary y-axis)
- 5-class label regions (colored rectangles, optional)

Colorscale Notes:
- Non-differentiated data: Grayscale, auto-scaled
- Differentiated data: Diverging RdBu_r (blue=negative, white=zero, red=positive)
  Uses 99th percentile for color range to enhance contrast by clipping outliers.
  Extreme values saturate to the colorscale limits rather than washing out the image.

Usage:
    python visualization/visualize_processed_RFlable.py
    python visualization/visualize_processed_RFlable.py --seed 42
    python visualization/visualize_processed_RFlable.py --exp-path /path/to/experiment
    python visualization/visualize_processed_RFlable.py --config config/config.yaml
    python visualization/visualize_processed_RFlable.py --no-labels  # hide label rectangles

Arguments:
    --config    : Path to config file (default: config/config.yaml)
    --seed      : Random seed for experiment selection
    --exp-path  : Specific experiment path (overrides random selection)
    --no-labels : Hide label region rectangles (labels shown by default)
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

# Load label config for joystick filters and class definitions
label_config_path = os.path.join(project_root, "preprocessing", "label_logic", "label_config.yaml")
with open(label_config_path, 'r') as f:
    label_config = yaml.safe_load(f)
filters_config = label_config.get('filters', {})

# Get class configuration
_classes_config = label_config.get('classes', {})
_INCLUDE_NOISE = _classes_config.get('include_noise', True)
_NUM_CLASSES = 5 if _INCLUDE_NOISE else 4

# Get class names from config
_config_names = _classes_config.get('names', {})
LABEL_NAMES = {int(k): v for k, v in _config_names.items()} if _config_names else {
    0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'
}

# Label colors (fill, border) - low alpha for transparency
# These are always defined for all 5 classes (visualization shows raw labels)
LABEL_COLORS = {
    0: ('rgba(255, 0, 255, 0.15)', 'rgba(255, 0, 255, 0.5)'),     # Noise - magenta
    1: ('rgba(0, 255, 0, 0.15)', 'rgba(0, 255, 0, 0.5)'),         # Up - green
    2: ('rgba(255, 0, 0, 0.15)', 'rgba(255, 0, 0, 0.5)'),         # Down - red
    3: ('rgba(0, 0, 255, 0.15)', 'rgba(0, 0, 255, 0.5)'),         # Left - blue
    4: ('rgba(255, 165, 0, 0.15)', 'rgba(255, 165, 0, 0.5)')      # Right - orange
}

# Active labels based on include_noise setting
ACTIVE_LABELS = list(range(5)) if _INCLUDE_NOISE else list(range(1, 5))


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
    # Find all unique labels in the data
    unique_labels = np.unique(labels)
    regions = {i: [] for i in unique_labels}

    for label_val in unique_labels:
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        for s, e in zip(starts, ends):
            regions[label_val].append((s, e))

    return regions


def add_label_regions(fig, labels, row, col, y_max, show_noise=True):
    """Add colored rectangles for label regions."""
    regions = find_label_regions(labels)

    for label_val in regions.keys():
        if label_val == 0 and not show_noise:
            continue  # Skip noise regions unless explicitly requested

        # Get colors with fallback for unknown labels
        fill_color, border_color = LABEL_COLORS.get(
            label_val, ('rgba(128, 128, 128, 0.15)', 'rgba(128, 128, 128, 0.5)')
        )

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


def create_visualization(exp_path, data, show_labels=True):
    """
    Create visualization with 6 rows (single column):
    - For each US channel (0, 1, 2):
      - Row: US heatmap + Joystick X + X derivative + labels
      - Row: US heatmap + Joystick Y + Y derivative + labels
    """
    processed_us = data['processed_us']  # (C, Depth, Pulses)
    joystick = data['joystick']  # (4, Pulses) - [trigger, x, y, button]
    labels = data['labels']
    config_info = data['config_info']

    n_channels = processed_us.shape[0]
    depth = processed_us.shape[1]
    n_pulses = processed_us.shape[2]

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
        deriv_x_scaled = deriv_x / deriv_scale * 50  # Scale to ±50 range
        deriv_y_scaled = deriv_y / deriv_scale * 50
    else:
        deriv_x_scaled = deriv_x
        deriv_y_scaled = deriv_y

    # Check if data is differentiated for title labeling
    is_diff = config_info.get('differentiation') is not None
    us_label = 'dUS/dt' if is_diff else 'US'

    # Build subplot titles
    subplot_titles = []
    for ch in range(n_channels):
        subplot_titles.append(f'{us_label} Ch{ch} + Joystick X + Labels [{depth}×{n_pulses}]')
        subplot_titles.append(f'{us_label} Ch{ch} + Joystick Y + Labels [{depth}×{n_pulses}]')

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

    # Determine colorscale based on whether differentiation was applied
    if is_diff:
        # Diverging colorscale for differentiated data (blue=negative, white=zero, red=positive)
        us_colorscale = 'RdBu_r'
        # Use 99th percentile for color range to enhance contrast by clipping outliers.
        # This prevents extreme values from compressing the colorscale and washing out
        # the visualization. Outlier values simply saturate to the colorscale limits.
        diff_max = np.percentile(np.abs(processed_us), 99)
        us_zmin, us_zmax, us_zmid = -diff_max, diff_max, 0
        colorbar_title = 'dUS/dt'
    else:
        # Grayscale for non-differentiated (RF/envelope) data
        us_colorscale = 'gray'
        us_zmin, us_zmax, us_zmid = None, None, None
        colorbar_title = 'Amp'

    for ch in range(n_channels):
        row_x = ch * 2 + 1  # X-axis row
        row_y = ch * 2 + 2  # Y-axis row

        # === X-AXIS ROW ===
        # US heatmap
        heatmap_kwargs = dict(
            z=processed_us[ch],
            colorscale=us_colorscale,
            showscale=False,
            name=f'US Ch{ch}'
        )
        if is_diff:
            heatmap_kwargs.update(zmin=us_zmin, zmax=us_zmax, zmid=us_zmid)
        fig.add_trace(go.Heatmap(**heatmap_kwargs), row=row_x, col=1)

        # Add label regions (respects include_noise setting)
        if show_labels:
            add_label_regions(fig, labels, row_x, 1, depth, show_noise=_INCLUDE_NOISE)

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
        # US heatmap
        heatmap_kwargs_y = dict(
            z=processed_us[ch],
            colorscale=us_colorscale,
            showscale=(ch == n_channels - 1),
            colorbar=dict(title=colorbar_title, x=1.02) if ch == n_channels - 1 else None,
            name=f'US Ch{ch} (Y)'
        )
        if is_diff:
            heatmap_kwargs_y.update(zmin=us_zmin, zmax=us_zmax, zmid=us_zmid)
        fig.add_trace(go.Heatmap(**heatmap_kwargs_y), row=row_y, col=1)

        # Add label regions (respects include_noise setting)
        if show_labels:
            add_label_regions(fig, labels, row_y, 1, depth, show_noise=_INCLUDE_NOISE)

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
    if config_info.get('differentiation'): processing_parts.append(f"Diff:{config_info['differentiation']}")
    if dec_factor > 1: processing_parts.append(f"Dec:÷{dec_factor}")

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

    # Label distribution - use unique labels from data, filtered by include_noise
    unique_labels = sorted(np.unique(labels))
    if not _INCLUDE_NOISE:
        unique_labels = [l for l in unique_labels if l != 0]  # Exclude noise from display
    label_counts = {i: np.sum(labels == i) for i in unique_labels}
    label_dist = " | ".join([f"{LABEL_NAMES.get(i, f'Class {i}')}: {label_counts[i]}" for i in unique_labels])

    # Build legend dynamically
    legend_parts = []
    label_colors_html = {0: 'gray', 1: 'green', 2: 'red', 3: 'blue', 4: 'orange'}
    for i in unique_labels:
        color = label_colors_html.get(i, 'gray')
        name = LABEL_NAMES.get(i, f'Class {i}')
        legend_parts.append(f"<span style='color:{color}'>■{name}</span>")

    # Info annotation
    info_lines = [
        f"<b>Session:</b> {session_name}  |  <b>Exp:</b> {exp_num}",
        f"<b>Processing:</b> {' → '.join(processing_parts)}",
        f"<b>Labels:</b> {config_info['label_method']} (pos={config_info.get('pp_pos_thresh', 'N/A')}%, deriv={config_info.get('pp_deriv_thresh', 'N/A')}%)",
        f"<b>Distribution:</b> {label_dist}",
        f"<b>Legend:</b> {' '.join(legend_parts)}"
    ]
    fig.add_annotation(
        text="<br>".join(info_lines),
        xref="paper", yref="paper",
        x=0.5, y=1.02,
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
    parser = argparse.ArgumentParser(description='Visualize processed data with 5-class labels')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for experiment selection')
    parser.add_argument('--exp-path', type=str, default=None,
                        help='Specific experiment path (overrides random selection)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Hide label region rectangles')
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

    # Label distribution (filtered by include_noise setting)
    unique_labels = sorted(np.unique(data['labels']))
    if not _INCLUDE_NOISE:
        unique_labels = [l for l in unique_labels if l != 0]
    print(f"\nLabel Distribution ({len(unique_labels)} active classes, include_noise={_INCLUDE_NOISE}):")
    for i in unique_labels:
        count = np.sum(data['labels'] == i)
        pct = 100.0 * count / len(data['labels'])
        name = LABEL_NAMES.get(i, f'Class {i}')
        print(f"  {name:6s}: {count:5d} ({pct:5.1f}%)")

    # Create and show visualization
    fig = create_visualization(exp_path, data, show_labels=not args.no_labels)
    fig.show()

    print("\nVisualization opened in browser")


if __name__ == '__main__':
    main()
