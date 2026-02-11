import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml

from matplotlib import pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from preprocessing.processor import DataProcessor
from preprocessing.signal_utils import apply_joystick_filters

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



config_path = os.path.join('/config/config.yaml')
processor = DataProcessor(config_file=config_path, auto_run=False)
exp_path = '/scratch/US_EMG_preprocessing/session14_exp0'
USdata = processor.process_single_experiment(exp_path)

from include.emg.emg_processing.filter import Butter
from scipy.signal import iirnotch, butter, filtfilt
from scipy.signal import resample, spectrogram

fs_us = 175
fs_emg = 2048


EMGdata_path = '/scratch/US_EMG_preprocessing/session14_exp0/exp1.csv'
EMGtrigger = pd.read_csv(EMGdata_path).iloc[:,5]
EMGdata = pd.read_csv(EMGdata_path).iloc[:,2:5]
columns = ['ch1', 'ch2', 'ch3']
EMGdata.columns = columns

EMGdata = Butter(EMGdata, 20, 380, fs_emg, 4).data_filtered

b_notch, a_notch = iirnotch(50, 30, fs_emg)
EMGdata_notched = filtfilt(b_notch, a_notch, EMGdata, axis=0)

# Rectify + LP filter at 10 Hz + moving average (linear envelope)
EMGdata_rect = np.abs(EMGdata_notched)
b_lp, a_lp = butter(4, 10, btype='low', fs=fs_emg)
EMGdata_lp = filtfilt(b_lp, a_lp, EMGdata_rect, axis=0)

ksize = 100
kernel = np.ones(ksize) / ksize
EMGdata_env = np.column_stack([np.convolve(EMGdata_lp[:, ch], kernel, mode='same') for ch in range(EMGdata_lp.shape[1])])


# --- Synchronize EMG and US using trigger-count alignment ---
joystick_trigger = USdata["joystick"][3, :]  # US trigger at fs_us
emg_trigger = EMGtrigger.values.astype(float)  # EMG trigger at fs_emg
n_us_pulses = USdata["processed_us"].shape[2]

# Step 1: Binarize triggers
joy_trig_bin = (joystick_trigger > 0.5).astype(int)
emg_trig_bin = (emg_trigger > 10000).astype(int)

joy_rising = np.where(np.diff(joy_trig_bin) == 1)[0]
emg_rising = np.where(np.diff(emg_trig_bin) == 1)[0]

# Step 3: Align at first rising edge — clip everything before it
us_start = joy_rising[0]
emg_start = emg_rising[0]

joy_first_time = us_start / fs_us
emg_first_time = emg_start / fs_emg

# Recompute rising edges relative to clipped start
joy_rising_clipped = joy_rising[joy_rising >= us_start] - us_start
emg_rising_clipped = emg_rising[emg_rising >= emg_start] - emg_start

# Step 4: Determine minimum trigger count across both signals
min_triggers = min(len(joy_rising_clipped), len(emg_rising_clipped))
print(f"Triggers after alignment — Joystick: {len(joy_rising_clipped)}, "
      f"EMG: {len(emg_rising_clipped)}, using min: {min_triggers}")

# Step 5: Clip both signals at the last shared trigger
# End index = sample of the min_triggers-th rising edge (0-indexed, so index min_triggers-1)
us_end_rel = joy_rising_clipped[min_triggers - 1] + 1   # +1 to include last trigger sample
emg_end_rel = emg_rising_clipped[min_triggers - 1] + 1

us_end = us_start + us_end_rel
emg_end = emg_start + emg_end_rel

# Crop signals
EMGdata_env = EMGdata_env[emg_start:emg_end, :]
EMGdata_notched = EMGdata_notched[emg_start:emg_end, :]
emg_trigger_cropped = emg_trigger[emg_start:emg_end]
USdata["processed_us"] = USdata["processed_us"][:, :, us_start:us_end]
USdata["joystick"] = USdata["joystick"][:, us_start:us_end]
USdata["labels"] = USdata["labels"][us_start:us_end]

print(f"Cropped — US: {USdata['processed_us'].shape[2]} pulses "
      f"({us_end_rel / fs_us:.2f}s), EMG: {EMGdata_env.shape[0]} samples "
      f"({emg_end_rel / fs_emg:.2f}s)")

# Step 6: Keep native-rate EMG for plotting, resample a copy for model use
EMGdata_native = EMGdata_env.copy()  # native 2048 Hz for plotting
n_us_pulses = USdata["processed_us"].shape[2]
EMGdata_resampled = resample(EMGdata_env, n_us_pulses, axis=0)
# Nearest-neighbor resample for binary trigger (no FFT ringing)
emg_trig_indices = np.round(np.linspace(0, len(emg_trigger_cropped) - 1, n_us_pulses)).astype(int)
EMGtrigger_resampled = (emg_trigger_cropped[emg_trig_indices] > 10000).astype(float)  # binarize to 0-1
joystick_trigger = USdata["joystick"][3, :]

print(f"Final — US: {n_us_pulses} pulses, EMG native: {EMGdata_native.shape[0]} samples, "
      f"EMG resampled: {EMGdata_resampled.shape[0]} samples, "
      f"spanning {min_triggers} trigger cycles")

# Debug: plot aligned triggers with rising edges marked (exactly min_triggers each)
joy_edges_plot = joy_rising_clipped[:min_triggers]
emg_edges_plot = emg_rising_clipped[:min_triggers]

fig_dbg, (ax_dbg1, ax_dbg2) = plt.subplots(2, 1, figsize=(14, 4))
t_emg = np.arange(len(emg_trigger_cropped)) / fs_emg
ax_dbg1.plot(t_emg, emg_trigger_cropped, color='yellow', linewidth=0.8)
ax_dbg1.plot(emg_edges_plot / fs_emg,
             emg_trigger_cropped[emg_edges_plot], 'rv', markersize=6)
ax_dbg1.set_title(f'EMG Trigger (aligned, {min_triggers} triggers marked)')
ax_dbg1.set_ylabel('Value')
ax_dbg1.set_xlabel('Time [s]')

t_us = np.arange(USdata["joystick"].shape[1]) / fs_us
ax_dbg2.plot(t_us, USdata["joystick"][3, :], color='red', linewidth=0.8)
ax_dbg2.plot(joy_edges_plot / fs_us,
             USdata["joystick"][3, joy_edges_plot], 'rv', markersize=6)
ax_dbg2.set_title(f'Joystick Trigger (aligned, {min_triggers} triggers marked)')
ax_dbg2.set_ylabel('Value')
ax_dbg2.set_xlabel('Time [s]')
plt.tight_layout()
plt.show()


# --- Combined US + EMG plot (3 subplots, US heatmap + EMG overlay per channel) ---
processed_us = USdata["processed_us"]  # (C, Depth, Pulses)
labels = USdata["labels"]
time_ax = np.arange(n_us_pulses) / fs_us

colormap = 'grey'

# Matplotlib-compatible label colors
LABEL_COLORS_MPL = {
    0: (1, 0, 1, 0.15),        # Noise - magenta
    1: (0, 1, 0, 0.15),        # Up - green
    2: (1, 0, 0, 0.15),        # Down - red
    3: (0, 0, 1, 0.15),        # Left - blue
    4: (1, 0.647, 0, 0.15),    # Right - orange
}
label_regions = find_label_regions(labels)

fig_combined, axes_c = plt.subplots(3, 1, figsize=(14, 10))
for ch in range(3):
    # US M-mode heatmap
    axes_c[ch].imshow(processed_us[ch], cmap=colormap, aspect='auto',
                       extent=[0, time_ax[-1], processed_us.shape[1], 0])
    axes_c[ch].set_ylabel('Depth')
    axes_c[ch].set_title(f'Ch{ch+1}')

    # Shaded label regions
    for lv, regions in label_regions.items():
        if lv == 0 and not _INCLUDE_NOISE:
            continue
        color = LABEL_COLORS_MPL.get(lv, (0.5, 0.5, 0.5, 0.15))
        for i, (s, e) in enumerate(regions):
            lbl = LABEL_NAMES.get(lv, f'Class {lv}') if (ch == 0 and i == 0) else None
            axes_c[ch].axvspan(s / fs_us, (e + 1) / fs_us, color=color, label=lbl)

    # EMG overlay on secondary y-axis (native 2048 Hz, own time axis)
    ax_emg = axes_c[ch].twinx()
    t_emg_native = np.linspace(0, time_ax[-1], EMGdata_native.shape[0])
    ax_emg.plot(t_emg_native, EMGdata_native[:, ch], color='cyan', alpha=0.6, linewidth=0.7, label='EMG')
    ylim = max(abs(EMGdata_native[:, ch].min()), abs(EMGdata_native[:, ch].max()))
    ax_emg.set_ylim(-ylim * 2, ylim * 2)
    ax_emg.set_ylabel('EMG [mV]', color='cyan')
    ax_emg.tick_params(axis='y', labelcolor='cyan')

    # Joystick X/Y on third y-axis
    ax_joy = axes_c[ch].twinx()
    ax_joy.spines['right'].set_position(('axes', 1.08))
    ax_joy.plot(time_ax, USdata["joystick"][1, :], color='lime', alpha=0.7, linewidth=0.8, label='Joy X')
    ax_joy.plot(time_ax, USdata["joystick"][2, :], color='orange', alpha=0.7, linewidth=0.8, label='Joy Y')
    ax_joy.set_ylabel('Joystick', color='lime')
    ax_joy.tick_params(axis='y', labelcolor='lime')

    # Trigger on fourth y-axis
    ax_trig = axes_c[ch].twinx()
    ax_trig.spines['right'].set_position(('axes', 1.16))
    ax_trig.plot(time_ax, joystick_trigger, color='red', alpha=0.7, linewidth=0.8, label='Joy Trigger')
    ax_trig.plot(time_ax, EMGtrigger_resampled, color='yellow', alpha=0.7, linewidth=0.8, label='EMG Trigger')
    ax_trig.set_ylim(0, 1)
    ax_trig.set_ylabel('Trigger', color='red')
    ax_trig.tick_params(axis='y', labelcolor='red')
    if ch == 0:
        lines_lbl, labels_lbl = axes_c[ch].get_legend_handles_labels()
        lines_emg, labels_emg = ax_emg.get_legend_handles_labels()
        lines_joy, labels_joy = ax_joy.get_legend_handles_labels()
        lines_trig, labels_trig = ax_trig.get_legend_handles_labels()
        ax_trig.legend(lines_lbl + lines_emg + lines_joy + lines_trig,
                       labels_lbl + labels_emg + labels_joy + labels_trig, loc='upper right', fontsize=8)

    axes_c[ch].set_xlim(0, time_ax[-1])

axes_c[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('US&EMG_labels.png')
plt.show()



# --- Spectrogram: Original (top) vs Per-Frequency Normalized (bottom) ---
# Both signals are cropped to the same trigger window, verify durations match
emg_duration = EMGdata_notched.shape[0] / fs_emg
us_duration = n_us_pulses / fs_us
print(f"Duration check — EMG: {emg_duration:.3f}s, US: {us_duration:.3f}s, "
      f"ratio: {emg_duration/us_duration:.6f}")

# Solid border colors for label lines on spectrogram (high contrast)
LABEL_LINE_COLORS = {
    0: (1, 0, 1, 0.8),        # Noise - magenta
    1: (0, 1, 0, 0.8),        # Up - green
    2: (1, 0, 0, 0.8),        # Down - red
    3: (0, 0.5, 1, 0.8),      # Left - bright blue
    4: (1, 0.647, 0, 0.8),    # Right - orange
}

fig_spec, axes_spec = plt.subplots(3, 2, figsize=(18, 10))
colormap = 'inferno'
for ch in range(3):
    # Compute spectrogram, then rescale time axis to US time frame
    # (same approach as combined plot: map EMG duration onto US duration)
    f, t_raw, Sxx = spectrogram(EMGdata_notched[:, ch], fs=fs_emg, nperseg=256, noverlap=192)
    t = t_raw * (us_duration / emg_duration)  # rescale to US time frame

    # --- Left column: Original spectrogram with percentile-clipped colormap ---
    # Standard PSD in dB, but colormap range clipped to [5th, 95th] percentile
    # to prevent extreme low-frequency power from compressing the colorscale.
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    vmin, vmax = np.percentile(Sxx_dB, [5, 95])
    im_orig = axes_spec[ch, 0].pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap=colormap,
                                           vmin=vmin, vmax=vmax)
    axes_spec[ch, 0].set_ylabel('Freq [Hz]')
    axes_spec[ch, 0].set_title(f'Ch{ch+1} — Original (percentile-clipped)')

    # # Label regions: shaded fills
    # for lv, regions in label_regions.items():
    #     if lv == 0 and not _INCLUDE_NOISE:
    #         continue
    #     fill = LABEL_COLORS_MPL.get(lv, (0.5, 0.5, 0.5, 0.15))
    #     for s, e in regions:
    #         axes_spec[ch, 0].axvspan(s / fs_us, (e + 1) / fs_us, color=fill)

    # EMG time-domain overlay (mapped to US time frame, same as combined plot)
    t_emg_spec = np.linspace(0, us_duration, EMGdata_notched.shape[0])
    ax_ts = axes_spec[ch, 0].twinx()
    ax_ts.plot(t_emg_spec, EMGdata_notched[:, ch], color='white', alpha=0.3, linewidth=0.4)
    ylim_ts = max(abs(EMGdata_notched[:, ch].min()), abs(EMGdata_notched[:, ch].max()))
    ax_ts.set_ylim(-ylim_ts * 2, ylim_ts * 2)
    ax_ts.set_ylabel('Amp', color='white', fontsize=8)
    ax_ts.tick_params(axis='y', labelcolor='white', labelsize=7)

    # --- Right column: Per-frequency normalized (similar to Event-Related Spectral Perturbation = ERSP) ---
    # Each frequency bin is divided by its time-averaged power, removing the
    # spectral envelope (1/f roll-off). Shows relative power changes so that
    # activity at high frequencies is as visible as at low frequencies.
    Sxx_norm = Sxx / np.mean(Sxx, axis=1, keepdims=True)
    Sxx_norm_dB = 10 * np.log10(Sxx_norm + 1e-12)
    vmin_n, vmax_n = np.percentile(Sxx_norm_dB, [5, 95])
    im_norm = axes_spec[ch, 1].pcolormesh(t, f, Sxx_norm_dB, shading='gouraud', cmap=colormap,
                                           vmin=vmin_n, vmax=vmax_n)
    axes_spec[ch, 1].set_ylabel('Freq [Hz]')
    axes_spec[ch, 1].set_title(f'Ch{ch+1} — Per-Frequency Normalized')

    # # Label regions: shaded fills
    # for lv, regions in label_regions.items():
    #     if lv == 0 and not _INCLUDE_NOISE:
    #         continue
    #     fill = LABEL_COLORS_MPL.get(lv, (0.5, 0.5, 0.5, 0.15))
    #     for s, e in regions:
    #         axes_spec[ch, 1].axvspan(s / fs_us, (e + 1) / fs_us, color=fill)

    # EMG overlay
    ax_ts2 = axes_spec[ch, 1].twinx()
    ax_ts2.plot(t_emg_spec, EMGdata_notched[:, ch], color='white', alpha=0.3, linewidth=0.4)
    ax_ts2.set_ylim(-ylim_ts * 2, ylim_ts * 2)
    ax_ts2.set_ylabel('Amp', color='white', fontsize=8)
    ax_ts2.tick_params(axis='y', labelcolor='white', labelsize=7)

axes_spec[2, 0].set_xlabel('Time [s]')
axes_spec[2, 1].set_xlabel('Time [s]')
fig_spec.tight_layout(rect=[0, 0, 0.9, 1])

# Colorbars
cbar_ax1 = fig_spec.add_axes([0.91, 0.55, 0.015, 0.35])
fig_spec.colorbar(im_orig, cax=cbar_ax1, label='Power [dB]')
cbar_ax2 = fig_spec.add_axes([0.91, 0.1, 0.015, 0.35])
fig_spec.colorbar(im_norm, cax=cbar_ax2, label='Rel. Power [dB]')
plt.savefig('EMGspectrogram.png')
plt.show()