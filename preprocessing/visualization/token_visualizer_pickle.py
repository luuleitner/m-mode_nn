"""
Token Sample Visualizer for Pickle Datasets

Visualizes token samples from precomputed train/val/test pkl files.
Shows what the model actually sees during training, including augmented samples.
Also loads corresponding raw joystick data to show full signal context.

Display layout:
- Row 1: Full joystick signal with token window highlighted
- Row 2: Info box | Zoomed joystick window | Label distribution
- Row 3: Sample Info | Batch Info | Augmentation status
- Row 4: 3 US channel M-mode images

Usage:
    python visualization/token_visualizer_pickle.py
    python visualization/token_visualizer_pickle.py --split train --seed 42
    python visualization/token_visualizer_pickle.py --by-class
    python visualization/token_visualizer_pickle.py --compare
    python visualization/token_visualizer_pickle.py --data /path/to/pkl/folder

Options:
    --data, -d  Path to folder containing pkl files (default: from config)
    --split     Which split to visualize: train, val, test (default: train)
    --seed      Random seed for reproducible sample selection
    --by-class  Show 3 separate figures, one for each label class
    --compare   Show original and augmented version side-by-side
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config
from preprocessing.label_logic.label_logic import (
    create_position_peak_labels,
    create_5class_position_peak_labels
)
from preprocessing.signal_utils import apply_joystick_filters


class PickleTokenVisualizer:
    """Visualizer for pkl dataset files with joystick context."""

    def __init__(self, data_path, config_path=None):
        """
        Initialize the visualizer.

        Args:
            data_path: Path to folder containing train_ds.pkl, val_ds.pkl, test_ds.pkl
            config_path: Path to config.yaml (optional)
        """
        self.data_path = data_path
        self.datasets = {}

        # Load config
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        self.config = load_config(config_path, create_dirs=False)

        # Get raw data path from config
        self.raw_data_path = os.path.join(self.config.preprocess.data.basepath, 'raw')

        # Get processing parameters
        self.token_window = self.config.preprocess.tokenization.window
        self.token_stride = self.config.preprocess.tokenization.stride

        # Load label config
        label_config_path = os.path.join(project_root, 'preprocessing', 'label_logic', 'label_config.yaml')
        if os.path.exists(label_config_path):
            with open(label_config_path, 'r') as f:
                label_config = yaml.safe_load(f)
            self.label_method = label_config.get('method', 'position_peak')
            self.label_axis = label_config.get('axis', 'dual')
            self.filters_config = label_config.get('filters', {})
            # Position peak config
            pp_config = label_config.get('position_peak', {})
            self.pp_deriv_thresh = pp_config.get('deriv_threshold_percent', 10.0)
            self.pp_pos_thresh = pp_config.get('pos_threshold_percent', 5.0)
            self.pp_peak_window = pp_config.get('peak_window', 3)
            self.pp_timeout = pp_config.get('timeout_samples', 500)
        else:
            self.label_method = 'position_peak'
            self.label_axis = 'dual'
            self.filters_config = {}
            self.pp_deriv_thresh = 10.0
            self.pp_pos_thresh = 5.0
            self.pp_peak_window = 3
            self.pp_timeout = 500

        # Load available datasets
        for split in ['train', 'val', 'test']:
            pkl_path = os.path.join(data_path, f'{split}_ds.pkl')
            if os.path.exists(pkl_path):
                print(f"Loading {split}_ds.pkl...")
                with open(pkl_path, 'rb') as f:
                    self.datasets[split] = pickle.load(f)
                print(f"  {split}: {len(self.datasets[split])} batches")

        if not self.datasets:
            raise FileNotFoundError(f"No pkl files found in {data_path}")

    def load_raw_joystick(self, session, participant, experiment):
        """
        Load raw joystick data for an experiment.

        Returns:
            numpy array of joystick data or None if not found
        """
        try:
            # New hierarchy: P{participant}/session{session}/exp{experiment}
            exp_folder = os.path.join(
                self.raw_data_path,
                f"P{int(participant):03d}",
                f"session{int(session):03d}",
                f"exp{int(experiment):03d}"
            )

            if not os.path.exists(exp_folder):
                return None

            joy_path = os.path.join(exp_folder, '_joystick.npy')
            if os.path.exists(joy_path):
                return np.load(joy_path)
            return None
        except Exception as e:
            print(f"Warning: Could not load joystick data: {e}")
            return None

    def compute_token_window_indices(self, sequence_id, num_pulses):
        """Compute the pulse indices for a given sequence ID."""
        start_pulse = int(sequence_id) * self.token_stride
        end_pulse = start_pulse + self.token_window

        # Clamp to valid range
        start_pulse = max(0, min(start_pulse, num_pulses - self.token_window))
        end_pulse = start_pulse + self.token_window

        return start_pulse, end_pulse

    def create_labels_from_joystick(self, joystick_data):
        """
        Create per-sample 5-class labels from joystick data using both X and Y axes.

        Returns:
            dict with:
                'labels': merged 5-class labels (0=Noise, 1=Up, 2=Down, 3=Left, 4=Right)
                'x_position', 'x_derivative': filtered X-axis signals
                'y_position', 'y_derivative': filtered Y-axis signals
                'thresholds': dict with threshold values for both axes
                'x_markers', 'y_markers': edge/peak markers for each axis
        """
        # Get raw positions for both axes
        raw_x_position = joystick_data[:, 1]  # X axis
        raw_y_position = joystick_data[:, 2]  # Y axis

        # Apply filters to X position
        if self.filters_config:
            x_position = apply_joystick_filters(raw_x_position.copy(), self.filters_config, 'position')
        else:
            x_position = raw_x_position.copy()

        x_derivative = np.gradient(x_position)
        if self.filters_config:
            x_derivative = apply_joystick_filters(x_derivative, self.filters_config, 'derivative')

        # Apply filters to Y position
        if self.filters_config:
            y_position = apply_joystick_filters(raw_y_position.copy(), self.filters_config, 'position')
        else:
            y_position = raw_y_position.copy()

        y_derivative = np.gradient(y_position)
        if self.filters_config:
            y_derivative = apply_joystick_filters(y_derivative, self.filters_config, 'derivative')

        # Create 5-class labels using both axes
        labels, thresholds, markers = create_5class_position_peak_labels(
            x_position, y_position,
            x_derivative, y_derivative,
            self.pp_deriv_thresh, self.pp_pos_thresh,
            self.pp_peak_window, self.pp_timeout
        )

        return {
            'labels': labels,
            'x_position': x_position,
            'x_derivative': x_derivative,
            'y_position': y_position,
            'y_derivative': y_derivative,
            'thresholds': thresholds,
            'x_markers': markers.get('x', {}),
            'y_markers': markers.get('y', {})
        }

    def load_random_sample(self, split='train', seed=None, augmented_only=False,
                           original_only=False, target_label=None):
        """Load a random sample from the dataset."""
        if seed is not None:
            np.random.seed(seed)

        ds = self.datasets[split]

        # Filter batch_mapping based on criteria
        valid_batches = []
        for batch_idx, batch_info in enumerate(ds.batch_mapping):
            for sample_idx, item in enumerate(batch_info):
                metadata = item['sequence_metadata']

                # Check augmented filter
                is_augmented = item.get('is_augmented', metadata.get('is_augmented', False))
                if augmented_only and not is_augmented:
                    continue
                if original_only and is_augmented:
                    continue

                # Check label filter
                if target_label is not None:
                    label = metadata.get('label_logic', metadata.get('token label_logic', -1))
                    if int(label) != target_label:
                        continue

                valid_batches.append((batch_idx, sample_idx, item))

        if not valid_batches:
            raise ValueError(f"No samples found matching criteria in {split} split")

        # Random selection
        batch_idx, sample_idx, item = valid_batches[np.random.randint(len(valid_batches))]

        # Load actual data
        batch_data = ds[batch_idx]
        tokens = batch_data['tokens'].numpy()
        labels = batch_data['labels'].numpy() if batch_data['labels'] is not None else None

        metadata = item['sequence_metadata']

        return {
            'token': tokens[sample_idx],
            'label': labels[sample_idx] if labels is not None else None,
            'batch_idx': batch_idx,
            'sample_idx': sample_idx,
            'is_augmented': item.get('is_augmented', metadata.get('is_augmented', False)),
            'metadata': metadata,
            'split': split,
            'file_path': item['file_path']
        }

    def load_specific_sample(self, split='train', batch_idx=0, sample_idx=0):
        """Load a specific sample by batch and sample index."""
        ds = self.datasets[split]

        if batch_idx >= len(ds):
            raise IndexError(f"Batch index {batch_idx} out of range (max: {len(ds)-1})")

        batch_info = ds.batch_mapping[batch_idx]
        if sample_idx >= len(batch_info):
            raise IndexError(f"Sample index {sample_idx} out of range (max: {len(batch_info)-1})")

        item = batch_info[sample_idx]

        # Load actual data
        batch_data = ds[batch_idx]
        tokens = batch_data['tokens'].numpy()
        labels = batch_data['labels'].numpy() if batch_data['labels'] is not None else None

        metadata = item['sequence_metadata']

        return {
            'token': tokens[sample_idx],
            'label': labels[sample_idx] if labels is not None else None,
            'batch_idx': batch_idx,
            'sample_idx': sample_idx,
            'is_augmented': item.get('is_augmented', metadata.get('is_augmented', False)),
            'metadata': metadata,
            'split': split,
            'file_path': item['file_path']
        }

    def visualize_sample(self, sample):
        """Create visualization for a sample with 5-class dual-axis joystick context."""
        token_data = sample['token']
        label_data = sample['label']
        metadata = sample['metadata']

        # Extract metadata
        session = metadata.get('session', 'N/A')
        participant = metadata.get('participant', 'N/A')
        experiment = metadata.get('experiment', 'N/A')
        seq_id = metadata.get('sequence_id', 0)

        # Load joystick data
        joystick_data = None
        if session != 'N/A' and experiment != 'N/A':
            joystick_data = self.load_raw_joystick(session, participant, experiment)

        has_joystick = joystick_data is not None

        # 5-class label colors
        label_colors_5class = {
            0: 'rgba(128, 128, 128, 0.15)',  # Noise - gray
            1: 'rgba(0, 200, 0, 0.25)',       # Up - green
            2: 'rgba(220, 50, 50, 0.25)',     # Down - red
            3: 'rgba(50, 100, 220, 0.25)',    # Left - blue
            4: 'rgba(255, 165, 0, 0.25)',     # Right - orange
        }

        # Create figure layout
        if has_joystick:
            # Full layout with dual-axis joystick context (5 rows)
            fig = make_subplots(
                rows=5, cols=3,
                specs=[
                    [{"colspan": 3, "secondary_y": True}, None, None],  # Row 1: X-axis
                    [{"colspan": 3, "secondary_y": True}, None, None],  # Row 2: Y-axis
                    [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],  # Row 3
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],  # Row 4
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],  # Row 5
                ],
                subplot_titles=[
                    f"X-Axis (Left/Right) - Sequence {seq_id}",
                    "Y-Axis (Up/Down)",
                    "Sample Info", "Zoomed Window", "Label Distribution",
                    "Batch Info", "Augmentation Status", "",
                    "US Channel 1", "US Channel 2", "US Channel 3"
                ],
                vertical_spacing=0.05,
                horizontal_spacing=0.06,
                row_heights=[0.18, 0.18, 0.16, 0.12, 0.36]
            )
        else:
            # Fallback layout without joystick
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    "Sample Info", "Label Distribution", "Batch Info",
                    "Augmentation Status", "", "",
                    "US Channel 1", "US Channel 2", "US Channel 3"
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
                row_heights=[0.25, 0.15, 0.60]
            )

        # ===== JOYSTICK CONTEXT (if available) =====
        if has_joystick:
            num_pulses = joystick_data.shape[0]
            start_pulse, end_pulse = self.compute_token_window_indices(seq_id, num_pulses)

            # Create 5-class labels from joystick (both axes)
            label_result = self.create_labels_from_joystick(joystick_data)
            labels_joy = label_result['labels']
            x_position = label_result['x_position']
            x_derivative = label_result['x_derivative']
            y_position = label_result['y_position']
            y_derivative = label_result['y_derivative']
            thresholds = label_result['thresholds']
            x_markers = label_result['x_markers']
            y_markers = label_result['y_markers']

            x_full = np.arange(len(labels_joy))

            # Helper to add label regions to a row
            def add_label_regions(row):
                for label_val in [1, 2, 3, 4]:  # Skip noise for cleaner display
                    mask = labels_joy == label_val
                    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    for s, e in zip(starts, ends):
                        fig.add_vrect(x0=s, x1=e, fillcolor=label_colors_5class[label_val],
                                     layer="below", line_width=0, row=row, col=1)

            # Helper to add markers for an axis
            def add_axis_markers(row, position, derivative, markers, axis_name):
                if markers is None:
                    return
                # Edge up markers
                if len(markers.get('edge_up', [])) > 0:
                    edge_up_idx = np.array(markers['edge_up'])
                    fig.add_trace(
                        go.Scatter(x=edge_up_idx, y=position[edge_up_idx],
                                  mode='markers',
                                  marker=dict(symbol='triangle-up', size=10, color='green',
                                             line=dict(color='white', width=1)),
                                  opacity=0.8, name=f'{axis_name} Edge+', legendgroup=f'{axis_name}_edge_up',
                                  showlegend=(row == 1)),
                        row=row, col=1, secondary_y=False
                    )
                # Edge down markers
                if len(markers.get('edge_down', [])) > 0:
                    edge_down_idx = np.array(markers['edge_down'])
                    fig.add_trace(
                        go.Scatter(x=edge_down_idx, y=position[edge_down_idx],
                                  mode='markers',
                                  marker=dict(symbol='triangle-down', size=10, color='red',
                                             line=dict(color='white', width=1)),
                                  opacity=0.8, name=f'{axis_name} Edge-', legendgroup=f'{axis_name}_edge_down',
                                  showlegend=(row == 1)),
                        row=row, col=1, secondary_y=False
                    )
                # Peak markers
                for key, color, name_suffix in [('peak_up', 'darkgreen', 'Peak+'), ('peak_down', 'darkred', 'Peak-')]:
                    if key in markers and len(markers[key]) > 0:
                        idx = np.array(markers[key])
                        fig.add_trace(
                            go.Scatter(x=idx, y=derivative[idx],
                                      mode='markers',
                                      marker=dict(symbol='circle', size=8, color=color),
                                      name=f'{axis_name} {name_suffix}', legendgroup=f'{axis_name}_{key}',
                                      showlegend=(row == 1)),
                            row=row, col=1, secondary_y=True
                        )

            # === ROW 1: X-axis (Left/Right) ===
            add_label_regions(1)

            # X Position trace
            fig.add_trace(
                go.Scatter(x=x_full, y=x_position, mode='lines',
                          line=dict(color='blue', width=1.5),
                          name='X Position'),
                row=1, col=1, secondary_y=False
            )

            # X Derivative trace
            fig.add_trace(
                go.Scatter(x=x_full, y=x_derivative, mode='lines',
                          line=dict(color='purple', width=1), opacity=0.6,
                          name='X Derivative'),
                row=1, col=1, secondary_y=True
            )

            # X thresholds
            x_pos_thresh = thresholds.get('x_pos')
            if x_pos_thresh is not None:
                fig.add_hline(y=x_pos_thresh, line=dict(color='blue', dash='dash', width=1),
                             opacity=0.4, row=1, col=1, secondary_y=False)
                fig.add_hline(y=-x_pos_thresh, line=dict(color='blue', dash='dash', width=1),
                             opacity=0.4, row=1, col=1, secondary_y=False)

            add_axis_markers(1, x_position, x_derivative, x_markers, 'X')

            # Token window highlight on row 1
            fig.add_vrect(x0=start_pulse, x1=end_pulse, fillcolor='rgba(0, 255, 255, 0.2)',
                         layer="above", line=dict(color='cyan', width=2), row=1, col=1)

            # === ROW 2: Y-axis (Up/Down) ===
            add_label_regions(2)

            # Y Position trace
            fig.add_trace(
                go.Scatter(x=x_full, y=y_position, mode='lines',
                          line=dict(color='darkgreen', width=1.5),
                          name='Y Position'),
                row=2, col=1, secondary_y=False
            )

            # Y Derivative trace
            fig.add_trace(
                go.Scatter(x=x_full, y=y_derivative, mode='lines',
                          line=dict(color='orange', width=1), opacity=0.6,
                          name='Y Derivative'),
                row=2, col=1, secondary_y=True
            )

            # Y thresholds
            y_pos_thresh = thresholds.get('y_pos')
            if y_pos_thresh is not None:
                fig.add_hline(y=y_pos_thresh, line=dict(color='darkgreen', dash='dash', width=1),
                             opacity=0.4, row=2, col=1, secondary_y=False)
                fig.add_hline(y=-y_pos_thresh, line=dict(color='darkgreen', dash='dash', width=1),
                             opacity=0.4, row=2, col=1, secondary_y=False)

            add_axis_markers(2, y_position, y_derivative, y_markers, 'Y')

            # Token window highlight on row 2
            fig.add_vrect(x0=start_pulse, x1=end_pulse, fillcolor='rgba(0, 255, 255, 0.2)',
                         layer="above", line=dict(color='cyan', width=2), row=2, col=1)
            fig.add_vline(x=start_pulse, line=dict(color='magenta', width=2), row=2, col=1)
            fig.add_vline(x=end_pulse, line=dict(color='magenta', width=2), row=2, col=1)

            # Fix x-axis range (autorange=False forces explicit range)
            num_samples = len(labels_joy)
            fig.update_xaxes(range=[0, num_samples], autorange=False, row=1, col=1)
            fig.update_xaxes(range=[0, num_samples], autorange=False, row=2, col=1)
            fig.update_xaxes(title_text="Pulse Index", row=1, col=1)
            fig.update_xaxes(title_text="Pulse Index", row=2, col=1)
            fig.update_yaxes(title_text="X Position", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="X Derivative", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Y Position", row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Y Derivative", row=2, col=1, secondary_y=True)

            # === ROW 3, COL 1: Sample Info ===
            aug_status = "AUGMENTED" if sample['is_augmented'] else "ORIGINAL"
            aug_color = "orange" if sample['is_augmented'] else "green"

            info_text = (
                f"<b>Sample Info</b><br>"
                f"<span style='color:{aug_color}'><b>{aug_status}</b></span><br><br>"
                f"Split: {sample['split']}<br>"
                f"Session: {session}<br>"
                f"Participant: {participant}<br>"
                f"Experiment: {experiment}<br>"
                f"Sequence ID: {seq_id}<br>"
                f"<b>5-Class Dual-Axis</b>"
            )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers',
                                     marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                          row=3, col=1)
            fig.add_annotation(
                x=0.5, y=0.5, text=info_text,
                showarrow=False, font=dict(size=10),
                xref="x3 domain", yref="y3 domain",
                align="left", bgcolor="rgba(255,255,255,0.9)",
                bordercolor=aug_color, borderwidth=2
            )
            fig.update_xaxes(visible=False, row=3, col=1)
            fig.update_yaxes(visible=False, row=3, col=1)

            # === ROW 3, COL 2: Zoomed joystick window (both axes) ===
            x_zoom = np.arange(start_pulse, end_pulse)
            x_pos_zoom = x_position[start_pulse:end_pulse]
            y_pos_zoom = y_position[start_pulse:end_pulse]

            # Add 5-class label regions in zoomed view
            labels_zoom = labels_joy[start_pulse:end_pulse]
            for label_val in [1, 2, 3, 4]:
                mask = labels_zoom == label_val
                diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                starts_z = np.where(diff == 1)[0] + start_pulse
                ends_z = np.where(diff == -1)[0] + start_pulse
                for s, e in zip(starts_z, ends_z):
                    fig.add_vrect(x0=s, x1=e, fillcolor=label_colors_5class[label_val],
                                 layer="below", line_width=0, row=3, col=2)

            # Show both X and Y positions in zoomed view
            fig.add_trace(
                go.Scatter(x=x_zoom, y=x_pos_zoom, mode='lines',
                          line=dict(color='blue', width=2),
                          name='X (zoom)', showlegend=False),
                row=3, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=x_zoom, y=y_pos_zoom, mode='lines',
                          line=dict(color='darkgreen', width=2),
                          name='Y (zoom)', showlegend=False),
                row=3, col=2, secondary_y=True
            )

            # === ROW 3, COL 3: Label distribution (5-class) ===
            self._add_label_distribution(fig, label_data, row=3, col=3)

            # === ROW 4, COL 1: Batch Info ===
            self._add_batch_info(fig, sample, row=4, col=1, xref="x7 domain", yref="y7 domain")

            # === ROW 4, COL 2: Augmentation Status ===
            self._add_augmentation_info(fig, sample, row=4, col=2, xref="x8 domain", yref="y8 domain")

            # === ROW 5: US Channels ===
            us_row = 5

        else:
            # Fallback: No joystick data
            # === ROW 1: Info panels ===
            aug_status = "AUGMENTED" if sample['is_augmented'] else "ORIGINAL"
            aug_color = "orange" if sample['is_augmented'] else "green"

            info_text = (
                f"<b>Sample Info</b><br>"
                f"<span style='color:{aug_color}'><b>{aug_status}</b></span><br><br>"
                f"Split: {sample['split']}<br>"
                f"Batch: {sample['batch_idx']}, Sample: {sample['sample_idx']}<br>"
                f"Session: {session}<br>"
                f"Participant: {participant}<br>"
                f"Experiment: {experiment}<br>"
                f"Sequence ID: {seq_id}"
            )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers',
                                     marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                          row=1, col=1)
            fig.add_annotation(
                x=0.5, y=0.5, text=info_text,
                showarrow=False, font=dict(size=11),
                xref="x domain", yref="y domain",
                align="left", bgcolor="rgba(255,255,255,0.9)",
                bordercolor=aug_color, borderwidth=2
            )
            fig.update_xaxes(visible=False, row=1, col=1)
            fig.update_yaxes(visible=False, row=1, col=1)

            self._add_label_distribution(fig, label_data, row=1, col=2)
            self._add_batch_info(fig, sample, row=1, col=3, xref="x3 domain", yref="y3 domain")

            # === ROW 2: Augmentation info ===
            self._add_augmentation_info(fig, sample, row=2, col=1, xref="x4 domain", yref="y4 domain")

            us_row = 3

        # === US CHANNELS ===
        token_display = token_data
        if len(token_data.shape) == 4:
            mid_idx = token_data.shape[0] // 2
            token_display = token_data[mid_idx]

        for ch_idx in range(min(3, token_display.shape[0])):
            ch_data = token_display[ch_idx]
            fig.add_trace(
                go.Heatmap(
                    z=ch_data,
                    colorscale='greys',
                    showscale=(ch_idx == 2),
                    colorbar=dict(title="Amp", x=1.02) if ch_idx == 2 else None
                ),
                row=us_row, col=ch_idx + 1
            )
            fig.update_xaxes(title_text="Pulse", row=us_row, col=ch_idx + 1)
            fig.update_yaxes(title_text="Depth" if ch_idx == 0 else "",
                            autorange="reversed", row=us_row, col=ch_idx + 1)

        # Layout
        title_text = f"Token Visualization ({sample['split'].upper()}) - "
        title_text += f"S{session}_P{participant}_E{experiment}"
        if sample['is_augmented']:
            title_text += " [AUGMENTED]"

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=16)),
            height=1200 if has_joystick else 700,
            width=1400,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def _add_label_distribution(self, fig, label_data, row, col):
        """Add label distribution bar chart (5-class)."""
        if label_data is not None:
            label_flat = label_data.flatten()
            # 5-class names and colors
            class_names = ['Noise', 'Up', 'Down', 'Left', 'Right']
            colors = ['gray', 'green', 'red', 'blue', 'orange']

            if label_flat.size == 1:
                label_val = int(label_flat[0])
                probs = [1.0 if i == label_val else 0.0 for i in range(5)]
            elif label_data.dtype in [np.float32, np.float64] and np.allclose(label_flat.sum(), 1.0, atol=0.1):
                probs = label_flat.tolist()
                if len(probs) < 5:
                    probs.extend([0] * (5 - len(probs)))
            else:
                counts = [np.sum(label_flat == i) for i in range(5)]
                total = sum(counts)
                probs = [c / total if total > 0 else 0 for c in counts]

            fig.add_trace(
                go.Bar(x=class_names, y=probs, marker_color=colors, showlegend=False),
                row=row, col=col
            )
            fig.update_yaxes(range=[0, 1], title_text="Prob", row=row, col=col)
        else:
            fig.add_annotation(
                x=0.5, y=0.5, text="No label data",
                showarrow=False, row=row, col=col
            )

    def _add_batch_info(self, fig, sample, row, col, xref, yref):
        """Add batch info panel."""
        ds = self.datasets[sample['split']]
        batch_info = ds.batch_mapping[sample['batch_idx']]

        batch_labels = []
        batch_augmented = 0
        for item in batch_info:
            meta = item['sequence_metadata']
            lbl = meta.get('label_logic', meta.get('token label_logic', 0))
            batch_labels.append(int(lbl))
            if item.get('is_augmented', meta.get('is_augmented', False)):
                batch_augmented += 1

        batch_counts = {0: batch_labels.count(0), 1: batch_labels.count(1), 2: batch_labels.count(2)}
        batch_total = len(batch_labels)

        batch_text = (
            f"<b>Batch Info</b><br><br>"
            f"Batch Index: {sample['batch_idx']}<br>"
            f"Batch Size: {batch_total}<br>"
            f"Augmented: {batch_augmented}<br>"
            f"Original: {batch_total - batch_augmented}<br><br>"
            f"<b>Class Distribution:</b><br>"
            f"0: {batch_counts[0]} | 1: {batch_counts[1]} | 2: {batch_counts[2]}"
        )
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers',
                                 marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                      row=row, col=col)
        fig.add_annotation(
            x=0.5, y=0.5, text=batch_text,
            showarrow=False, font=dict(size=11),
            xref=xref, yref=yref,
            align="left", bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray", borderwidth=1
        )
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def _add_augmentation_info(self, fig, sample, row, col, xref, yref):
        """Add augmentation status panel."""
        is_aug = sample['is_augmented']
        metadata = sample['metadata']

        if is_aug:
            aug_seed = metadata.get('augment_seed', 'N/A')
            aug_text = (
                f"<b style='color:orange'>AUGMENTED SAMPLE</b><br><br>"
                f"This sample was created by<br>"
                f"oversampling a minority class.<br><br>"
                f"Augmentation seed: {aug_seed}<br><br>"
                f"Transforms applied:<br>"
                f"- Gaussian noise<br>"
                f"- Amplitude scaling<br>"
                f"- Temporal shift"
            )
            border_color = "orange"
        else:
            aug_text = (
                f"<b style='color:green'>ORIGINAL SAMPLE</b><br><br>"
                f"This is an unmodified sample<br>"
                f"from the source H5 file.<br><br>"
                f"No augmentation applied."
            )
            border_color = "green"

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers',
                                 marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                      row=row, col=col)
        fig.add_annotation(
            x=0.5, y=0.5, text=aug_text,
            showarrow=False, font=dict(size=11),
            xref=xref, yref=yref,
            align="left", bgcolor="rgba(255,255,255,0.9)",
            bordercolor=border_color, borderwidth=2
        )
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def visualize_compare(self, seed=None, split='train'):
        """Show original and augmented version side by side with joystick context."""
        ds = self.datasets[split]

        if seed is not None:
            np.random.seed(seed)

        # Find pairs from same source
        samples_by_source = {}
        for batch_idx, batch_info in enumerate(ds.batch_mapping):
            for sample_idx, item in enumerate(batch_info):
                metadata = item['sequence_metadata']
                seq_id = metadata.get('sequence_id', 0)
                file_path = item['file_path']
                key = (file_path, seq_id)

                is_augmented = item.get('is_augmented', metadata.get('is_augmented', False))

                if key not in samples_by_source:
                    samples_by_source[key] = {'original': [], 'augmented': []}

                if is_augmented:
                    samples_by_source[key]['augmented'].append((batch_idx, sample_idx))
                else:
                    samples_by_source[key]['original'].append((batch_idx, sample_idx))

        valid_sources = [k for k, v in samples_by_source.items()
                        if v['original'] and v['augmented']]

        if not valid_sources:
            print("No source samples found with both original and augmented versions.")
            # Try to find any augmented samples
            try:
                sample_orig = self.load_random_sample(split=split, original_only=True, seed=seed)
                sample_aug = self.load_random_sample(split=split, augmented_only=True, seed=seed)
                print("Showing random original and random augmented instead.")
                same_source = False
            except ValueError:
                print("No augmented samples found in dataset. Showing two random original samples.")
                sample_orig = self.load_random_sample(split=split, original_only=True, seed=seed)
                # Use different seed for second sample
                sample_aug = self.load_random_sample(split=split, original_only=True, seed=(seed + 1) if seed else None)
                same_source = False
        else:
            source_key = valid_sources[np.random.randint(len(valid_sources))]
            orig_loc = samples_by_source[source_key]['original'][0]
            aug_loc = samples_by_source[source_key]['augmented'][0]

            sample_orig = self.load_specific_sample(split, orig_loc[0], orig_loc[1])
            sample_aug = self.load_specific_sample(split, aug_loc[0], aug_loc[1])
            same_source = True

        # Get metadata for joystick
        orig_meta = sample_orig['metadata']
        aug_meta = sample_aug['metadata']
        session = orig_meta.get('session', 'N/A')
        participant = orig_meta.get('participant', 'N/A')
        experiment = orig_meta.get('experiment', 'N/A')
        seq_id = orig_meta.get('sequence_id', 0)

        # Load joystick
        joystick_data = None
        if session != 'N/A' and experiment != 'N/A':
            joystick_data = self.load_raw_joystick(session, participant, experiment)

        has_joystick = joystick_data is not None

        if has_joystick:
            # Full layout with joystick - 3 columns for info row
            fig = make_subplots(
                rows=4, cols=3,
                specs=[
                    [{"colspan": 3, "secondary_y": True}, None, None],
                    [{"colspan": 1}, {"colspan": 1, "secondary_y": True}, {"colspan": 1}],
                    [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
                    [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
                ],
                subplot_titles=[
                    f"Full Joystick Signal (Sequence {seq_id})",
                    "Sample Info", f"Zoomed Window (pulses {seq_id * self.token_stride}-{seq_id * self.token_stride + self.token_window})", "Label Comparison",
                    "Original Ch1", "Original Ch2", "Original Ch3",
                    "Augmented Ch1", "Augmented Ch2", "Augmented Ch3"
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
                row_heights=[0.28, 0.15, 0.285, 0.285]
            )

            # Add joystick row
            num_pulses = joystick_data.shape[0]
            start_pulse, end_pulse = self.compute_token_window_indices(seq_id, num_pulses)

            labels_joy, position, derivative, pos_threshold, deriv_threshold, markers = \
                self.create_labels_from_joystick(joystick_data)

            x_full = np.arange(len(position))

            # Label regions
            label_colors = {1: 'rgba(144, 238, 144, 0.2)', 2: 'rgba(240, 128, 128, 0.2)'}
            if labels_joy is not None:
                for label_val in [1, 2]:
                    mask = labels_joy == label_val
                    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    for s, e in zip(starts, ends):
                        fig.add_vrect(x0=s, x1=e, fillcolor=label_colors[label_val],
                                     layer="below", line_width=0, row=1, col=1)

            fig.add_trace(
                go.Scatter(x=x_full, y=position, mode='lines',
                          line=dict(color='blue', width=1), name='Position'),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=x_full, y=derivative, mode='lines',
                          line=dict(color='red', width=1), opacity=0.7, name='Derivative'),
                row=1, col=1, secondary_y=True
            )

            # Highlight token window with prominent cyan box
            fig.add_vrect(x0=start_pulse, x1=end_pulse,
                         fillcolor='rgba(0, 255, 255, 0.25)', layer="above",
                         line=dict(color='cyan', width=3), row=1, col=1)

            # Add vertical lines at window boundaries
            fig.add_vline(x=start_pulse, line=dict(color='magenta', width=2, dash='solid'),
                         row=1, col=1)
            fig.add_vline(x=end_pulse, line=dict(color='magenta', width=2, dash='solid'),
                         row=1, col=1)

            # Add annotation pointing to the window
            mid_pulse = (start_pulse + end_pulse) / 2
            fig.add_annotation(
                x=mid_pulse, y=1.05,
                text=f"<b>Token Window</b><br>(pulses {start_pulse}-{end_pulse})",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='magenta',
                ax=0, ay=-40,
                font=dict(size=10, color='magenta'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='magenta', borderwidth=1,
                xref="x", yref="y domain",
                row=1, col=1
            )

            # Add scatter markers for edge/peak/derivative crossings
            if markers is not None:
                # Edge up markers on POSITION trace (green triangles)
                if len(markers.get('edge_up', [])) > 0:
                    edge_up_idx = markers['edge_up']
                    fig.add_trace(
                        go.Scatter(x=edge_up_idx, y=position[edge_up_idx],
                                  mode='markers',
                                  marker=dict(symbol='triangle-up', size=12, color='green',
                                             line=dict(color='white', width=1)),
                                  opacity=0.8, name='Edge Up', legendgroup='edge_up',
                                  showlegend=True),
                        row=1, col=1, secondary_y=False
                    )

                # Edge down markers on POSITION trace (red triangles)
                if len(markers.get('edge_down', [])) > 0:
                    edge_down_idx = markers['edge_down']
                    fig.add_trace(
                        go.Scatter(x=edge_down_idx, y=position[edge_down_idx],
                                  mode='markers',
                                  marker=dict(symbol='triangle-down', size=12, color='red',
                                             line=dict(color='white', width=1)),
                                  opacity=0.8, name='Edge Down', legendgroup='edge_down',
                                  showlegend=True),
                        row=1, col=1, secondary_y=False
                    )

                # Peak markers for edge_to_peak method (circles on derivative)
                if 'peak_up' in markers and len(markers['peak_up']) > 0:
                    peak_up_idx = markers['peak_up']
                    fig.add_trace(
                        go.Scatter(x=peak_up_idx, y=derivative[peak_up_idx],
                                  mode='markers',
                                  marker=dict(symbol='circle', size=10, color='darkgreen'),
                                  name='Peak Up', legendgroup='peak_up',
                                  showlegend=True),
                        row=1, col=1, secondary_y=True
                    )

                if 'peak_down' in markers and len(markers['peak_down']) > 0:
                    peak_down_idx = markers['peak_down']
                    fig.add_trace(
                        go.Scatter(x=peak_down_idx, y=derivative[peak_down_idx],
                                  mode='markers',
                                  marker=dict(symbol='circle', size=10, color='darkred'),
                                  name='Peak Down', legendgroup='peak_down',
                                  showlegend=True),
                        row=1, col=1, secondary_y=True
                    )

                # Derivative crossing markers for edge_to_derivative method (squares)
                if 'deriv_cross_up' in markers and len(markers['deriv_cross_up']) > 0:
                    cross_up_idx = markers['deriv_cross_up']
                    fig.add_trace(
                        go.Scatter(x=cross_up_idx, y=derivative[cross_up_idx],
                                  mode='markers',
                                  marker=dict(symbol='square', size=10, color='darkgreen',
                                             line=dict(color='white', width=1)),
                                  name='Deriv Cross Up', legendgroup='deriv_cross_up',
                                  showlegend=True),
                        row=1, col=1, secondary_y=True
                    )

                if 'deriv_cross_down' in markers and len(markers['deriv_cross_down']) > 0:
                    cross_down_idx = markers['deriv_cross_down']
                    fig.add_trace(
                        go.Scatter(x=cross_down_idx, y=derivative[cross_down_idx],
                                  mode='markers',
                                  marker=dict(symbol='square', size=10, color='darkred',
                                             line=dict(color='white', width=1)),
                                  name='Deriv Cross Down', legendgroup='deriv_cross_down',
                                  showlegend=True),
                        row=1, col=1, secondary_y=True
                    )

            fig.update_yaxes(title_text="Position", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Derivative", row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Pulse Index", row=1, col=1)

            # === ROW 2, COL 2: Zoomed window ===
            x_zoom = np.arange(start_pulse, end_pulse)
            pos_zoom = position[start_pulse:end_pulse]
            deriv_zoom = derivative[start_pulse:end_pulse]

            # Add label regions in zoomed view
            if labels_joy is not None:
                labels_zoom = labels_joy[start_pulse:end_pulse]
                for label_val in [1, 2]:
                    mask = labels_zoom == label_val
                    diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                    starts_z = np.where(diff == 1)[0] + start_pulse
                    ends_z = np.where(diff == -1)[0] + start_pulse
                    for s, e in zip(starts_z, ends_z):
                        fig.add_vrect(x0=s, x1=e, fillcolor=label_colors[label_val],
                                     layer="below", line_width=0, row=2, col=2)

            fig.add_trace(
                go.Scatter(x=x_zoom, y=pos_zoom, mode='lines',
                          line=dict(color='blue', width=2),
                          name='Position (zoom)', showlegend=False),
                row=2, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=x_zoom, y=deriv_zoom, mode='lines',
                          line=dict(color='red', width=2), opacity=0.7,
                          name='Derivative (zoom)', showlegend=False),
                row=2, col=2, secondary_y=True
            )

            # Add scatter markers to zoomed view (only markers within the window)
            if markers is not None:
                def filter_to_window(idx_array):
                    """Filter marker indices to those within the token window."""
                    if len(idx_array) == 0:
                        return np.array([])
                    mask = (idx_array >= start_pulse) & (idx_array < end_pulse)
                    return idx_array[mask]

                # Edge up markers in zoomed view
                edge_up_zoom = filter_to_window(np.array(markers.get('edge_up', [])))
                if len(edge_up_zoom) > 0:
                    fig.add_trace(
                        go.Scatter(x=edge_up_zoom, y=position[edge_up_zoom],
                                  mode='markers',
                                  marker=dict(symbol='triangle-up', size=14, color='green',
                                             line=dict(color='white', width=2)),
                                  name='Edge Up (zoom)', showlegend=False),
                        row=2, col=2, secondary_y=False
                    )

                # Edge down markers in zoomed view
                edge_down_zoom = filter_to_window(np.array(markers.get('edge_down', [])))
                if len(edge_down_zoom) > 0:
                    fig.add_trace(
                        go.Scatter(x=edge_down_zoom, y=position[edge_down_zoom],
                                  mode='markers',
                                  marker=dict(symbol='triangle-down', size=14, color='red',
                                             line=dict(color='white', width=2)),
                                  name='Edge Down (zoom)', showlegend=False),
                        row=2, col=2, secondary_y=False
                    )

                # Peak/derivative crossing markers in zoomed view
                for key, color, symbol in [('peak_up', 'darkgreen', 'circle'),
                                            ('peak_down', 'darkred', 'circle'),
                                            ('deriv_cross_up', 'darkgreen', 'square'),
                                            ('deriv_cross_down', 'darkred', 'square')]:
                    if key in markers:
                        idx_zoom = filter_to_window(np.array(markers[key]))
                        if len(idx_zoom) > 0:
                            fig.add_trace(
                                go.Scatter(x=idx_zoom, y=derivative[idx_zoom],
                                          mode='markers',
                                          marker=dict(symbol=symbol, size=12, color=color,
                                                     line=dict(color='white', width=1)),
                                          showlegend=False),
                                row=2, col=2, secondary_y=True
                            )

            info_row = 2
            us_orig_row = 3
            us_aug_row = 4
            xref_info = "x2"
        else:
            # Compact layout without joystick - 2 columns for info row
            fig = make_subplots(
                rows=3, cols=3,
                specs=[
                    [{"colspan": 1}, {"colspan": 2}, None],
                    [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
                    [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
                ],
                subplot_titles=[
                    "Sample Info", "Label Comparison",
                    "Original Ch1", "Original Ch2", "Original Ch3",
                    "Augmented Ch1", "Augmented Ch2", "Augmented Ch3"
                ],
                vertical_spacing=0.10,
                horizontal_spacing=0.06,
                row_heights=[0.15, 0.425, 0.425]
            )
            info_row = 1
            us_orig_row = 2
            us_aug_row = 3
            xref_info = "x"

        # Combined info panel (Original + Augmented)
        info_text = (
            f"<b>Sample Info</b><br><br>"
            f"<b style='color:green'>ORIGINAL</b>: "
            f"Batch {sample_orig['batch_idx']}, Sample {sample_orig['sample_idx']}<br>"
            f"<b style='color:orange'>AUGMENTED</b>: "
            f"Batch {sample_aug['batch_idx']}, Sample {sample_aug['sample_idx']}<br><br>"
            f"Session: {session} | Participant: {participant}<br>"
            f"Experiment: {experiment} | Sequence: {seq_id}"
        )

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers',
                                 marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                      row=info_row, col=1)
        fig.add_annotation(
            x=0.5, y=0.5, text=info_text,
            showarrow=False, font=dict(size=10),
            xref=f"{xref_info} domain", yref=f"y{info_row if has_joystick else ''} domain".replace("y domain", "y domain"),
            align="left", bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray", borderwidth=1
        )
        fig.update_xaxes(visible=False, row=info_row, col=1)
        fig.update_yaxes(visible=False, row=info_row, col=1)

        # Label comparison
        def get_probs(label_data):
            if label_data is None:
                return [0, 0, 0]
            label_flat = label_data.flatten()
            if label_flat.size == 1:
                lbl = int(label_flat[0])
                return [1.0 if i == lbl else 0.0 for i in range(3)]
            elif np.allclose(label_flat.sum(), 1.0, atol=0.1):
                probs = label_flat.tolist()
                return probs + [0] * (3 - len(probs))
            else:
                counts = [np.sum(label_flat == i) for i in range(3)]
                total = sum(counts)
                return [c / total if total > 0 else 0 for c in counts]

        probs_orig = get_probs(sample_orig['label'])
        probs_aug = get_probs(sample_aug['label'])
        class_names = ['Noise', 'Up/Right', 'Down/Left']

        # Label column is 3 when we have joystick (zoomed window is col 2), otherwise col 2
        label_comparison_col = 3 if has_joystick else 2

        fig.add_trace(
            go.Bar(name='Original', x=class_names, y=probs_orig,
                   marker_color='green', opacity=0.7),
            row=info_row, col=label_comparison_col
        )
        fig.add_trace(
            go.Bar(name='Augmented', x=class_names, y=probs_aug,
                   marker_color='orange', opacity=0.7),
            row=info_row, col=label_comparison_col
        )
        fig.update_yaxes(range=[0, 1], title_text="Prob", row=info_row, col=label_comparison_col)
        fig.update_layout(barmode='group')

        # US channels - Original
        token_orig = sample_orig['token']
        if len(token_orig.shape) == 4:
            token_orig = token_orig[token_orig.shape[0] // 2]

        for ch_idx in range(min(3, token_orig.shape[0])):
            fig.add_trace(
                go.Heatmap(z=token_orig[ch_idx], colorscale='greys', showscale=False),
                row=us_orig_row, col=ch_idx + 1
            )
            fig.update_xaxes(title_text="Pulse", row=us_orig_row, col=ch_idx + 1)
            fig.update_yaxes(title_text="Depth" if ch_idx == 0 else "",
                            autorange="reversed", row=us_orig_row, col=ch_idx + 1)

        # US channels - Augmented
        token_aug = sample_aug['token']
        if len(token_aug.shape) == 4:
            token_aug = token_aug[token_aug.shape[0] // 2]

        for ch_idx in range(min(3, token_aug.shape[0])):
            fig.add_trace(
                go.Heatmap(z=token_aug[ch_idx], colorscale='greys',
                          showscale=(ch_idx == 2),
                          colorbar=dict(title="Amp", x=1.02) if ch_idx == 2 else None),
                row=us_aug_row, col=ch_idx + 1
            )
            fig.update_xaxes(title_text="Pulse", row=us_aug_row, col=ch_idx + 1)
            fig.update_yaxes(title_text="Depth" if ch_idx == 0 else "",
                            autorange="reversed", row=us_aug_row, col=ch_idx + 1)

        # Title
        source_info = "Same Source Sequence" if same_source else "Different Sequences"
        fig.update_layout(
            title=dict(
                text=f"Original vs Augmented ({split.upper()}) - {source_info}",
                font=dict(size=16)
            ),
            height=1000 if has_joystick else 800,
            width=1400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        return fig

    def show(self, split='train', seed=None):
        """Show visualization for a random sample."""
        import plotly.io as pio
        pio.renderers.default = 'browser'

        sample = self.load_random_sample(split=split, seed=seed)
        fig = self.visualize_sample(sample)
        fig.show()

    def show_by_class(self, split='train', seed=None):
        """Show 5 separate visualizations, one for each label class."""
        import plotly.io as pio
        pio.renderers.default = 'browser'

        label_names = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}

        for target_label in [0, 1, 2, 3, 4]:
            try:
                sample = self.load_random_sample(split=split, seed=seed, target_label=target_label)
                fig = self.visualize_sample(sample)
                fig.update_layout(
                    title=dict(
                        text=f"Label {target_label} ({label_names[target_label]}) - {split.upper()}",
                        font=dict(size=16)
                    )
                )
                fig.show()
                print(f"Opened figure for Label {target_label} ({label_names[target_label]})")
            except ValueError as e:
                print(f"Warning: {e}")

    def show_compare(self, split='train', seed=None):
        """Show original vs augmented comparison."""
        import plotly.io as pio
        pio.renderers.default = 'browser'

        fig = self.visualize_compare(seed=seed, split=split)
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize token samples from pkl datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualization/token_visualizer_pickle.py
    python visualization/token_visualizer_pickle.py --split train --seed 42
    python visualization/token_visualizer_pickle.py --by-class
    python visualization/token_visualizer_pickle.py --compare
    python visualization/token_visualizer_pickle.py --data /path/to/pkl/folder
        """
    )
    parser.add_argument('--data', '-d', type=str, default=None,
                       help='Path to folder containing pkl files (default: from config)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Which split to visualize (default: train)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sample selection')
    parser.add_argument('--by-class', action='store_true',
                       help='Show 3 separate figures, one for each label class')
    parser.add_argument('--compare', action='store_true',
                       help='Show original vs augmented side-by-side')

    args = parser.parse_args()

    # Find data path - use argument if provided, otherwise from config
    if args.data:
        data_path = args.data
    else:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        config = load_config(config_path, create_dirs=False)
        data_path = config.get_train_data_root()

    print(f"Using data path: {data_path}")

    if not data_path or not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        return 1

    # Create visualizer
    viz = PickleTokenVisualizer(data_path)

    # Visualize based on mode
    if args.compare:
        viz.show_compare(split=args.split, seed=args.seed)
    elif args.by_class:
        viz.show_by_class(split=args.split, seed=args.seed)
    else:
        viz.show(split=args.split, seed=args.seed)

    return 0


if __name__ == '__main__':
    exit(main())
