"""
Token Sample Visualizer (Dynamic Multi-Class Version)

Visualizes a random token sample from PROCESSED data (train_base_data_path).
Uses H5 files from a preprocessing run folder to show tokens and their labels.
Also loads corresponding raw joystick data to show full signal context.

Display layout:
- Row 1: X-axis joystick (position + derivative) with label regions
- Row 2: Y-axis joystick (position + derivative) with label regions
- Row 3: Info box, Zoomed window, Label distribution
- Row 4: 3 US channel M-mode images

Class Labels (configurable via label_config.yaml include_noise setting):
- 5-class mode: Noise(0), Up(1), Down(2), Left(3), Right(4)
- 4-class mode: Up(1), Down(2), Left(3), Right(4) - noise excluded

Usage:
    python visualization/token_visualizer_h5.py
    python visualization/token_visualizer_h5.py --seed 42
    python visualization/token_visualizer_h5.py --data-path /path/to/processed/run_folder
    python visualization/token_visualizer_h5.py --config config/config.yaml
    python visualization/token_visualizer_h5.py --by-class              # Show one sample per label class
    python visualization/token_visualizer_h5.py --by-class --seed 42    # Reproducible by-class visualization

Options:
    -d, --data-path   Path to processed data folder (run_YYYYMMDD_HHMMSS or 'latest')
                      If not specified, uses most recent run folder from config
    -c, --config      Path to config.yaml (default: config/config.yaml)
    -s, --seed        Random seed for reproducible sample selection
    --by-class        Show separate figures, one for each active label class
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py
import pandas as pd
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path (go up 3 levels: visualization -> preprocessing -> m-mode_nn)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config
from preprocessing.label_logic.label_logic import (
    create_position_peak_labels,
    create_5class_position_peak_labels
)
from preprocessing.signal_utils import apply_joystick_filters


class TokenVisualizer:
    def __init__(self, processed_data_path, config_path=None):
        """
        Initialize the token visualizer.

        Args:
            processed_data_path: Path to processed data folder (run_YYYYMMDD_HHMMSS)
            config_path: Path to config.yaml (optional, will try to find it)
        """
        self.processed_path = processed_data_path

        # Load config
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        self.config = load_config(config_path)

        # Get raw data path from config
        self.raw_data_path = os.path.join(self.config.preprocess.data.basepath, 'raw')

        # Load metadata
        metadata_path = os.path.join(processed_data_path, 'metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = None
            print(f"Warning: No metadata.csv found at {metadata_path}")

        # Get processing parameters
        self.token_window = self.config.preprocess.tokenization.window
        self.token_stride = self.config.preprocess.tokenization.stride

        # Load label config from label_logic/label_config.yaml
        label_config_path = os.path.join(project_root, 'preprocessing', 'label_logic', 'label_config.yaml')
        with open(label_config_path, 'r') as f:
            label_config = yaml.safe_load(f)

        self.label_method = label_config.get('method', 'position_peak')
        self.label_axis = label_config.get('axis', 'dual')
        self.filters_config = label_config.get('filters', {})

        # Get class configuration
        classes_config = label_config.get('classes', {})
        self.include_noise = classes_config.get('include_noise', True)
        self.num_classes = 5 if self.include_noise else 4
        config_names = classes_config.get('names', {})
        self.class_names = {int(k): v for k, v in config_names.items()} if config_names else {
            0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'
        }
        self.class_colors = ['gray', 'green', 'red', 'blue', 'orange']

        # Position peak config
        pp_config = label_config.get('position_peak', {})
        self.pp_deriv_thresh = pp_config.get('deriv_threshold_percent', 10.0)
        self.pp_pos_thresh = pp_config.get('pos_threshold_percent', 5.0)
        self.pp_peak_window = pp_config.get('peak_window', 3)
        self.pp_timeout = pp_config.get('timeout_samples', 500)

        # Check if soft labels
        soft_cfg = label_config.get('soft_labels', {})
        self.soft_labels_enabled = soft_cfg.get('enabled', False)

        # Find all H5 files (search only in participant subfolders, not recursive)
        self.h5_files = sorted(glob.glob(os.path.join(processed_data_path, 'P*', '*.h5')))
        if not self.h5_files:
            # Fallback: try recursive search
            self.h5_files = sorted(glob.glob(os.path.join(processed_data_path, '**', '*.h5'), recursive=True))

        if not self.h5_files:
            raise FileNotFoundError(f"No H5 files found in {processed_data_path}")

        print(f"Found {len(self.h5_files)} H5 files")

    def load_random_sample(self, seed=None):
        """
        Load a random token sample.

        Returns:
            dict with token data, label, and metadata
        """
        if seed is not None:
            np.random.seed(seed)

        # Pick random H5 file
        h5_path = np.random.choice(self.h5_files)

        with h5py.File(h5_path, 'r') as f:
            tokens = f['token'][:]
            labels = f['label'][:]

        # Pick random token index
        num_samples = len(tokens)
        sample_idx = np.random.randint(0, num_samples)

        token_data = tokens[sample_idx]
        label_data = labels[sample_idx]

        # Parse experiment info from filename
        # Format: S{session}_P{participant}_E{experiment}_Xy.h5
        filename = os.path.basename(h5_path)
        parts = filename.replace('_Xy.h5', '').split('_')
        session_id = parts[0].replace('S', '')
        participant_id = parts[1].replace('P', '')
        experiment_id = parts[2].replace('E', '')

        return {
            'token': token_data,
            'label': label_data,
            'sample_idx': sample_idx,
            'num_samples': num_samples,
            'h5_path': h5_path,
            'session_id': session_id,
            'participant_id': participant_id,
            'experiment_id': experiment_id
        }

    def _get_label_class(self, label_data):
        """
        Determine the class from label data (supports 5-class labels).

        Handles both hard labels (single int) and soft labels (probability distribution).
        """
        label_flat = label_data.flatten()

        if label_flat.size == 1:
            # Hard label
            return int(label_flat[0])
        elif label_data.dtype in [np.float32, np.float64] and np.allclose(label_flat.sum(), 1.0, atol=0.1):
            # Soft labels - return dominant class
            return int(np.argmax(label_flat))
        else:
            # Sequence of hard labels - return most common
            unique_labels = np.unique(label_flat)
            counts = {int(lbl): np.sum(label_flat == lbl) for lbl in unique_labels}
            return int(max(counts, key=counts.get))

    def load_sample_by_label(self, target_label, seed=None):
        """
        Load a random token sample with a specific label class.

        Args:
            target_label: The target label class (0, 1, or 2)
            seed: Random seed for reproducibility

        Returns:
            dict with token data, label, and metadata
        """
        if seed is not None:
            np.random.seed(seed)

        # Shuffle H5 files for randomness
        h5_files_shuffled = np.random.permutation(self.h5_files)

        for h5_path in h5_files_shuffled:
            with h5py.File(h5_path, 'r') as f:
                tokens = f['token'][:]
                labels = f['label'][:]

            # Find indices matching target label
            matching_indices = []
            for idx in range(len(labels)):
                if self._get_label_class(labels[idx]) == target_label:
                    matching_indices.append(idx)

            if matching_indices:
                # Pick random matching sample
                sample_idx = np.random.choice(matching_indices)
                token_data = tokens[sample_idx]
                label_data = labels[sample_idx]

                # Parse experiment info from filename
                filename = os.path.basename(h5_path)
                parts = filename.replace('_Xy.h5', '').split('_')
                session_id = parts[0].replace('S', '')
                participant_id = parts[1].replace('P', '')
                experiment_id = parts[2].replace('E', '')

                return {
                    'token': token_data,
                    'label': label_data,
                    'sample_idx': sample_idx,
                    'num_samples': len(tokens),
                    'h5_path': h5_path,
                    'session_id': session_id,
                    'participant_id': participant_id,
                    'experiment_id': experiment_id
                }

        raise ValueError(f"No samples found with label {target_label}")

    def load_raw_data(self, session_id, participant_id, experiment_id):
        """
        Load raw US and joystick data for an experiment.
        """
        # New hierarchy: P{participant}/session{session}/exp{experiment}
        exp_folder = os.path.join(
            self.raw_data_path,
            f"P{int(participant_id):03d}",
            f"session{int(session_id):03d}",
            f"exp{int(experiment_id):03d}"
        )

        if not os.path.exists(exp_folder):
            raise FileNotFoundError(f"Experiment folder not found: {exp_folder}")

        # Load US channels
        us_data = []
        for ch in [1, 2, 3]:
            us_path = os.path.join(exp_folder, f'_US_ch{ch}.npy')
            if os.path.exists(us_path):
                us_data.append(np.load(us_path))

        # Load joystick
        joy_path = os.path.join(exp_folder, '_joystick.npy')
        joystick_data = np.load(joy_path) if os.path.exists(joy_path) else None

        return {
            'us_channels': us_data,  # List of [pulses, A-mode] arrays
            'joystick': joystick_data,  # [pulses, 4]
            'exp_folder': exp_folder
        }

    def compute_token_window_indices(self, sample_idx, num_pulses):
        """
        Compute the pulse indices for a given token sample index.
        """
        start_pulse = sample_idx * self.token_stride
        end_pulse = start_pulse + self.token_window

        # Clamp to valid range
        start_pulse = max(0, min(start_pulse, num_pulses - self.token_window))
        end_pulse = start_pulse + self.token_window

        return start_pulse, end_pulse

    def create_labels_from_joystick(self, joystick_data):
        """
        Create per-sample 5-class labels from joystick data using both X and Y axes.
        Applies the same filters as the preprocessing pipeline for consistency.

        Returns:
            dict with:
                'labels': merged 5-class labels (0=Noise, 1=Up, 2=Down, 3=Left, 4=Right)
                'x_position', 'x_derivative': filtered X-axis signals
                'y_position', 'y_derivative': filtered Y-axis signals
                'thresholds': dict with pos and deriv thresholds
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

    def visualize_sample(self, sample=None, seed=None):
        """
        Create the visualization for a token sample with 5-class dual-axis display.
        """
        # Load random sample if not provided
        if sample is None:
            sample = self.load_random_sample(seed)

        # Load raw data
        raw_data = self.load_raw_data(
            sample['session_id'],
            sample['participant_id'],
            sample['experiment_id']
        )

        joystick = raw_data['joystick']
        us_channels = raw_data['us_channels']

        # Compute token window
        num_pulses = joystick.shape[0]
        start_pulse, end_pulse = self.compute_token_window_indices(sample['sample_idx'], num_pulses)

        # Create 5-class labels from both axes
        label_result = self.create_labels_from_joystick(joystick)
        labels = label_result['labels']
        x_position = label_result['x_position']
        x_derivative = label_result['x_derivative']
        y_position = label_result['y_position']
        y_derivative = label_result['y_derivative']
        thresholds = label_result['thresholds']
        x_markers = label_result['x_markers']
        y_markers = label_result['y_markers']

        # Create figure with 4 rows
        fig = make_subplots(
            rows=4, cols=3,
            specs=[
                [{"colspan": 3, "secondary_y": True}, None, None],  # Row 1: X-axis full width
                [{"colspan": 3, "secondary_y": True}, None, None],  # Row 2: Y-axis full width
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],  # Row 3
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],  # Row 4: US
            ],
            subplot_titles=[
                f"X-Axis (Left/Right) - Token {sample['sample_idx']}/{sample['num_samples']}",
                "Y-Axis (Up/Down)",
                "",  # Info box
                f"Zoomed Window (pulses {start_pulse}-{end_pulse})",
                "Label Distribution",
                "US Channel 1", "US Channel 2", "US Channel 3"
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.05,
            row_heights=[0.22, 0.22, 0.20, 0.36]
        )

        x_full = np.arange(len(labels))

        # 5-class label colors
        label_colors_5class = {
            0: 'rgba(128, 128, 128, 0.15)',  # Noise - gray
            1: 'rgba(0, 200, 0, 0.25)',       # Up - green
            2: 'rgba(220, 50, 50, 0.25)',     # Down - red
            3: 'rgba(50, 100, 220, 0.25)',    # Left - blue
            4: 'rgba(255, 165, 0, 0.25)',     # Right - orange
        }
        label_names_5class = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}

        # Helper to add label regions to a row
        def add_label_regions(row):
            for label_val in [1, 2, 3, 4]:  # Skip noise (0) for cleaner display
                mask = labels == label_val
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

        # ===== ROW 1: X-axis (Left/Right) =====
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

        # X thresholds (from 5-class function)
        x_pos_thresh = thresholds.get('x_pos')
        y_pos_thresh = thresholds.get('y_pos')
        if x_pos_thresh is not None:
            fig.add_hline(y=x_pos_thresh, line=dict(color='blue', dash='dash', width=1),
                         opacity=0.4, row=1, col=1, secondary_y=False)
            fig.add_hline(y=-x_pos_thresh, line=dict(color='blue', dash='dash', width=1),
                         opacity=0.4, row=1, col=1, secondary_y=False)

        add_axis_markers(1, x_position, x_derivative, x_markers, 'X')

        # Token window highlight on row 1
        fig.add_vrect(x0=start_pulse, x1=end_pulse, fillcolor='rgba(0, 255, 255, 0.2)',
                     layer="above", line=dict(color='cyan', width=2), row=1, col=1)

        # ===== ROW 2: Y-axis (Up/Down) =====
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

        # ===== ROW 3, COL 1: Info box =====
        info_text = (
            f"<b>Sample Info</b><br>"
            f"Session: {sample['session_id']}<br>"
            f"Participant: {sample['participant_id']}<br>"
            f"Experiment: {sample['experiment_id']}<br>"
            f"Token Index: {sample['sample_idx']}<br>"
            f"Window: {self.token_window} pulses<br>"
            f"Stride: {self.token_stride} pulses<br>"
            f"Label Method: {self.label_method}<br>"
            f"<b>5-Class Dual-Axis</b>"
        )
        fig.add_annotation(
            x=0.5, y=0.5, text=info_text,
            showarrow=False, font=dict(size=10),
            xref="x3 domain", yref="y3 domain",
            align="left", bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray", borderwidth=1
        )
        fig.update_xaxes(visible=False, row=3, col=1)
        fig.update_yaxes(visible=False, row=3, col=1)

        # ===== ROW 3, COL 2: Zoomed window (X-axis) =====
        x_zoom = np.arange(start_pulse, end_pulse)
        x_pos_zoom = x_position[start_pulse:end_pulse]
        y_pos_zoom = y_position[start_pulse:end_pulse]
        labels_zoom = labels[start_pulse:end_pulse]

        # Add label regions in zoomed view
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

        # ===== ROW 3, COL 3: Label distribution =====
        label_data = sample['label']
        label_flat = label_data.flatten()

        # Get class names and colors dynamically
        all_class_names = [self.class_names.get(i, f'Class {i}') for i in range(5)]
        all_colors = self.class_colors

        if label_flat.size == 1:
            # Single hard label
            label_val = int(label_flat[0])
            cls_name = self.class_names.get(label_val, str(label_val))
            cls_color = all_colors[label_val] if label_val < len(all_colors) else 'purple'
            fig.add_trace(
                go.Bar(x=[cls_name], y=[1],
                      marker_color=cls_color,
                      name='Label', showlegend=False),
                row=3, col=3
            )
            fig.update_yaxes(visible=False, row=3, col=3)
        elif label_data.dtype in [np.float32, np.float64] and np.allclose(label_flat.sum(), 1.0, atol=0.1):
            # Soft labels (probability distribution)
            n_classes = len(label_flat)
            fig.add_trace(
                go.Bar(x=all_class_names[:n_classes], y=label_flat,
                      marker_color=all_colors[:n_classes],
                      name='Class Probabilities', showlegend=False),
                row=3, col=3
            )
            fig.update_yaxes(range=[0, 1], title_text="Probability", row=3, col=3)
        else:
            # Sequence of hard labels - show distribution
            unique_labels = np.unique(label_flat)
            max_label = int(max(unique_labels)) + 1
            counts = [np.sum(label_flat == i) for i in range(max_label)]
            total = sum(counts)
            if total > 0:
                probs = [c / total for c in counts]
            else:
                probs = [0] * max_label
            fig.add_trace(
                go.Bar(x=all_class_names[:max_label], y=probs,
                      marker_color=all_colors[:max_label],
                      name='Label Distribution', showlegend=False),
                row=3, col=3
            )
            fig.update_yaxes(range=[0, 1], title_text="Fraction", row=3, col=3)

        # ===== ROW 4: US Channels =====
        token_us = sample['token']  # Shape: [C, H, W] or [seq, C, H, W]

        # Handle different shapes
        if len(token_us.shape) == 4:
            # Transformer mode: [seq, C, H, W] - take middle token
            mid_idx = token_us.shape[0] // 2
            token_us = token_us[mid_idx]

        # token_us is now [C, H, W]
        for ch_idx in range(min(3, token_us.shape[0])):
            ch_data = token_us[ch_idx]  # [H, W]

            fig.add_trace(
                go.Heatmap(
                    z=ch_data,
                    colorscale='greys',
                    showscale=(ch_idx == 2),
                    colorbar=dict(title="Amplitude", x=1.02) if ch_idx == 2 else None
                ),
                row=4, col=ch_idx + 1
            )
            fig.update_xaxes(title_text="Pulse", row=4, col=ch_idx + 1)
            fig.update_yaxes(title_text="Depth", autorange="reversed", row=4, col=ch_idx + 1)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"5-Class Token Visualization - S{sample['session_id']}_P{sample['participant_id']}_E{sample['experiment_id']}",
                font=dict(size=16)
            ),
            height=1100,
            width=1400,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        # Update axes labels and fix x-axis range to start at 0 (autorange=False forces explicit range)
        num_samples = len(labels)
        fig.update_yaxes(title_text="X Position", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="X Derivative", row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Pulse Index", range=[0, num_samples], autorange=False, row=1, col=1)

        fig.update_yaxes(title_text="Y Position", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Y Derivative", row=2, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Pulse Index", range=[0, num_samples], autorange=False, row=2, col=1)

        return fig

    def show(self, seed=None):
        """Show visualization for a random sample in browser."""
        import plotly.io as pio
        pio.renderers.default = 'browser'

        fig = self.visualize_sample(seed=seed)
        fig.show()

    def show_by_class(self, seed=None):
        """
        Show separate visualizations, one for each label class.

        Opens browser tabs with samples from each active class.
        Number of classes depends on include_noise setting in label_config.yaml:
        - 5 classes: Noise(0), Up(1), Down(2), Left(3), Right(4)
        - 4 classes: Up(1), Down(2), Left(3), Right(4)
        """
        import plotly.io as pio
        pio.renderers.default = 'browser'

        # Determine which labels to show based on include_noise setting
        if self.include_noise:
            target_labels = [0, 1, 2, 3, 4]
        else:
            target_labels = [1, 2, 3, 4]

        for target_label in target_labels:
            try:
                sample = self.load_sample_by_label(target_label, seed=seed)
                fig = self.visualize_sample(sample=sample)

                # Update title to indicate the label class
                label_name = self.class_names.get(target_label, f'Class {target_label}')
                fig.update_layout(
                    title=dict(
                        text=f"Label {target_label} ({label_name}) - "
                             f"S{sample['session_id']}_P{sample['participant_id']}_E{sample['experiment_id']}",
                        font=dict(size=16)
                    )
                )
                fig.show()
                print(f"Opened figure for Label {target_label} ({label_name})")
            except ValueError as e:
                print(f"Warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize token samples")
    parser.add_argument('--data-path', '-d', type=str, default=None,
                       help='Path to processed data folder (run_YYYYMMDD_HHMMSS)')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible sample selection')
    parser.add_argument('--by-class', action='store_true',
                       help='Show separate figures, one for each active label class')

    args = parser.parse_args()

    # Find data path
    if args.data_path is None:
        # Try to find most recent run folder
        config = load_config(args.config or os.path.join(project_root, 'config', 'config.yaml'))
        processed_base = os.path.join(config.preprocess.data.basepath, 'processed')

        # Find most recent run folder (non-recursive, faster)
        run_folders = []
        for dataset_dir in glob.glob(os.path.join(processed_base, '*')):
            if os.path.isdir(dataset_dir):
                for param_dir in glob.glob(os.path.join(dataset_dir, '*')):
                    if os.path.isdir(param_dir):
                        runs = glob.glob(os.path.join(param_dir, 'run_*'))
                        run_folders.extend(runs)

        run_folders = sorted(run_folders)
        if run_folders:
            args.data_path = run_folders[-1]
            print(f"Using most recent run: {args.data_path}")
        else:
            print("Error: No processed data found. Specify --data-path")
            return 1

    # Create visualizer and show
    viz = TokenVisualizer(args.data_path, args.config)

    if args.by_class:
        viz.show_by_class(seed=args.seed)
    else:
        viz.show(seed=args.seed)

    return 0


if __name__ == '__main__':
    exit(main())
