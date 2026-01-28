#!/usr/bin/env python3
"""
Token Sample Visualizer

Visualizes a random token sample in context:
- Row 1: Full joystick signal with token window highlighted
- Row 2: Zoomed window view (joystick + derivative + labels)
- Row 3: 3 US channel M-mode images

Usage:
    python visualization/token_visualizer.py [--data-path /path/to/processed/run_folder]
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.configurator import load_config
from preprocessing.label_logic.labeling import (
    create_derivative_labels,
    create_edge_to_peak_labels,
    create_edge_to_derivative_labels
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
        self.label_method = getattr(self.config.preprocess.labels, 'method', 'derivative')
        self.label_threshold = getattr(self.config.preprocess.labels, 'threshold_percent', 5.0)
        self.label_axis = getattr(self.config.preprocess.labels, 'axis', 'x')

        # Load joystick filter config from label_logic config
        label_logic_config_path = os.path.join(project_root, 'preprocessing', 'label_logic', 'config.yaml')
        if os.path.exists(label_logic_config_path):
            with open(label_logic_config_path, 'r') as f:
                label_logic_config = yaml.safe_load(f)
            self.filters_config = label_logic_config.get('filters', {})
        else:
            self.filters_config = {}

        # Check if soft labels
        soft_cfg = getattr(self.config.preprocess.labels, 'soft_labels', None)
        self.soft_labels_enabled = getattr(soft_cfg, 'enabled', False) if soft_cfg else False

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
        Determine the class from label data.

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
            counts = [np.sum(label_flat == i) for i in range(3)]
            return int(np.argmax(counts))

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
        # Find session folder (format: session{N}_W_{participant})
        session_pattern = f"session{int(session_id)}_W_{int(participant_id):03d}"
        session_folders = glob.glob(os.path.join(self.raw_data_path, f"session{int(session_id)}_W_*"))

        if not session_folders:
            raise FileNotFoundError(f"Session folder not found: {session_pattern}")

        session_folder = session_folders[0]
        exp_folder = os.path.join(session_folder, str(int(experiment_id)))

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
        Create per-sample labels from joystick data.
        Applies the same filters as visualize.py for consistency.

        Returns:
            labels, position, derivative, pos_threshold, deriv_threshold, markers
        """
        # Select axis
        if self.label_axis == 'x':
            raw_position = joystick_data[:, 1]
        elif self.label_axis == 'y':
            raw_position = joystick_data[:, 2]
        else:
            raw_position = joystick_data[:, 1]

        # Apply filters to position (same as visualize.py)
        if self.filters_config:
            position = apply_joystick_filters(raw_position.copy(), self.filters_config, 'raw')
        else:
            position = raw_position.copy()

        derivative = np.gradient(position)

        # Apply filters to derivative (same as visualize.py)
        if self.filters_config:
            derivative = apply_joystick_filters(derivative, self.filters_config, 'derivative')

        if self.label_method == 'derivative':
            labels, threshold = create_derivative_labels(derivative, self.label_threshold)
            return labels, position, derivative, threshold, None, None
        elif self.label_method == 'edge_to_peak':
            labels, threshold, markers = create_edge_to_peak_labels(position, derivative, self.label_threshold)
            return labels, position, derivative, threshold, None, markers
        elif self.label_method == 'edge_to_derivative':
            labels, pos_thresh, deriv_thresh, markers = create_edge_to_derivative_labels(position, derivative, self.label_threshold)
            return labels, position, derivative, pos_thresh, deriv_thresh, markers

        return None, position, derivative, None, None, None

    def visualize_sample(self, sample=None, seed=None):
        """
        Create the visualization for a token sample.
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

        # Create labels (with same filtering as visualize.py)
        labels, position, derivative, pos_threshold, deriv_threshold, markers = self.create_labels_from_joystick(joystick)

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{"colspan": 3, "secondary_y": True}, None, None],  # Row 1: full width
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],  # Row 2
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],  # Row 3
            ],
            subplot_titles=[
                f"Full Joystick Signal (Token {sample['sample_idx']}/{sample['num_samples']})",
                "",  # Row 2 col 1
                f"Zoomed Window (pulses {start_pulse}-{end_pulse})",
                "Label Distribution",
                "US Channel 1", "US Channel 2", "US Channel 3"
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            row_heights=[0.35, 0.25, 0.40]
        )

        x_full = np.arange(len(position))

        # ===== ROW 1: Full joystick signal =====
        # Add label regions first (background)
        label_colors = {1: 'rgba(144, 238, 144, 0.2)', 2: 'rgba(240, 128, 128, 0.2)'}

        # Add label regions
        label_colors = {1: 'rgba(144, 238, 144, 0.2)', 2: 'rgba(240, 128, 128, 0.2)'}
        for label_val in [1, 2]:
            mask = labels == label_val
            diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                fig.add_vrect(x0=s, x1=e, fillcolor=label_colors[label_val],
                             layer="below", line_width=0, row=1, col=1)

        # Position trace
        fig.add_trace(
            go.Scatter(x=x_full, y=position, mode='lines',
                      line=dict(color='blue', width=1),
                      name=f'Position ({self.label_axis.upper()})'),
            row=1, col=1, secondary_y=False
        )

        # Derivative trace
        fig.add_trace(
            go.Scatter(x=x_full, y=derivative, mode='lines',
                      line=dict(color='red', width=1), opacity=0.7,
                      name='Derivative'),
            row=1, col=1, secondary_y=True
        )

        # Position threshold lines (for edge-based methods: on position trace)
        if pos_threshold is not None:
            is_position_threshold = (self.label_method in ['edge_to_peak', 'edge_to_derivative'])
            thresh_color = 'blue' if is_position_threshold else 'gray'
            fig.add_hline(y=pos_threshold, line=dict(color=thresh_color, dash='dash', width=1),
                         opacity=0.5, row=1, col=1, secondary_y=False)
            fig.add_hline(y=-pos_threshold, line=dict(color=thresh_color, dash='dash', width=1),
                         opacity=0.5, row=1, col=1, secondary_y=False)

        # Derivative threshold lines (for edge_to_derivative method)
        if deriv_threshold is not None:
            fig.add_hline(y=deriv_threshold, line=dict(color='orange', dash='dash', width=1),
                         opacity=0.5, row=1, col=1, secondary_y=True)
            fig.add_hline(y=-deriv_threshold, line=dict(color='orange', dash='dash', width=1),
                         opacity=0.5, row=1, col=1, secondary_y=True)

        # Highlight token window with prominent cyan box (on top of everything)
        fig.add_vrect(
            x0=start_pulse, x1=end_pulse,
            fillcolor='rgba(0, 255, 255, 0.25)',
            layer="above",
            line=dict(color='cyan', width=3),
            row=1, col=1
        )

        # Add vertical lines at window boundaries for extra visibility
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

        # Add scatter markers for edge/peak/derivative crossings (same as visualize.py)
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

        # ===== ROW 2, COL 1: Info box =====
        info_text = (
            f"<b>Sample Info</b><br>"
            f"Session: {sample['session_id']}<br>"
            f"Participant: {sample['participant_id']}<br>"
            f"Experiment: {sample['experiment_id']}<br>"
            f"Token Index: {sample['sample_idx']}<br>"
            f"Window: {self.token_window} pulses<br>"
            f"Stride: {self.token_stride} pulses<br>"
            f"Label Method: {self.label_method}"
        )
        fig.add_annotation(
            x=0.5, y=0.5, text=info_text,
            showarrow=False, font=dict(size=11),
            xref="x2 domain", yref="y2 domain",
            align="left", bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray", borderwidth=1
        )
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

        # ===== ROW 2, COL 2: Zoomed window =====
        x_zoom = np.arange(start_pulse, end_pulse)
        pos_zoom = position[start_pulse:end_pulse]
        deriv_zoom = derivative[start_pulse:end_pulse]
        labels_zoom = labels[start_pulse:end_pulse]

        # Add label regions in zoomed view
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

        # ===== ROW 2, COL 3: Label distribution =====
        label_data = sample['label']

        # Determine label type based on actual data shape
        # Transformer mode: [seq_window, 1] -> hard labels per sequence position
        # Flat mode soft: [num_classes] -> probability distribution
        # Flat mode hard: [1] or scalar -> single hard label

        label_flat = label_data.flatten()

        if label_flat.size == 1:
            # Single hard label
            label_val = int(label_flat[0])
            label_names = {0: 'Noise', 1: 'Up/Right', 2: 'Down/Left'}
            label_colors_bar = {0: 'gray', 1: 'green', 2: 'red'}
            fig.add_trace(
                go.Bar(x=[label_names.get(label_val, str(label_val))], y=[1],
                      marker_color=label_colors_bar.get(label_val, 'blue'),
                      name='Label', showlegend=False),
                row=2, col=3
            )
            fig.update_yaxes(visible=False, row=2, col=3)
        elif label_data.dtype in [np.float32, np.float64] and np.allclose(label_flat.sum(), 1.0, atol=0.1):
            # Soft labels (probability distribution that sums to ~1)
            class_names = ['Noise', 'Up/Right', 'Down/Left']
            colors = ['gray', 'green', 'red']
            fig.add_trace(
                go.Bar(x=class_names[:len(label_flat)], y=label_flat,
                      marker_color=colors[:len(label_flat)],
                      name='Class Probabilities', showlegend=False),
                row=2, col=3
            )
            fig.update_yaxes(range=[0, 1], title_text="Probability", row=2, col=3)
        else:
            # Sequence of hard labels (transformer mode) - show distribution
            class_names = ['Noise', 'Up/Right', 'Down/Left']
            colors = ['gray', 'green', 'red']
            counts = [np.sum(label_flat == i) for i in range(3)]
            total = sum(counts)
            if total > 0:
                probs = [c / total for c in counts]
            else:
                probs = [0, 0, 0]
            fig.add_trace(
                go.Bar(x=class_names, y=probs,
                      marker_color=colors,
                      name='Label Distribution', showlegend=False),
                row=2, col=3
            )
            fig.update_yaxes(range=[0, 1], title_text="Fraction", row=2, col=3)

        # ===== ROW 3: US Channels =====
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
                row=3, col=ch_idx + 1
            )
            fig.update_xaxes(title_text="Pulse", row=3, col=ch_idx + 1)
            fig.update_yaxes(title_text="Depth", autorange="reversed", row=3, col=ch_idx + 1)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Token Visualization - S{sample['session_id']}_P{sample['participant_id']}_E{sample['experiment_id']}",
                font=dict(size=16)
            ),
            height=900,
            width=1400,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        # Update axes labels
        fig.update_yaxes(title_text="Position", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Derivative", row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Pulse Index", row=1, col=1)

        return fig

    def show(self, seed=None):
        """Show visualization for a random sample in browser."""
        import plotly.io as pio
        pio.renderers.default = 'browser'

        fig = self.visualize_sample(seed=seed)
        fig.show()

    def show_by_class(self, seed=None):
        """
        Show 3 separate visualizations, one for each label class.

        Opens 3 browser tabs with samples from:
        - Label 0 (Noise)
        - Label 1 (Up/Right)
        - Label 2 (Down/Left)
        """
        import plotly.io as pio
        pio.renderers.default = 'browser'

        label_names = {0: 'Noise', 1: 'Up/Right', 2: 'Down/Left'}

        for target_label in [0, 1, 2]:
            try:
                sample = self.load_sample_by_label(target_label, seed=seed)
                fig = self.visualize_sample(sample=sample)

                # Update title to indicate the label class
                fig.update_layout(
                    title=dict(
                        text=f"Label {target_label} ({label_names[target_label]}) - "
                             f"S{sample['session_id']}_P{sample['participant_id']}_E{sample['experiment_id']}",
                        font=dict(size=16)
                    )
                )
                fig.show()
                print(f"Opened figure for Label {target_label} ({label_names[target_label]})")
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
                       help='Show 3 separate figures, one for each label class (0, 1, 2)')

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
