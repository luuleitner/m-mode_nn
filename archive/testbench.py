"""
Testbench for M-mode Ultrasound Dataset Visualization and Split Validation.

This module provides utilities for testing and visualizing the FilteredSplitH5Dataset:

- **Visualization**: Grid-based plotting of single-channel and multi-channel M-mode
  ultrasound samples with automatic layout optimization.
- **Split Testing**: Comprehensive testing of train/test/validation dataset splits
  with support for session, participant, experiment, and label_logic-based filtering.
- **Data Leakage Validation**: Ensures no overlap between dataset splits at both
  sequence and experiment levels.
- **Analysis Output**: Saves plots with metadata and detailed statistics to
  timestamped directories for reproducibility.

Main Functions:
    test_simple_grid: Visualize single-channel samples in an optimal grid layout.
    test_multichannel_grid: Visualize multiple channels per sample with configurable
        arrangements (horizontal, vertical, square).
    comprehensive_dataset_test: Full dataset testing with filtering, splitting,
        and visualization.
    validate_no_data_leakage: Verify train/test/val splits have no overlapping data.
    quick_split_test: Fast verification of splitting logic without plotting.

Usage:
    Run as a script to execute the default test scenarios, or import functions
    for custom testing workflows.
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import random
from pathlib import Path
import gc
import math
import os
from datetime import datetime

# PyCharm plot configuration
matplotlib.use('TkAgg')
plt.ion()


def create_analysis_directory(data_root, run_name="analyses"):
    """
    Create timestamped analysis directory in dataset folder.

    Args:
        data_root: Root path of the dataset
        run_name: Name of the run folder (default: "analyses")

    Returns:
        Path: Path to the created timestamped directory
    """
    if data_root is None:
        # Fallback to current directory if no data_root provided
        data_root = Path.cwd()
    else:
        data_root = Path(data_root)

    # Create run folder if it doesn't exist
    run_folder = data_root / "run"
    run_folder.mkdir(exist_ok=True)

    # Create timestamped analyses folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = run_folder / f"{run_name}_{timestamp}"
    analysis_folder.mkdir(exist_ok=True)

    print(f"Created analysis directory: {analysis_folder}")
    return analysis_folder


def save_plot_with_metadata(fig, plot_name, analysis_dir, plot_info=None):
    """
    Save plot with metadata file describing the analysis.

    Args:
        fig: Matplotlib figure object
        plot_name: Base name for the plot file
        analysis_dir: Directory to save in
        plot_info: Dictionary with plot metadata
    """
    analysis_dir = Path(analysis_dir)

    # Save the plot
    plot_path = analysis_dir / f"{plot_name}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")

    # Save metadata file
    if plot_info:
        metadata_path = analysis_dir / f"{plot_name}_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Plot Analysis Metadata\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Plot file: {plot_name}.png\n\n")

            for key, value in plot_info.items():
                f.write(f"{key}: {value}\n")

        print(f"Metadata saved: {metadata_path}")

    return plot_path


def calculate_optimal_grid(n_items, max_cols=4, prefer_square=True):
    """
    Calculate optimal grid dimensions for given number of items.

    Args:
        n_items: Number of items to arrange
        max_cols: Maximum number of columns allowed
        prefer_square: Whether to prefer square-like arrangements

    Returns:
        tuple: (rows, cols) for optimal arrangement
    """
    if n_items <= 0:
        return (1, 1)

    if n_items == 1:
        return (1, 1)
    elif n_items <= 2:
        return (1, 2)
    elif n_items <= 4:
        return (2, 2)
    elif n_items <= 6:
        return (2, 3)
    elif n_items <= 9:
        return (3, 3)
    elif n_items <= 12:
        return (3, 4)
    else:
        # For larger numbers, calculate based on preference
        if prefer_square:
            cols = min(int(np.ceil(np.sqrt(n_items))), max_cols)
            rows = int(np.ceil(n_items / cols))
        else:
            cols = min(max_cols, n_items)
            rows = int(np.ceil(n_items / cols))

        return (rows, cols)


def validate_grid_parameters(n_samples, n_channels_per_sample, max_samples=16, max_channels=4):
    """
    Validate and constrain grid parameters to reasonable limits.

    Args:
        n_samples: Requested number of samples
        n_channels_per_sample: Requested channels per sample
        max_samples: Maximum allowed samples
        max_channels: Maximum allowed channels per sample

    Returns:
        tuple: (validated_samples, validated_channels)
    """
    # Constrain to reasonable limits
    validated_samples = min(max(1, n_samples), max_samples)
    validated_channels = min(max(1, n_channels_per_sample), max_channels)

    # Warn if values were changed
    if validated_samples != n_samples:
        print(f"Warning: n_samples constrained from {n_samples} to {validated_samples}")
    if validated_channels != n_channels_per_sample:
        print(f"Warning: n_channels_per_sample constrained from {n_channels_per_sample} to {validated_channels}")

    return validated_samples, validated_channels


def test_simple_grid(dataset, n_samples=4, channel_idx=0, max_samples=16, analysis_dir=None):
    """
    Simple grid: One channel per sample in optimal arrangement.

    Args:
        dataset: Dataset instance
        n_samples: Number of samples to plot
        channel_idx: Which channel to display (single integer)
        max_samples: Maximum samples allowed
        analysis_dir: Directory to save plots (if None, creates new timestamped dir)
    """
    print(f"=== SIMPLE GRID: {n_samples} samples, channel {channel_idx} ===")

    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Warning: Dataset is empty, cannot create plot")
        return None, analysis_dir

    # Create analysis directory if not provided
    if analysis_dir is None:
        data_root = getattr(dataset, 'data_root', None)
        analysis_dir = create_analysis_directory(data_root)

    # Validate parameters
    n_samples = min(max(1, n_samples), max_samples, len(dataset))

    # Calculate optimal grid
    rows, cols = calculate_optimal_grid(n_samples)
    print(f"Grid layout: {rows} rows × {cols} columns")

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    # Add split information to title
    split_info = dataset.get_split_info()
    split_type = split_info.get('current_split', 'unknown')
    fig.suptitle(f'Simple Grid [{split_type.upper()}]: {n_samples} samples, Channel {channel_idx}', fontsize=14)

    # Get random samples
    random_batch_indices = random.sample(range(len(dataset)), n_samples)

    # Store plot metadata
    plot_info = {
        'plot_type': 'simple_grid',
        'dataset_split': split_type,
        'n_samples': n_samples,
        'channel_idx': channel_idx,
        'grid_dimensions': f"{rows}x{cols}",
        'batch_indices': random_batch_indices,
        'dataset_size': len(dataset)
    }

    # Add split information to metadata
    plot_info.update(split_info)

    for i, batch_idx in enumerate(random_batch_indices):
        row = i // cols
        col = i % cols

        try:
            # Load data
            batch_data = dataset[batch_idx]
            batch_metadata = dataset.get_batch_metadata(batch_idx)
            seq_idx = random.randint(0, len(batch_data) - 1)
            sequence = batch_data[seq_idx]
            seq_metadata = batch_metadata.iloc[seq_idx]

            # Plot single channel
            channel_data = sequence[:, channel_idx, :, :].numpy() if hasattr(sequence, 'numpy') else sequence[:,
                                                                                                     channel_idx, :, :]
            raster_data = channel_data.reshape(sequence.shape[0], -1)

            # Memory constraint
            if raster_data.shape[1] > 800:
                raster_data = raster_data[:, :800]

            im = axes[row, col].imshow(raster_data.T, aspect='auto', cmap='viridis', origin='lower')

            # Labels
            participant = seq_metadata.get('participant', 'N/A')
            session = seq_metadata.get('session', 'N/A')
            axes[row, col].set_title(f'B{batch_idx}_S{seq_idx}\nP{participant}_Sess{session}', fontsize=8)
            axes[row, col].set_xlabel('Time', fontsize=7)
            axes[row, col].set_ylabel('Spatial', fontsize=7)
            axes[row, col].tick_params(labelsize=6)

            # Colorbar
            cbar = plt.colorbar(im, ax=axes[row, col], shrink=0.7)
            cbar.ax.tick_params(labelsize=5)

        except Exception as e:
            print(f"Error plotting sample {i}: {e}")
            axes[row, col].text(0.5, 0.5, f'Error loading\nBatch {batch_idx}',
                                ha='center', va='center', transform=axes[row, col].transAxes)

    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # Save with metadata
    plot_name = f"simple_grid_{split_type}_{n_samples}samples_ch{channel_idx}"
    save_plot_with_metadata(fig, plot_name, analysis_dir, plot_info)

    return fig, analysis_dir


def test_multichannel_grid(dataset, n_samples=4, channels_to_plot=None,
                           channel_arrangement='horizontal', max_samples=9, max_channels=4, analysis_dir=None):
    """
    Multi-channel grid: Multiple channels per sample with automatic layout.

    Args:
        dataset: Dataset instance
        n_samples: Number of samples to plot
        channels_to_plot: List of channel indices [0,1,2] or None for auto
        channel_arrangement: 'horizontal', 'vertical', or 'square'
        max_samples: Maximum samples allowed
        max_channels: Maximum channels per sample allowed
        analysis_dir: Directory to save plots (if None, creates new timestamped dir)
    """
    print(f"=== MULTI-CHANNEL GRID: {n_samples} samples, {channel_arrangement} arrangement ===")

    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Warning: Dataset is empty, cannot create plot")
        return None, analysis_dir

    # Create analysis directory if not provided
    if analysis_dir is None:
        data_root = getattr(dataset, 'data_root', None)
        analysis_dir = create_analysis_directory(data_root)

    # Validate and determine parameters
    n_samples = min(max(1, n_samples), max_samples, len(dataset))

    # Determine channels to plot
    if channels_to_plot is None:
        sample_seq = dataset[0][0]  # Get sample to check available channels
        available_channels = sample_seq.shape[1]
        channels_to_plot = list(range(min(available_channels, max_channels)))
    else:
        channels_to_plot = channels_to_plot[:max_channels]

    n_channels = len(channels_to_plot)
    print(f"Plotting channels: {channels_to_plot}")

    # Calculate sample grid (for arranging different samples)
    sample_rows, sample_cols = calculate_optimal_grid(n_samples, max_cols=3)

    # Calculate channel subgrid (for arranging channels within each sample)
    if channel_arrangement == 'horizontal':
        ch_rows, ch_cols = 1, n_channels
    elif channel_arrangement == 'vertical':
        ch_rows, ch_cols = n_channels, 1
    elif channel_arrangement == 'square':
        ch_rows, ch_cols = calculate_optimal_grid(n_channels)
    else:
        ch_rows, ch_cols = 1, n_channels  # Default to horizontal

    print(f"Sample grid: {sample_rows}×{sample_cols}")
    print(f"Channel subgrid per sample: {ch_rows}×{ch_cols}")

    # Calculate total subplot grid
    total_rows = sample_rows * ch_rows
    total_cols = sample_cols * ch_cols

    # Create figure
    fig_width = total_cols * 3
    fig_height = total_rows * 2.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Add split information to title
    split_info = dataset.get_split_info()
    split_type = split_info.get('current_split', 'unknown')
    fig.suptitle(
        f'Multi-channel Grid [{split_type.upper()}]: {n_samples} samples × {n_channels} channels ({channel_arrangement})',
        fontsize=14)

    # Get random samples
    random_batch_indices = random.sample(range(len(dataset)), n_samples)

    # Store plot metadata
    plot_info = {
        'plot_type': 'multichannel_grid',
        'dataset_split': split_type,
        'n_samples': n_samples,
        'channels_to_plot': channels_to_plot,
        'channel_arrangement': channel_arrangement,
        'sample_grid': f"{sample_rows}x{sample_cols}",
        'channel_subgrid': f"{ch_rows}x{ch_cols}",
        'total_grid': f"{total_rows}x{total_cols}",
        'batch_indices': random_batch_indices,
        'dataset_size': len(dataset)
    }

    # Add split information to metadata
    plot_info.update(split_info)

    for sample_idx, batch_idx in enumerate(random_batch_indices):
        # Calculate sample position in grid
        sample_row = sample_idx // sample_cols
        sample_col = sample_idx % sample_cols

        try:
            # Load data
            batch_data = dataset[batch_idx]
            batch_metadata = dataset.get_batch_metadata(batch_idx)
            seq_idx = random.randint(0, len(batch_data) - 1)
            sequence = batch_data[seq_idx]
            seq_metadata = batch_metadata.iloc[seq_idx]

            print(f"Sample {sample_idx + 1}: Batch {batch_idx}, Seq {seq_idx}")

            # Plot each channel in the subgrid
            for ch_idx_pos, ch_idx in enumerate(channels_to_plot):
                # Calculate channel position within sample subgrid
                ch_row = ch_idx_pos // ch_cols if ch_cols > 0 else 0
                ch_col = ch_idx_pos % ch_cols if ch_cols > 0 else ch_idx_pos

                # Calculate absolute subplot position
                abs_row = sample_row * ch_rows + ch_row
                abs_col = sample_col * ch_cols + ch_col
                subplot_idx = abs_row * total_cols + abs_col + 1

                ax = fig.add_subplot(total_rows, total_cols, subplot_idx)

                # Get channel data
                channel_data = sequence[:, ch_idx, :, :].numpy() if hasattr(sequence, 'numpy') else sequence[:, ch_idx,
                                                                                                    :, :]
                raster_data = channel_data.reshape(sequence.shape[0], -1)

                # Memory safety
                if raster_data.shape[1] > 600:
                    raster_data = raster_data[:, :600]

                im = ax.imshow(raster_data.T, aspect='auto', cmap='viridis', origin='lower')

                # Formatting
                participant = seq_metadata.get('participant', 'N/A')
                session = seq_metadata.get('session', 'N/A')

                if ch_idx_pos == 0:  # First channel gets sample info
                    ax.set_title(f'S{sample_idx + 1}: P{participant}_Sess{session}\nCh{ch_idx}', fontsize=8)
                else:
                    ax.set_title(f'Ch{ch_idx}', fontsize=8)

                # Labels only on edges
                if abs_row == total_rows - 1:  # Bottom row
                    ax.set_xlabel('Time', fontsize=7)
                if abs_col == 0:  # Left column
                    ax.set_ylabel('Spatial', fontsize=7)

                ax.tick_params(labelsize=6)

                # Compact colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(labelsize=5)

        except Exception as e:
            print(f"Error plotting sample {sample_idx}: {e}")
            # Create error subplot for first channel position
            sample_row = sample_idx // sample_cols
            sample_col = sample_idx % sample_cols
            abs_row = sample_row * ch_rows
            abs_col = sample_col * ch_cols
            subplot_idx = abs_row * total_cols + abs_col + 1
            ax = fig.add_subplot(total_rows, total_cols, subplot_idx)
            ax.text(0.5, 0.5, f'Error loading\nBatch {batch_idx}',
                    ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # Save with metadata
    plot_name = f"multichannel_grid_{split_type}_{n_samples}samples_{n_channels}channels_{channel_arrangement}"
    save_plot_with_metadata(fig, plot_name, analysis_dir, plot_info)
    print(f"Total subplots: {n_samples * n_channels}")

    return fig, analysis_dir


def comprehensive_dataset_test(metadata_file, data_root=None, target_batch_size=200,
                               # Test/Val filtering parameters
                               test_val_session_filter=None,
                               test_val_participant_filter=None,
                               test_val_experiment_filter=None,
                               test_val_label_filter=None,
                               test_val_split_ratio=0.5,
                               split_level='sequence',
                               # Global filtering parameters
                               global_session_filter=None,
                               global_participant_filter=None,
                               global_experiment_filter=None,
                               global_label_filter=None,
                               # Other parameters
                               enable_plotting=True,
                               analysis_dir=None,
                               random_seed=42):
    """
    Comprehensive dataset test with the new FilteredSplitH5Dataset structure.

    Args:
        test_val_*_filter: Filters to identify data for test/validation splits
        test_val_split_ratio: How to split filtered data between test/val
        split_level: 'sequence' or 'experiment' level splitting
        global_*_filter: Global filters applied to all data before splitting
        analysis_dir: Custom analysis directory (if None, creates timestamped dir in dataset)
    """
    print("=== COMPREHENSIVE FILTERED DATASET TEST ===\n")

    # 1. Test metadata loading
    print("1. Testing metadata loading...")
    df = test_metadata_loading(metadata_file)
    if df is None:
        return None

    # 2. Create datasets for all splits
    print("\n2. Creating filtered split datasets...")
    try:
        # Import from your actual module name - adjust this to match your file
        # Option 1: If you saved the FilteredSplitH5Dataset in a file called 'loader.py'
        from src.data.datasets import create_filtered_split_datasets, FilteredSplitH5Dataset

        # Option 2: If you saved it in a different file, update the import accordingly
        # from your_actual_filename import create_filtered_split_datasets

        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            metadata_file=metadata_file,
            target_batch_size=target_batch_size,
            data_root=data_root,
            test_val_session_filter=test_val_session_filter,
            test_val_participant_filter=test_val_participant_filter,
            test_val_experiment_filter=test_val_experiment_filter,
            test_val_label_filter=test_val_label_filter,
            test_val_split_ratio=test_val_split_ratio,
            split_level=split_level,
            global_session_filter=global_session_filter,
            global_participant_filter=global_participant_filter,
            global_experiment_filter=global_experiment_filter,
            global_label_filter=global_label_filter,
            random_seed=random_seed,
            shuffle_experiments=True,
            shuffle_sequences=True
        )

        # Store data_root in datasets for later use
        for ds in [train_ds, test_ds, val_ds]:
            if not hasattr(ds, 'data_root'):
                ds.data_root = data_root

        # 3. Create analysis directory
        if analysis_dir is None:
            analysis_dir = create_analysis_directory(data_root, "split_analyses")

        # 4. Save comprehensive statistics
        print("\n3. Saving dataset statistics...")
        save_split_statistics(train_ds, test_ds, val_ds, analysis_dir, metadata_file,
                              test_val_session_filter, test_val_split_ratio, split_level)

        # 5. Test each non-empty dataset
        datasets_to_test = []
        if len(train_ds) > 0:
            datasets_to_test.append(('train', train_ds))
        if len(test_ds) > 0:
            datasets_to_test.append(('test', test_ds))
        if len(val_ds) > 0:
            datasets_to_test.append(('val', val_ds))

        if not datasets_to_test:
            print("Warning: All datasets are empty!")
            return None

        # 6. Plotting examples
        if enable_plotting:
            print(f"\n4. Testing plotting configurations for {len(datasets_to_test)} non-empty datasets...")

            for split_name, dataset in datasets_to_test:
                print(f"\n--- Testing {split_name.upper()} dataset ---")

                # Simple grid example
                print(f"4a. Simple grid for {split_name} (4 samples, channel 0):")
                test_simple_grid(dataset, n_samples=min(4, len(dataset)),
                                 channel_idx=0, analysis_dir=analysis_dir)

                # Multi-channel horizontal example (if dataset has enough data)
                if len(dataset) >= 2:
                    print(f"4b. Multi-channel for {split_name} (2 samples, 3 channels):")
                    test_multichannel_grid(dataset, n_samples=2, channels_to_plot=[0, 1, 2],
                                           channel_arrangement='horizontal', analysis_dir=analysis_dir)

        return (train_ds, test_ds, val_ds), analysis_dir

    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_split_statistics(train_ds, test_ds, val_ds, analysis_dir, metadata_file,
                          test_val_session_filter, test_val_split_ratio, split_level):
    """Save comprehensive statistics for all dataset splits"""

    stats_file = Path(analysis_dir) / "split_dataset_statistics.txt"

    with open(stats_file, 'w') as f:
        f.write("Filtered Split Dataset Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metadata file: {metadata_file}\n")
        f.write(f"Test/Val session filter: {test_val_session_filter}\n")
        f.write(f"Test/Val split ratio: {test_val_split_ratio}\n")
        f.write(f"Split level: {split_level}\n\n")

        # Write statistics for each split
        for split_name, dataset in [('TRAIN', train_ds), ('TEST', test_ds), ('VAL', val_ds)]:
            f.write(f"{split_name} Dataset:\n")
            f.write("-" * 20 + "\n")

            if len(dataset) == 0:
                f.write("  Empty dataset\n\n")
                continue

            split_info = dataset.get_split_info()

            f.write(f"  Total sequences: {split_info.get('current_split_size', 0)}\n")
            f.write(f"  Total batches: {len(dataset)}\n")
            f.write(f"  Unique experiments: {split_info.get(f'{split_name.lower()}_experiments', 0)}\n")
            f.write(f"  Unique participants: {split_info.get(f'{split_name.lower()}_participants', 0)}\n")
            f.write(f"  Unique sessions: {split_info.get(f'{split_name.lower()}_sessions', 0)}\n")

            # Show session distribution
            if len(dataset.metadata) > 0 and 'session' in dataset.metadata.columns:
                sessions = sorted(dataset.metadata['session'].unique())
                f.write(f"  Sessions: {sessions}\n")

            f.write("\n")

        # Overall summary
        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        total_train = len(train_ds.metadata) if len(train_ds) > 0 else 0
        total_test = len(test_ds.metadata) if len(test_ds) > 0 else 0
        total_val = len(val_ds.metadata) if len(val_ds) > 0 else 0
        total_all = total_train + total_test + total_val

        if total_all > 0:
            f.write(f"Total sequences: {total_all}\n")
            f.write(f"Train: {total_train} ({100 * total_train / total_all:.1f}%)\n")
            f.write(f"Test: {total_test} ({100 * total_test / total_all:.1f}%)\n")
            f.write(f"Val: {total_val} ({100 * total_val / total_all:.1f}%)\n")

    print(f"Split statistics saved: {stats_file}")


def test_metadata_loading(metadata_file):
    """Test metadata loading (no plotting)"""
    print(f"Testing metadata file: {metadata_file}")

    try:
        df = pd.read_csv(metadata_file)
        print("Metadata columns:", df.columns.tolist())
        print("Metadata shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())

        file_path_col = 'file path' if 'file path' in df.columns else 'file_path'

        print(f"\nUnique experiments: {df[file_path_col].nunique()}")
        print(f"Unique participants: {df['participant'].nunique()}")
        print(f"Unique sessions: {df['session'].nunique()}")
        print(f"Session values: {[int(x) for x in sorted(df['session'].unique())]}")

        return df

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def demo_all_split_layouts(train_ds, test_ds, val_ds, analysis_dir=None):
    """
    Demonstrate all layout options for each dataset split.
    """
    print("=== DEMONSTRATING ALL LAYOUT OPTIONS FOR SPLITS ===")

    # Create analysis directory if not provided
    if analysis_dir is None:
        data_root = getattr(train_ds, 'data_root', None) or getattr(test_ds, 'data_root', None) or getattr(val_ds,
                                                                                                           'data_root',
                                                                                                           None)
        analysis_dir = create_analysis_directory(data_root, "demo_layouts")

    splits_to_demo = []
    if len(train_ds) > 0:
        splits_to_demo.append(('train', train_ds))
    if len(test_ds) > 0:
        splits_to_demo.append(('test', test_ds))
    if len(val_ds) > 0:
        splits_to_demo.append(('val', val_ds))

    for split_name, dataset in splits_to_demo:
        print(f"\n=== {split_name.upper()} DATASET LAYOUTS ===")

        print(f"\n1. Simple grids with different sample counts for {split_name}:")
        max_samples_to_test = min(9, len(dataset))  # Don't test more than available
        sample_counts = [2, 4, 6, 9]
        sample_counts = [n for n in sample_counts if n <= max_samples_to_test]

        for n_samples in sample_counts:
            print(f"\n--- {n_samples} samples for {split_name} ---")
            test_simple_grid(dataset, n_samples=n_samples, channel_idx=0, analysis_dir=analysis_dir)
            plt.pause(1.0)

        if len(dataset) >= 2:  # Need at least 2 samples for multichannel demo
            print(f"\n2. Multi-channel arrangements for {split_name}:")
            arrangements = ['horizontal', 'vertical', 'square']
            for arrangement in arrangements:
                print(f"\n--- {arrangement} arrangement for {split_name} ---")
                test_multichannel_grid(dataset, n_samples=2, channels_to_plot=[0, 1, 2],
                                       channel_arrangement=arrangement, analysis_dir=analysis_dir)
                plt.pause(1.0)

    return analysis_dir


def quick_split_test(metadata_file, data_root, test_val_session_filter, test_val_split_ratio=0.5):
    """
    Quick test function to verify splitting logic without plotting
    """
    print(f"\n=== QUICK SPLIT TEST ===")
    print(f"Test/Val sessions: {test_val_session_filter}")
    print(f"Split ratio: {test_val_split_ratio}")

    try:
        # Import from your actual module name
        from src.data.datasets import create_filtered_split_datasets

        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            metadata_file=metadata_file,
            target_batch_size=200,
            data_root=data_root,
            test_val_session_filter=test_val_session_filter,
            test_val_split_ratio=test_val_split_ratio,
            split_level='sequence',
            random_seed=42
        )

        # Print summary
        splits_info = []
        for name, ds in [('Train', train_ds), ('Test', test_ds), ('Val', val_ds)]:
            size = len(ds.metadata) if len(ds) > 0 else 0
            sessions = sorted(ds.metadata['session'].unique()) if size > 0 else []
            splits_info.append(f"{name}: {size} sequences, sessions {sessions}")

        print("\nResults:")
        for info in splits_info:
            print(f"  {info}")

        return train_ds, test_ds, val_ds

    except Exception as e:
        print(f"Error in quick test: {e}")
        return None


def validate_no_data_leakage(train_ds, test_ds, val_ds, split_level='sequence'):
    """
    Validate that there's no data leakage between splits
    """
    print(f"\n=== DATA LEAKAGE VALIDATION ({split_level} level) ===")

    if split_level == 'experiment':
        # Check experiment-level separation
        train_experiments = set(train_ds.metadata['file_path']) if len(train_ds) > 0 else set()
        test_experiments = set(test_ds.metadata['file_path']) if len(test_ds) > 0 else set()
        val_experiments = set(val_ds.metadata['file_path']) if len(val_ds) > 0 else set()

        # Check for overlaps
        train_test_overlap = train_experiments & test_experiments
        train_val_overlap = train_experiments & val_experiments
        test_val_overlap = test_experiments & val_experiments

        print(f"Train experiments: {len(train_experiments)}")
        print(f"Test experiments: {len(test_experiments)}")
        print(f"Val experiments: {len(val_experiments)}")
        print(f"Train-Test overlap: {len(train_test_overlap)} {'✓' if len(train_test_overlap) == 0 else '✗'}")
        print(f"Train-Val overlap: {len(train_val_overlap)} {'✓' if len(train_val_overlap) == 0 else '✗'}")
        print(f"Test-Val overlap: {len(test_val_overlap)} {'✓' if len(test_val_overlap) == 0 else '✗'}")

    elif split_level == 'sequence':
        # Check sequence-level separation (more complex)
        def create_sequence_ids(df):
            if len(df) == 0:
                return set()
            return set(df['file_path'].astype(str) + '_' +
                       df['sequence_id'].astype(str) + '_' +
                       df['token_id'].astype(str))

        train_seq_ids = create_sequence_ids(train_ds.metadata) if len(train_ds) > 0 else set()
        test_seq_ids = create_sequence_ids(test_ds.metadata) if len(test_ds) > 0 else set()
        val_seq_ids = create_sequence_ids(val_ds.metadata) if len(val_ds) > 0 else set()

        # Check overlaps
        train_test_overlap = train_seq_ids & test_seq_ids
        train_val_overlap = train_seq_ids & val_seq_ids
        test_val_overlap = test_seq_ids & val_seq_ids

        print(f"Train sequences: {len(train_seq_ids)}")
        print(f"Test sequences: {len(test_seq_ids)}")
        print(f"Val sequences: {len(val_seq_ids)}")
        print(f"Train-Test overlap: {len(train_test_overlap)} {'✓' if len(train_test_overlap) == 0 else '✗'}")
        print(f"Train-Val overlap: {len(train_val_overlap)} {'✓' if len(train_val_overlap) == 0 else '✗'}")
        print(f"Test-Val overlap: {len(test_val_overlap)} {'✓' if len(test_val_overlap) == 0 else '✗'}")

    # Overall validation
    no_leakage = all([
        len(train_test_overlap) == 0,
        len(train_val_overlap) == 0,
        len(test_val_overlap) == 0
    ])

    print(
        f"\nValidation result: {'PASS - No data leakage detected' if no_leakage else 'FAIL - Data leakage detected!'}")

    return no_leakage


def test_scenario_session_split():
    """Test scenario: Split specific session"""
    metadata_file = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010/metadata.csv'
    data_root = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010'

    print("SCENARIO: Session-based splitting")
    datasets = quick_split_test(metadata_file, data_root,
                                test_val_session_filter=[11],
                                test_val_split_ratio=0.5)

    if datasets:
        validate_no_data_leakage(*datasets, 'sequence')

    return datasets


def test_scenario_participant_split():
    """Test scenario: Split specific participants"""
    metadata_file = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010/metadata.csv'
    data_root = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010'

    print("SCENARIO: Participant-based splitting")

    try:
        from src.data.datasets import create_filtered_split_datasets

        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            metadata_file=metadata_file,
            data_root=data_root,
            test_val_participant_filter=[10, 23],  # These participants → test/val
            test_val_split_ratio=0.3,  # 30% test, 70% val
            split_level='experiment',  # Prevent data leakage
            random_seed=42
        )

        # Print results
        for name, ds in [('Train', train_ds), ('Test', test_ds), ('Val', val_ds)]:
            size = len(ds.metadata) if len(ds) > 0 else 0
            participants = sorted(ds.metadata['participant'].unique()) if size > 0 else []
            print(f"{name}: {size} sequences, participants {participants}")

        validate_no_data_leakage(train_ds, test_ds, val_ds, 'experiment')

        return train_ds, test_ds, val_ds

    except Exception as e:
        print(f"Error in participant test: {e}")
        return None


def run_comprehensive_example_test():
    """
    Extracted functionality from the original middle __main__ block.
    This contains the main example usage demonstration.
    """
    print("ADAPTED TESTBENCH FOR FILTERED SPLIT DATASET:")
    print("- Works with FilteredSplitH5Dataset")
    print("- Handles empty datasets gracefully")
    print("- Supports train/test/val split visualization")
    print("- Includes split-specific metadata in plots and files")
    print("- Comprehensive statistics for all splits")

    # Example usage with your specific use case
    metadata_file = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010/metadata.csv'
    data_root = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010'

    # Test your specific scenario: Session 11 for test/val, everything else for train
    result = comprehensive_dataset_test(
        metadata_file=metadata_file,
        data_root=data_root,
        target_batch_size=200,
        # Filter session 11 for test/val data
        test_val_session_filter=[18],
        test_val_split_ratio=0.5,  # 50% test, 50% val from session 11
        split_level='sequence',
        # Optional: Apply global filters to all data first
        # global_session_filter=[10, 11, 12, 13],  # Only include these sessions overall
        enable_plotting=True,
        random_seed=42
    )

    if result:
        (train_ds, test_ds, val_ds), analysis_dir = result
        print(f"\nAnalysis results saved in: {analysis_dir}")

        # Print verification information
        print("\n" + "=" * 60)
        print("SPLIT VERIFICATION:")
        print("=" * 60)

        print(f"Train dataset: {len(train_ds)} batches, {len(train_ds.metadata) if len(train_ds) > 0 else 0} sequences")
        if len(train_ds) > 0:
            train_sessions = sorted(train_ds.metadata['session'].unique())
            print(f"  Sessions: {[int(x) for x in train_sessions]}")

        print(f"Test dataset: {len(test_ds)} batches, {len(test_ds.metadata) if len(test_ds) > 0 else 0} sequences")
        if len(test_ds) > 0:
            test_sessions = sorted(test_ds.metadata['session'].unique())
            print(f"  Sessions: {[int(x) for x in test_sessions]}")

        print(f"Val dataset: {len(val_ds)} batches, {len(val_ds.metadata) if len(val_ds) > 0 else 0} sequences")
        if len(val_ds) > 0:
            val_sessions = sorted(val_ds.metadata['session'].unique())
            print(f"  Sessions: {[int(x) for x in val_sessions]}")

        # Optional: Demo all layout types
        # print("\nRunning layout demonstrations...")
        # demo_all_split_layouts(train_ds, test_ds, val_ds, analysis_dir)

    return result


if __name__ == "__main__":
    # Run specific test scenarios
    print("Running test scenarios...")

    # Main comprehensive example test
    run_comprehensive_example_test()

    # Uncomment to run different tests:
    # test_scenario_session_split()
    # test_scenario_participant_split()