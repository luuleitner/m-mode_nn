"""
Precompute Train/Val/Test Dataset Splits with Statistics

Creates and caches the dataset splits as pickle files without starting training.
Run this after preprocessing to prepare datasets for training.

Reads from PROCESSED data (train_base_data_path) and creates:
- train_ds.pkl, val_ds.pkl, test_ds.pkl
- stats/label_statistics.csv (label distribution per experiment)
- stats/summary_statistics.json (overall summary)

Usage:
    python -m src.data.precompute_datasets --config config/config.yaml
    python -m src.data.precompute_datasets --config config/config.yaml --force
    python -m src.data.precompute_datasets --config config/config.yaml --stats-only

Options:
    -c, --config      Path to config.yaml (required)
    -f, --force       Overwrite existing pkl files even if they exist
    --stats-only      Only compute statistics, skip dataset creation
"""

import os
import sys
import argparse
import pickle
import glob
import re
import json
import numpy as np
import pandas as pd
import h5py

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config
from src.data.datasets import create_filtered_split_datasets

import utils.logging_config as logconf
logger = logconf.get_logger("PRECOMPUTE")


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def parse_filename(filename):
    """Parse experiment info from filename: S{session}_P{participant}_E{experiment}_Xy.h5"""
    basename = os.path.basename(filename)
    match = re.match(r'S(\d+)_P(\d+)_E(\d+)_Xy\.h5', basename)
    if match:
        return {
            'session': int(match.group(1)),
            'participant': int(match.group(2)),
            'experiment': int(match.group(3))
        }
    return None


def get_label_counts(h5_path):
    """Extract label counts from h5 file. Handles soft and hard labels."""
    with h5py.File(h5_path, 'r') as f:
        labels = f['label'][:]

        # Handle soft labels (convert to hard via argmax)
        if labels.dtype in [np.float32, np.float64]:
            if labels.ndim == 2:
                hard_labels = np.argmax(labels, axis=1)
            elif labels.ndim == 3:
                hard_labels = np.argmax(labels, axis=-1).flatten()
            else:
                hard_labels = labels.flatten()
        else:
            hard_labels = labels.flatten()

        unique, counts = np.unique(hard_labels, return_counts=True)
        count_dict = {int(u): int(c) for u, c in zip(unique, counts)}

        for cls in [0, 1, 2]:
            if cls not in count_dict:
                count_dict[cls] = 0

        return count_dict


def compute_statistics(data_root):
    """Compute label statistics from all h5 files."""
    h5_files = sorted(glob.glob(os.path.join(data_root, 'P*', '*.h5')))

    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(data_root, '**', '*.h5'), recursive=True))

    if not h5_files:
        logger.warning(f"No h5 files found in {data_root}")
        return None, None

    logger.info(f"Computing statistics from {len(h5_files)} h5 files...")

    results = []
    for h5_path in h5_files:
        info = parse_filename(h5_path)
        if info is None:
            continue

        counts = get_label_counts(h5_path)
        total = sum(counts.values())

        results.append({
            'session': info['session'],
            'participant': info['participant'],
            'experiment': info['experiment'],
            'file': os.path.basename(h5_path),
            'label_0_count': counts[0],
            'label_1_count': counts[1],
            'label_2_count': counts[2],
            'total': total,
            'label_0_pct': 100 * counts[0] / total if total > 0 else 0,
            'label_1_pct': 100 * counts[1] / total if total > 0 else 0,
            'label_2_pct': 100 * counts[2] / total if total > 0 else 0,
        })

    df = pd.DataFrame(results)

    # Compute summary
    total_samples = df['total'].sum()
    total_label_0 = df['label_0_count'].sum()
    total_label_1 = df['label_1_count'].sum()
    total_label_2 = df['label_2_count'].sum()

    summary = {
        'total_samples': int(total_samples),
        'total_experiments': len(df),
        'total_sessions': df['session'].nunique(),
        'label_distribution': {
            'label_0': {'count': int(total_label_0), 'percent': round(100 * total_label_0 / total_samples, 2)},
            'label_1': {'count': int(total_label_1), 'percent': round(100 * total_label_1 / total_samples, 2)},
            'label_2': {'count': int(total_label_2), 'percent': round(100 * total_label_2 / total_samples, 2)},
        },
        'class_weights': {
            0: round(total_samples / (3 * total_label_0), 4) if total_label_0 > 0 else 0,
            1: round(total_samples / (3 * total_label_1), 4) if total_label_1 > 0 else 0,
            2: round(total_samples / (3 * total_label_2), 4) if total_label_2 > 0 else 0,
        }
    }

    return df, summary


def save_statistics(data_root, df, summary):
    """Save statistics to files."""
    stats_dir = os.path.join(data_root, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Save detailed CSV
    csv_path = os.path.join(stats_dir, 'label_statistics.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    # Save summary JSON
    json_path = os.path.join(stats_dir, 'summary_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {json_path}")

    return stats_dir


def print_statistics_summary(summary):
    """Print statistics summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("LABEL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total samples:     {summary['total_samples']:,}")
    logger.info(f"  Total experiments: {summary['total_experiments']}")
    logger.info(f"  Total sessions:    {summary['total_sessions']}")
    logger.info("")
    logger.info("  Label Distribution:")
    for label, data in summary['label_distribution'].items():
        logger.info(f"    {label}: {data['count']:,} ({data['percent']:.1f}%)")
    logger.info("")
    logger.info("  Computed Class Weights (inverse frequency):")
    for cls, weight in summary['class_weights'].items():
        logger.info(f"    Class {cls}: {weight:.4f}")


# =============================================================================
# DATASET PRECOMPUTATION
# =============================================================================

def precompute_datasets(config_path, force=False):
    """
    Precompute and save train/val/test dataset splits.

    Args:
        config_path: Path to config.yaml
        force: If True, overwrite existing pkl files
    """
    # Load config (don't create training directories)
    config = load_config(config_path, create_dirs=False)

    # Get paths
    data_root = config.get_train_data_root()
    if not data_root:
        logger.error("train_base_data_path not set in config")
        return False

    train_path = os.path.join(data_root, 'train_ds.pkl')
    val_path = os.path.join(data_root, 'val_ds.pkl')
    test_path = os.path.join(data_root, 'test_ds.pkl')

    # Check if files already exist
    all_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    if all_exist and not force:
        logger.info("Dataset pkl files already exist. Use --force to overwrite.")
        logger.info(f"  - {train_path}")
        logger.info(f"  - {val_path}")
        logger.info(f"  - {test_path}")
        return True

    # Create datasets
    logger.info("=" * 60)
    logger.info("PRECOMPUTING DATASET SPLITS")
    logger.info("=" * 60)
    logger.info(f"Data root: {data_root}")

    try:
        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            **config.get_dataset_parameters()
        )
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        return False

    # Save datasets
    logger.info("Saving datasets to pickle files...")

    with open(train_path, 'wb') as f:
        pickle.dump(train_ds, f)
    logger.info(f"  Saved: {train_path}")

    with open(val_path, 'wb') as f:
        pickle.dump(val_ds, f)
    logger.info(f"  Saved: {val_path}")

    with open(test_path, 'wb') as f:
        pickle.dump(test_ds, f)
    logger.info(f"  Saved: {test_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Train batches: {len(train_ds)}")
    logger.info(f"  Val batches:   {len(val_ds)}")
    logger.info(f"  Test batches:  {len(test_ds)}")

    # Print split info
    train_info = train_ds.get_split_info()
    logger.info(f"  Train sequences: {train_info.get('train_size', 'N/A')}")
    logger.info(f"  Val sequences:   {train_info.get('val_size', 'N/A')}")
    logger.info(f"  Test sequences:  {train_info.get('test_size', 'N/A')}")

    # Print class weights if available
    try:
        class_weights = train_ds.get_class_weights()
        logger.info(f"  Class weights: {class_weights}")
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("Precomputation complete!")
    logger.info(f"Set 'load_data_pickle: true' in config to use cached datasets.")
    logger.info("=" * 60)

    return True


def run_statistics(config_path):
    """Run statistics computation only."""
    config = load_config(config_path, create_dirs=False)
    data_root = config.get_train_data_root()

    if not data_root:
        logger.error("train_base_data_path not set in config")
        return False

    logger.info("=" * 60)
    logger.info("COMPUTING LABEL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Data root: {data_root}")

    df, summary = compute_statistics(data_root)

    if df is None:
        return False

    save_statistics(data_root, df, summary)
    print_statistics_summary(summary)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Precompute train/val/test dataset splits with statistics'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing pkl files'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only compute statistics, skip dataset creation'
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1

    if args.stats_only:
        success = run_statistics(config_path)
    else:
        # Run both dataset creation and statistics
        success = precompute_datasets(config_path, force=args.force)
        if success:
            run_statistics(config_path)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
