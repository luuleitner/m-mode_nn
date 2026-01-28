"""
Label Distribution Statistics

Computes label distributions across sessions and experiments from processed h5 files.
Saves statistics to <train_base_data_path>/stats/

Usage:
    python analysis/statistics_processed_data.py
    python analysis/statistics_processed_data.py --config config/config.yaml
"""

import os
import sys
import argparse
import glob
import re
import json
import numpy as np
import pandas as pd
import h5py
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_filename(filename):
    """
    Parse experiment info from filename.
    Format: S{session}_P{participant}_E{experiment}_Xy.h5
    """
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
    """
    Extract label counts from h5 file.
    Handles both soft labels (float, shape [N, num_classes]) and hard labels (int).
    """
    with h5py.File(h5_path, 'r') as f:
        labels = f['label'][:]

        # Handle soft labels (convert to hard labels via argmax)
        if labels.dtype in [np.float32, np.float64]:
            if labels.ndim == 2:
                # Shape: [num_samples, num_classes]
                hard_labels = np.argmax(labels, axis=1)
            elif labels.ndim == 3:
                # Shape: [num_sequences, seq_len, num_classes]
                hard_labels = np.argmax(labels, axis=-1).flatten()
            else:
                hard_labels = labels.flatten()
        else:
            # Already hard labels
            hard_labels = labels.flatten()

        # Count each class
        unique, counts = np.unique(hard_labels, return_counts=True)
        count_dict = {int(u): int(c) for u, c in zip(unique, counts)}

        # Ensure all classes 0, 1, 2 are present
        for cls in [0, 1, 2]:
            if cls not in count_dict:
                count_dict[cls] = 0

        total = int(np.sum(counts))

        return {
            'total': total,
            'class_0': count_dict[0],
            'class_1': count_dict[1],
            'class_2': count_dict[2],
            'pct_0': 100.0 * count_dict[0] / total if total > 0 else 0,
            'pct_1': 100.0 * count_dict[1] / total if total > 0 else 0,
            'pct_2': 100.0 * count_dict[2] / total if total > 0 else 0,
        }


def compute_statistics(data_path):
    """
    Compute label statistics for all h5 files in data path.
    Returns DataFrames for per-experiment and per-session statistics.
    """
    # Find all h5 files
    h5_files = glob.glob(os.path.join(data_path, '**', '*.h5'), recursive=True)

    if not h5_files:
        raise ValueError(f"No h5 files found in {data_path}")

    print(f"Found {len(h5_files)} h5 files")

    # Collect per-experiment data
    experiment_data = []

    for h5_path in sorted(h5_files):
        info = parse_filename(h5_path)
        if info is None:
            print(f"  Skipping (unknown format): {os.path.basename(h5_path)}")
            continue

        counts = get_label_counts(h5_path)

        experiment_data.append({
            'session': info['session'],
            'participant': info['participant'],
            'experiment': info['experiment'],
            'file': os.path.basename(h5_path),
            **counts
        })

    if not experiment_data:
        raise ValueError("No valid experiment files found")

    # Create experiment DataFrame
    df_exp = pd.DataFrame(experiment_data)
    df_exp = df_exp.sort_values(['session', 'participant', 'experiment'])

    # Aggregate by session
    df_session = df_exp.groupby('session').agg({
        'total': 'sum',
        'class_0': 'sum',
        'class_1': 'sum',
        'class_2': 'sum',
        'experiment': 'count'  # Count experiments per session
    }).reset_index()
    df_session = df_session.rename(columns={'experiment': 'num_experiments'})

    # Recalculate percentages for session aggregates
    df_session['pct_0'] = 100.0 * df_session['class_0'] / df_session['total']
    df_session['pct_1'] = 100.0 * df_session['class_1'] / df_session['total']
    df_session['pct_2'] = 100.0 * df_session['class_2'] / df_session['total']

    return df_exp, df_session


def compute_summary_statistics(df_exp, df_session):
    """Compute overall summary statistics."""
    summary = {
        'total_samples': int(df_exp['total'].sum()),
        'total_experiments': len(df_exp),
        'total_sessions': len(df_session),

        # Overall class distribution
        'overall': {
            'class_0': int(df_exp['class_0'].sum()),
            'class_1': int(df_exp['class_1'].sum()),
            'class_2': int(df_exp['class_2'].sum()),
            'pct_0': float(100.0 * df_exp['class_0'].sum() / df_exp['total'].sum()),
            'pct_1': float(100.0 * df_exp['class_1'].sum() / df_exp['total'].sum()),
            'pct_2': float(100.0 * df_exp['class_2'].sum() / df_exp['total'].sum()),
        },

        # Per-experiment statistics
        'per_experiment': {
            'samples': {
                'mean': float(df_exp['total'].mean()),
                'std': float(df_exp['total'].std()),
                'min': int(df_exp['total'].min()),
                'max': int(df_exp['total'].max()),
            },
            'pct_0': {
                'mean': float(df_exp['pct_0'].mean()),
                'std': float(df_exp['pct_0'].std()),
                'min': float(df_exp['pct_0'].min()),
                'max': float(df_exp['pct_0'].max()),
            },
            'pct_1': {
                'mean': float(df_exp['pct_1'].mean()),
                'std': float(df_exp['pct_1'].std()),
                'min': float(df_exp['pct_1'].min()),
                'max': float(df_exp['pct_1'].max()),
            },
            'pct_2': {
                'mean': float(df_exp['pct_2'].mean()),
                'std': float(df_exp['pct_2'].std()),
                'min': float(df_exp['pct_2'].min()),
                'max': float(df_exp['pct_2'].max()),
            },
        },

        # Per-session statistics
        'per_session': {
            'experiments': {
                'mean': float(df_session['num_experiments'].mean()),
                'std': float(df_session['num_experiments'].std()),
                'min': int(df_session['num_experiments'].min()),
                'max': int(df_session['num_experiments'].max()),
            },
            'samples': {
                'mean': float(df_session['total'].mean()),
                'std': float(df_session['total'].std()),
                'min': int(df_session['total'].min()),
                'max': int(df_session['total'].max()),
            },
        },
    }

    return summary


def print_summary(summary, df_session):
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION SUMMARY")
    print("=" * 60)

    print(f"\nDataset Overview:")
    print(f"  Total samples:     {summary['total_samples']:,}")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Total sessions:    {summary['total_sessions']}")

    print(f"\nOverall Class Distribution:")
    print(f"  Class 0 (neutral):  {summary['overall']['class_0']:>10,}  ({summary['overall']['pct_0']:5.1f}%)")
    print(f"  Class 1 (up):       {summary['overall']['class_1']:>10,}  ({summary['overall']['pct_1']:5.1f}%)")
    print(f"  Class 2 (down):     {summary['overall']['class_2']:>10,}  ({summary['overall']['pct_2']:5.1f}%)")

    print(f"\nPer-Experiment Statistics:")
    ps = summary['per_experiment']
    print(f"  Samples:   mean={ps['samples']['mean']:.1f}, std={ps['samples']['std']:.1f}, "
          f"min={ps['samples']['min']}, max={ps['samples']['max']}")
    print(f"  % Class 0: mean={ps['pct_0']['mean']:.1f}%, std={ps['pct_0']['std']:.1f}%")
    print(f"  % Class 1: mean={ps['pct_1']['mean']:.1f}%, std={ps['pct_1']['std']:.1f}%")
    print(f"  % Class 2: mean={ps['pct_2']['mean']:.1f}%, std={ps['pct_2']['std']:.1f}%")

    print(f"\nPer-Session Summary:")
    print(df_session[['session', 'num_experiments', 'total', 'pct_0', 'pct_1', 'pct_2']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Compute label distribution statistics')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Get data path
    data_path = config['global_setting']['paths']['train_base_data_path']
    print(f"Data path: {data_path}")

    if not os.path.exists(data_path):
        print(f"ERROR: Data path does not exist: {data_path}")
        return 1

    # Compute statistics
    print("\nComputing statistics...")
    df_exp, df_session = compute_statistics(data_path)
    summary = compute_summary_statistics(df_exp, df_session)

    # Print summary
    print_summary(summary, df_session)

    # Create output directory
    output_dir = os.path.join(data_path, 'stats')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to: {output_dir}")

    # Save experiment-level stats
    exp_path = os.path.join(output_dir, 'label_stats_by_experiment.csv')
    df_exp.to_csv(exp_path, index=False)
    print(f"  Saved: label_stats_by_experiment.csv ({len(df_exp)} rows)")

    # Save session-level stats
    session_path = os.path.join(output_dir, 'label_stats_by_session.csv')
    df_session.to_csv(session_path, index=False)
    print(f"  Saved: label_stats_by_session.csv ({len(df_session)} rows)")

    # Save summary
    summary_path = os.path.join(output_dir, 'label_stats_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: label_stats_summary.json")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
