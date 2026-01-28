"""
Dataset Split Statistics (Train/Val/Test PKL Analysis)

Analyzes the precomputed train/val/test pkl files to show:
- Label distribution per split (from metadata)
- Class balance comparison across splits
- Experiment/session coverage per split
- BATCH-LEVEL analysis (what model actually sees during training)
  - Per-batch label distribution
  - Batch size statistics
  - Dominant class per batch

Reads from PROCESSED data (train_base_data_path/latest) pkl files.

Usage:
    python analysis/statistics_train_val_test-pkl.py
    python analysis/statistics_train_val_test-pkl.py --config config/config.yaml
    python analysis/statistics_train_val_test-pkl.py --data-path /path/to/run_folder

Options:
    -c, --config      Path to config.yaml (default: config/config.yaml)
    -d, --data-path   Path to processed run folder (overrides config)
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.configurator import load_config


def load_dataset_pkl(pkl_path):
    """Load a pickled dataset."""
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_split_label_distribution(dataset):
    """
    Extract label distribution from a dataset.

    Returns:
        dict with label counts and percentages
    """
    if dataset is None or len(dataset) == 0:
        return {'total': 0, 'counts': {0: 0, 1: 0, 2: 0}, 'percentages': {0: 0, 1: 0, 2: 0}}

    # Get labels from metadata
    metadata = dataset.metadata

    if 'label_logic' in metadata.columns or 'token label_logic' in metadata.columns:
        label_col = 'label_logic' if 'label_logic' in metadata.columns else 'token label_logic'
        labels = metadata[label_col].astype(int).values
    else:
        # Try to extract from batch data
        labels = []
        for batch_info in dataset.batch_mapping:
            for item in batch_info:
                lbl = item['sequence_metadata'].get('label_logic',
                      item['sequence_metadata'].get('token label_logic', 0))
                labels.append(int(lbl))
        labels = np.array(labels)

    total = len(labels)
    counts = {0: 0, 1: 0, 2: 0}
    for lbl in labels:
        if lbl in counts:
            counts[lbl] += 1

    percentages = {k: round(100 * v / total, 2) if total > 0 else 0 for k, v in counts.items()}

    return {
        'total': total,
        'counts': counts,
        'percentages': percentages
    }


def get_split_metadata_info(dataset):
    """Extract metadata information from dataset."""
    if dataset is None or len(dataset) == 0:
        return {}

    metadata = dataset.metadata
    info = {
        'num_sequences': len(metadata),
        'num_batches': len(dataset.batch_mapping),
        'num_experiments': metadata['file_path'].nunique() if 'file_path' in metadata.columns else 0,
    }

    if 'session' in metadata.columns:
        info['sessions'] = sorted(metadata['session'].unique().tolist())
        info['num_sessions'] = len(info['sessions'])

    if 'participant' in metadata.columns:
        info['participants'] = sorted(metadata['participant'].unique().tolist())
        info['num_participants'] = len(info['participants'])

    if 'experiment' in metadata.columns:
        info['experiments'] = sorted(metadata['experiment'].unique().tolist())
        info['num_experiments'] = len(info['experiments'])

    return info


def analyze_batch_labels(dataset, max_batches=None):
    """
    Analyze actual label distribution per batch (what model sees).

    Returns:
        dict with batch-level statistics
    """
    if dataset is None or len(dataset) == 0:
        return None

    num_batches = len(dataset)
    if max_batches:
        num_batches = min(num_batches, max_batches)

    batch_stats = []
    all_labels = []

    print(f"    Analyzing {num_batches} batches...")

    for batch_idx in range(num_batches):
        try:
            batch = dataset[batch_idx]
            labels = batch['labels']

            if labels is None:
                continue

            # Convert to numpy
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()

            # Handle soft labels (argmax to get dominant class)
            if labels.dtype in [np.float32, np.float64] and labels.ndim >= 2:
                hard_labels = np.argmax(labels, axis=-1).flatten()
            else:
                hard_labels = labels.flatten().astype(int)

            all_labels.extend(hard_labels.tolist())

            # Per-batch stats
            batch_size = len(hard_labels)
            counts = {0: 0, 1: 0, 2: 0}
            for lbl in hard_labels:
                if lbl in counts:
                    counts[lbl] += 1

            # Dominant class in this batch
            dominant_class = max(counts, key=counts.get)

            batch_stats.append({
                'batch_idx': batch_idx,
                'batch_size': batch_size,
                'label_0': counts[0],
                'label_1': counts[1],
                'label_2': counts[2],
                'pct_0': 100 * counts[0] / batch_size if batch_size > 0 else 0,
                'pct_1': 100 * counts[1] / batch_size if batch_size > 0 else 0,
                'pct_2': 100 * counts[2] / batch_size if batch_size > 0 else 0,
                'dominant_class': dominant_class,
            })

        except Exception as e:
            print(f"    Warning: Could not analyze batch {batch_idx}: {e}")

    if not batch_stats:
        return None

    df_batches = pd.DataFrame(batch_stats)

    # Overall batch statistics
    total_samples = len(all_labels)
    overall_counts = {0: all_labels.count(0), 1: all_labels.count(1), 2: all_labels.count(2)}

    return {
        'num_batches_analyzed': len(batch_stats),
        'total_samples': total_samples,
        'overall_distribution': {
            'counts': overall_counts,
            'percentages': {k: round(100 * v / total_samples, 2) for k, v in overall_counts.items()}
        },
        'batch_size': {
            'mean': df_batches['batch_size'].mean(),
            'std': df_batches['batch_size'].std(),
            'min': df_batches['batch_size'].min(),
            'max': df_batches['batch_size'].max(),
        },
        'batches_by_dominant_class': {
            0: int((df_batches['dominant_class'] == 0).sum()),
            1: int((df_batches['dominant_class'] == 1).sum()),
            2: int((df_batches['dominant_class'] == 2).sum()),
        },
        'per_batch_pct_stats': {
            'label_0': {'mean': df_batches['pct_0'].mean(), 'std': df_batches['pct_0'].std()},
            'label_1': {'mean': df_batches['pct_1'].mean(), 'std': df_batches['pct_1'].std()},
            'label_2': {'mean': df_batches['pct_2'].mean(), 'std': df_batches['pct_2'].std()},
        },
        'batch_details': df_batches,
    }


def analyze_datasets(data_path):
    """
    Analyze train/val/test datasets.

    Returns:
        dict with analysis results for each split
    """
    train_path = os.path.join(data_path, 'train_ds.pkl')
    val_path = os.path.join(data_path, 'val_ds.pkl')
    test_path = os.path.join(data_path, 'test_ds.pkl')

    # Check if files exist
    missing = []
    for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not os.path.exists(path):
            missing.append(name)

    if missing:
        print(f"Warning: Missing pkl files: {missing}")
        print(f"Run: python -m src.data.precompute_datasets --config config/config.yaml")

    # Load datasets
    train_ds = load_dataset_pkl(train_path)
    val_ds = load_dataset_pkl(val_path)
    test_ds = load_dataset_pkl(test_path)

    results = {}

    for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        if ds is not None:
            print(f"  Analyzing {name} split...")
            label_dist = get_split_label_distribution(ds)
            meta_info = get_split_metadata_info(ds)
            batch_analysis = analyze_batch_labels(ds, max_batches=None)  # Analyze all batches
            results[name] = {
                'label_distribution': label_dist,
                'metadata': meta_info,
                'batch_analysis': batch_analysis
            }
        else:
            results[name] = None

    return results


def print_analysis(results, data_path):
    """Print formatted analysis results."""
    print("\n" + "=" * 70)
    print("DATASET SPLIT ANALYSIS")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print("=" * 70)

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Split':<10} {'Sequences':>12} {'Batches':>10} {'Experiments':>12} {'Sessions':>10}")
    print("-" * 70)

    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            print(f"{split:<10} {'(missing)':<12}")
            continue

        meta = results[split]['metadata']
        print(f"{split:<10} {meta.get('num_sequences', 0):>12,} {meta.get('num_batches', 0):>10} "
              f"{meta.get('num_experiments', 0):>12} {meta.get('num_sessions', 0):>10}")

    print("-" * 70)

    # Label distribution per split
    print("\n" + "-" * 70)
    print("LABEL DISTRIBUTION PER SPLIT")
    print("-" * 70)
    print(f"{'Split':<10} {'Label 0 (Noise)':>18} {'Label 1 (Up)':>18} {'Label 2 (Down)':>18}")
    print("-" * 70)

    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            continue

        dist = results[split]['label_distribution']
        counts = dist['counts']
        pcts = dist['percentages']

        print(f"{split:<10} {counts[0]:>8,} ({pcts[0]:>5.1f}%) "
              f"{counts[1]:>8,} ({pcts[1]:>5.1f}%) "
              f"{counts[2]:>8,} ({pcts[2]:>5.1f}%)")

    print("-" * 70)

    # Class balance comparison
    print("\n" + "-" * 70)
    print("CLASS BALANCE COMPARISON")
    print("-" * 70)

    # Calculate overall totals
    total_all = sum(results[s]['label_distribution']['total']
                    for s in ['train', 'val', 'test'] if results.get(s))

    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            continue

        dist = results[split]['label_distribution']
        split_total = dist['total']
        split_pct = 100 * split_total / total_all if total_all > 0 else 0

        print(f"{split}: {split_total:,} sequences ({split_pct:.1f}% of total)")

    print("-" * 70)

    # Session/experiment coverage
    print("\n" + "-" * 70)
    print("SESSION COVERAGE")
    print("-" * 70)

    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            continue

        meta = results[split]['metadata']
        sessions = meta.get('sessions', [])
        if sessions:
            print(f"{split}: Sessions {sessions}")

    print("-" * 70)

    # Compute class weights for training
    if results.get('train'):
        print("\n" + "-" * 70)
        print("RECOMMENDED CLASS WEIGHTS (from train set)")
        print("-" * 70)

        dist = results['train']['label_distribution']
        total = dist['total']
        counts = dist['counts']
        num_classes = 3

        weights = {}
        for cls in [0, 1, 2]:
            if counts[cls] > 0:
                weights[cls] = round(total / (num_classes * counts[cls]), 4)
            else:
                weights[cls] = 0

        for cls, weight in weights.items():
            label_names = {0: 'Noise', 1: 'Up/Right', 2: 'Down/Left'}
            print(f"  Class {cls} ({label_names[cls]}): {weight:.4f}")

        print("-" * 70)

    # Batch-level analysis (what model actually sees)
    print("\n" + "=" * 70)
    print("BATCH-LEVEL ANALYSIS (what model sees during training)")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        if results.get(split) is None or results[split].get('batch_analysis') is None:
            continue

        batch_info = results[split]['batch_analysis']

        print(f"\n{split.upper()} BATCHES:")
        print("-" * 50)
        print(f"  Batches analyzed: {batch_info['num_batches_analyzed']}")
        print(f"  Total samples:    {batch_info['total_samples']:,}")

        bs = batch_info['batch_size']
        print(f"  Batch size:       mean={bs['mean']:.1f}, std={bs['std']:.1f}, "
              f"min={bs['min']}, max={bs['max']}")

        print(f"\n  Label distribution (from actual batch data):")
        dist = batch_info['overall_distribution']
        for cls in [0, 1, 2]:
            print(f"    Class {cls}: {dist['counts'][cls]:>8,} ({dist['percentages'][cls]:>5.1f}%)")

        print(f"\n  Per-batch label percentages (mean +/- std):")
        pct_stats = batch_info['per_batch_pct_stats']
        for cls in [0, 1, 2]:
            label_key = f'label_{cls}'
            print(f"    Class {cls}: {pct_stats[label_key]['mean']:>5.1f}% +/- {pct_stats[label_key]['std']:>5.1f}%")

        print(f"\n  Batches by dominant class:")
        dom = batch_info['batches_by_dominant_class']
        total_batches = sum(dom.values())
        for cls in [0, 1, 2]:
            pct = 100 * dom[cls] / total_batches if total_batches > 0 else 0
            print(f"    Class {cls} dominant: {dom[cls]:>4} batches ({pct:>5.1f}%)")

    print("\n" + "=" * 70)
    print()


def save_analysis(results, data_path):
    """Save analysis results to files."""
    stats_dir = os.path.join(data_path, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(stats_dir, 'dataset_split_statistics.json')

    # Convert to JSON-serializable format
    json_results = {}
    for split, data in results.items():
        if data is None:
            json_results[split] = None
        else:
            json_results[split] = {
                'label_distribution': data['label_distribution'],
                'metadata': {
                    k: v for k, v in data['metadata'].items()
                    if k not in ['sessions', 'participants', 'experiments']  # Skip lists for cleaner JSON
                }
            }
            # Add lists separately
            for key in ['sessions', 'participants', 'experiments']:
                if key in data['metadata']:
                    json_results[split]['metadata'][key] = [int(x) for x in data['metadata'][key]]

            # Add batch-level statistics (excluding the DataFrame)
            if data.get('batch_analysis'):
                batch_info = data['batch_analysis']
                json_results[split]['batch_analysis'] = {
                    'num_batches_analyzed': batch_info['num_batches_analyzed'],
                    'total_samples': batch_info['total_samples'],
                    'overall_distribution': batch_info['overall_distribution'],
                    'batch_size': {k: float(v) for k, v in batch_info['batch_size'].items()},
                    'batches_by_dominant_class': batch_info['batches_by_dominant_class'],
                    'per_batch_pct_stats': {
                        k: {k2: float(v2) for k2, v2 in v.items()}
                        for k, v in batch_info['per_batch_pct_stats'].items()
                    }
                }

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved: {json_path}")

    # Save CSV summary
    csv_data = []
    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            continue

        dist = results[split]['label_distribution']
        meta = results[split]['metadata']

        csv_data.append({
            'split': split,
            'num_sequences': meta.get('num_sequences', 0),
            'num_batches': meta.get('num_batches', 0),
            'num_experiments': meta.get('num_experiments', 0),
            'num_sessions': meta.get('num_sessions', 0),
            'label_0_count': dist['counts'][0],
            'label_1_count': dist['counts'][1],
            'label_2_count': dist['counts'][2],
            'label_0_pct': dist['percentages'][0],
            'label_1_pct': dist['percentages'][1],
            'label_2_pct': dist['percentages'][2],
        })

    csv_path = os.path.join(stats_dir, 'dataset_split_summary.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save batch-level details for each split
    for split in ['train', 'val', 'test']:
        if results.get(split) is None:
            continue
        if results[split].get('batch_analysis') is None:
            continue

        batch_df = results[split]['batch_analysis'].get('batch_details')
        if batch_df is not None and len(batch_df) > 0:
            batch_csv_path = os.path.join(stats_dir, f'batch_details_{split}.csv')
            batch_df.to_csv(batch_csv_path, index=False)
            print(f"Saved: {batch_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze train/val/test dataset splits'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='Path to processed run folder (overrides config)'
    )

    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        # Load from config
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)

        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return 1

        config = load_config(config_path, create_dirs=False)
        data_path = config.get_train_data_root()

    if not data_path or not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        return 1

    # Run analysis
    results = analyze_datasets(data_path)
    print_analysis(results, data_path)
    save_analysis(results, data_path)

    return 0


if __name__ == '__main__':
    exit(main())
