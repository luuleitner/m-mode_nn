"""
Aggregate Cross-Validation Results

Performs hierarchical aggregation for nested within-participant CV:
  Level 1: Per-participant mean +/- std across inner folds (intra-subject variability)
  Level 2: Global mean +/- std across participants (inter-subject variability)

Usage:
    python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds
    python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds --metric balanced_accuracy
    python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds --output results_summary.csv

Input: CV folds directory containing fold_info.json and training results
Output: Aggregated statistics table and optional plots
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import utils.logging_config as logconf
logger = logconf.get_logger("AGGREGATE_CV")


def load_fold_results(cv_dir):
    """
    Load results from all folds in a CV directory.

    Looks for:
    - fold_info.json: Fold configuration (participant, inner_fold_idx, etc.)
    - results.json or metrics.json: Training results (accuracy, loss, etc.)
    - Alternatively: best_metrics.json from training

    Returns:
        List of dicts with fold info and metrics merged
    """
    cv_dir = Path(cv_dir)

    if not cv_dir.exists():
        raise FileNotFoundError(f"CV directory not found: {cv_dir}")

    # Find all fold directories
    fold_dirs = sorted([
        d for d in cv_dir.iterdir()
        if d.is_dir() and (d.name.startswith('fold_') or '_fold' in d.name)
    ])

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {cv_dir}")

    logger.info(f"Found {len(fold_dirs)} fold directories")

    results = []
    for fold_dir in fold_dirs:
        fold_data = {'fold_dir': str(fold_dir), 'fold_name': fold_dir.name}

        # Load fold info
        fold_info_path = fold_dir / 'fold_info.json'
        if fold_info_path.exists():
            with open(fold_info_path) as f:
                fold_info = json.load(f)
                fold_data.update(fold_info)
        else:
            logger.warning(f"No fold_info.json in {fold_dir}")

        # Load metrics from various possible files
        metrics_loaded = False
        for metrics_file in ['results.json', 'metrics.json', 'best_metrics.json', 'test_results.json']:
            metrics_path = fold_dir / metrics_file
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    fold_data['metrics'] = metrics
                    # Flatten common metrics to top level for easier access
                    for key in ['accuracy', 'balanced_accuracy', 'f1_score', 'loss',
                               'val_accuracy', 'val_balanced_accuracy', 'test_accuracy',
                               'test_balanced_accuracy', 'val_loss', 'test_loss']:
                        if key in metrics:
                            fold_data[key] = metrics[key]
                    metrics_loaded = True
                    break

        if not metrics_loaded:
            logger.warning(f"No metrics file found in {fold_dir}")

        results.append(fold_data)

    return results


def aggregate_nested_cv(results, metric='balanced_accuracy'):
    """
    Perform hierarchical aggregation for nested within-participant CV.

    Args:
        results: List of fold results from load_fold_results()
        metric: Metric name to aggregate (e.g., 'balanced_accuracy', 'accuracy', 'f1_score')

    Returns:
        dict with:
            'per_participant': DataFrame with per-participant stats
            'global': dict with overall stats
            'raw': DataFrame with all fold results
    """
    # Check if this is nested CV (has inner_fold_idx)
    is_nested = any('inner_fold_idx' in r for r in results)

    if not is_nested:
        logger.info("Standard CV detected (not nested), performing simple aggregation")
        return aggregate_simple_cv(results, metric)

    logger.info(f"Nested CV detected, aggregating metric: {metric}")

    # Group results by participant
    participant_results = defaultdict(list)
    for r in results:
        participant = r.get('participant', 'unknown')
        if metric in r:
            participant_results[participant].append(r[metric])

    if not participant_results:
        raise ValueError(f"No results found for metric '{metric}'. "
                        f"Available metrics: {[k for k in results[0].keys() if k not in ['fold_dir', 'fold_name', 'metrics']]}")

    # Level 1: Per-participant aggregation
    per_participant = []
    for participant, values in sorted(participant_results.items()):
        values = np.array(values)
        per_participant.append({
            'participant': participant,
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
            'min': np.min(values),
            'max': np.max(values),
            'n_folds': len(values),
        })

    per_participant_df = pd.DataFrame(per_participant)

    # Level 2: Global aggregation (mean of participant means)
    participant_means = per_participant_df['mean'].values
    participant_stds = per_participant_df['std'].values

    global_stats = {
        'metric': metric,
        'mean_of_means': np.mean(participant_means),
        'std_of_means': np.std(participant_means, ddof=1) if len(participant_means) > 1 else 0.0,
        'mean_intra_subject_std': np.mean(participant_stds),
        'n_participants': len(participant_means),
        'n_total_folds': sum(len(v) for v in participant_results.values()),
    }

    # Create raw results DataFrame
    raw_df = pd.DataFrame(results)

    return {
        'per_participant': per_participant_df,
        'global': global_stats,
        'raw': raw_df,
    }


def aggregate_simple_cv(results, metric='balanced_accuracy'):
    """Simple aggregation for non-nested CV."""
    values = [r[metric] for r in results if metric in r]

    if not values:
        raise ValueError(f"No results found for metric '{metric}'")

    values = np.array(values)

    global_stats = {
        'metric': metric,
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'min': np.min(values),
        'max': np.max(values),
        'n_folds': len(values),
    }

    raw_df = pd.DataFrame(results)

    return {
        'per_participant': None,
        'global': global_stats,
        'raw': raw_df,
    }


def print_summary(aggregated, metric='balanced_accuracy'):
    """Print a formatted summary of aggregated results."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Metric: {metric}")

    global_stats = aggregated['global']
    per_participant = aggregated.get('per_participant')

    if per_participant is not None and len(per_participant) > 0:
        # Nested CV summary
        print(f"\n{'=' * 70}")
        print("PER-PARTICIPANT RESULTS (intra-subject variability)")
        print("=" * 70)
        print(f"{'Participant':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Folds':>8}")
        print("-" * 70)

        for _, row in per_participant.iterrows():
            print(f"P{row['participant']:<11} {row['mean']:>10.4f} {row['std']:>10.4f} "
                  f"{row['min']:>10.4f} {row['max']:>10.4f} {row['n_folds']:>8}")

        print("-" * 70)
        print(f"\n{'=' * 70}")
        print("GLOBAL RESULTS (inter-subject variability)")
        print("=" * 70)
        print(f"Mean across participants:     {global_stats['mean_of_means']:.4f} +/- {global_stats['std_of_means']:.4f}")
        print(f"Mean intra-subject std:       {global_stats['mean_intra_subject_std']:.4f}")
        print(f"Total participants:           {global_stats['n_participants']}")
        print(f"Total folds:                  {global_stats['n_total_folds']}")

    else:
        # Simple CV summary
        print(f"\nMean:      {global_stats['mean']:.4f} +/- {global_stats['std']:.4f}")
        print(f"Min:       {global_stats['min']:.4f}")
        print(f"Max:       {global_stats['max']:.4f}")
        print(f"N folds:   {global_stats['n_folds']}")

    print("=" * 70 + "\n")


def save_results(aggregated, output_path, metric='balanced_accuracy'):
    """Save aggregated results to CSV files."""
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_stem = output_path.stem

    # Save per-participant results
    if aggregated.get('per_participant') is not None:
        per_part_path = output_dir / f"{output_stem}_per_participant.csv"
        aggregated['per_participant'].to_csv(per_part_path, index=False)
        logger.info(f"Saved per-participant results to: {per_part_path}")

    # Save global summary
    global_path = output_dir / f"{output_stem}_global.json"
    with open(global_path, 'w') as f:
        json.dump(aggregated['global'], f, indent=2)
    logger.info(f"Saved global summary to: {global_path}")

    # Save raw results
    raw_path = output_dir / f"{output_stem}_raw.csv"
    # Select only relevant columns for raw output
    raw_df = aggregated['raw']
    columns_to_save = ['fold_name', 'participant', 'inner_fold_idx', metric]
    columns_to_save = [c for c in columns_to_save if c in raw_df.columns]
    raw_df[columns_to_save].to_csv(raw_path, index=False)
    logger.info(f"Saved raw fold results to: {raw_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate cross-validation results with hierarchical statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds

  # Specify metric to aggregate
  python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds --metric accuracy

  # Save results to file
  python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds --output summary.csv

  # Aggregate multiple metrics
  python -m analysis.aggregate_cv_results --cv-dir /path/to/cv_folds --metric balanced_accuracy,accuracy,f1_score
        """
    )
    parser.add_argument(
        '--cv-dir', '-d',
        type=str,
        required=True,
        help='Path to CV folds directory'
    )
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default='balanced_accuracy',
        help='Metric(s) to aggregate (comma-separated for multiple)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for results (default: print to console only)'
    )

    args = parser.parse_args()

    # Load results
    try:
        results = load_fold_results(args.cv_dir)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1

    if not results:
        logger.error("No results to aggregate")
        return 1

    # Handle multiple metrics
    metrics = [m.strip() for m in args.metric.split(',')]

    for metric in metrics:
        try:
            aggregated = aggregate_nested_cv(results, metric=metric)
            print_summary(aggregated, metric=metric)

            if args.output:
                output_path = Path(args.output)
                if len(metrics) > 1:
                    # Add metric name to output path for multiple metrics
                    output_path = output_path.parent / f"{output_path.stem}_{metric}{output_path.suffix}"
                save_results(aggregated, output_path, metric=metric)

        except ValueError as e:
            logger.warning(f"Could not aggregate metric '{metric}': {e}")
            continue

    return 0


if __name__ == '__main__':
    exit(main())
