"""
Shared Cross-Validation Infrastructure

Common utilities for K-fold CV training pipelines (CNN classifier, UNet AE, etc.).
Provides fold loading, results aggregation, plotting, and run directory management.
"""

import os
import pickle
import json
import glob
import shutil
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils.logging_config as logconf
logger = logconf.get_logger("CV_UTILS")

# Default colors for up to 5 classes
DEFAULT_COLORS = ['#808080', '#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # gray, green, red, blue, orange


# ---------------------------------------------------------------------------
# Fold data loading
# ---------------------------------------------------------------------------

def load_fold_datasets(fold_dir, augmentation_config=None):
    """Load train/val/test datasets from a fold directory.

    Args:
        fold_dir: Path to fold directory containing pickled datasets.
        augmentation_config: Optional dict for on-the-fly training augmentation.
            If provided and enabled, attaches a SignalAugmenter to the train dataset.
    """
    train_path = os.path.join(fold_dir, 'train_ds.pkl')
    val_path = os.path.join(fold_dir, 'val_ds.pkl')
    test_path = os.path.join(fold_dir, 'test_ds.pkl')

    for path, name in [(train_path, 'Train'), (val_path, 'Val'), (test_path, 'Test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} dataset not found: {path}")

    with open(train_path, 'rb') as f:
        train_ds = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_ds = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_ds = pickle.load(f)

    # Attach on-the-fly augmenter to train set
    if augmentation_config and augmentation_config.get('enabled', False):
        from src.data.augmentations import SignalAugmenter
        aug_cfg = {k: v for k, v in augmentation_config.items() if k != 'enabled'}
        train_ds.set_general_augmenter(SignalAugmenter(config=aug_cfg))
        logger.info("On-the-fly training augmentation enabled")

    return train_ds, val_ds, test_ds


def load_fold_info(fold_dir):
    """Load fold metadata."""
    info_path = os.path.join(fold_dir, 'fold_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Results collection & aggregation
# ---------------------------------------------------------------------------

def collect_existing_fold_results(run_dir):
    """Collect results from already-trained folds in a run directory.

    Recursively discovers fold_results.json files so it works for both
    flat (run_dir/fold_0/) and nested (run_dir/P0/fold_0/) layouts.

    Args:
        run_dir: Timestamped run directory containing fold results.
    """
    fold_results = []

    result_files = sorted(glob.glob(os.path.join(run_dir, '**/fold_results.json'), recursive=True))

    for results_path in result_files:
        with open(results_path, 'r') as f:
            result = json.load(f)
        fold_idx = result.get('fold_idx', len(fold_results))
        result['fold_idx'] = fold_idx
        fold_results.append(result)
        rel = os.path.relpath(results_path, run_dir)
        logger.info(f"Loaded results for fold {fold_idx} from {rel}")

    if not fold_results:
        logger.warning(f"No fold_results.json files found in {run_dir}")

    # Sort by fold index
    fold_results.sort(key=lambda x: x['fold_idx'])

    return fold_results


def aggregate_cv_results(run_dir, fold_results, class_names):
    """Aggregate results across all folds.

    Args:
        run_dir: Timestamped run directory where results will be saved
        fold_results: List of fold result dictionaries
        class_names: List of class name strings (e.g. from label_config)
    """
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Run directory: {run_dir}")

    n_folds = len(fold_results)

    # Detect strategy from fold info
    strategy = fold_results[0].get('fold_info', {}).get('strategy', 'unknown') if fold_results else 'unknown'
    is_participant_within = strategy == 'participant_within'

    # Extract metrics
    val_accs = [r['best_val_acc'] for r in fold_results]
    test_accs = [r.get('test_accuracy', 0) for r in fold_results if 'test_accuracy' in r]

    # Per-class accuracy aggregation
    per_class_accs = {name: [] for name in class_names}
    for r in fold_results:
        if 'per_class_accuracy' in r:
            for name in class_names:
                if name in r['per_class_accuracy']:
                    per_class_accs[name].append(r['per_class_accuracy'][name])

    # Build summary
    summary = {
        'n_folds': n_folds,
        'strategy': strategy,
        'run_dir': run_dir,
        'aggregated_at': datetime.now().isoformat(),
        'validation_accuracy': {
            'mean': float(np.mean(val_accs)),
            'std': float(np.std(val_accs)),
            'min': float(np.min(val_accs)),
            'max': float(np.max(val_accs)),
            'per_fold': val_accs
        },
        'test_accuracy': {
            'mean': float(np.mean(test_accs)) if test_accs else 0,
            'std': float(np.std(test_accs)) if test_accs else 0,
            'min': float(np.min(test_accs)) if test_accs else 0,
            'max': float(np.max(test_accs)) if test_accs else 0,
            'per_fold': test_accs
        },
        'per_class_accuracy': {}
    }

    for name, accs in per_class_accs.items():
        if accs:
            summary['per_class_accuracy'][name] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'per_fold': accs
            }

    # For participant_within strategy: add per-participant breakdown and ranking
    if is_participant_within:
        per_participant = {}
        for r in fold_results:
            participant = r.get('fold_info', {}).get('participant', r['fold_idx'])
            per_participant[f"participant_{participant}"] = {
                'val_accuracy': r['best_val_acc'],
                'test_accuracy': r.get('test_accuracy', 0),
                'n_experiments': r.get('fold_info', {}).get('n_experiments', 0),
                'per_class_accuracy': r.get('per_class_accuracy', {})
            }

        summary['per_participant'] = per_participant

        # Create ranking by test accuracy
        ranking = sorted(
            [(p, data['test_accuracy']) for p, data in per_participant.items()],
            key=lambda x: x[1],
            reverse=True
        )
        summary['participant_ranking'] = [
            {'participant': p, 'test_accuracy': acc} for p, acc in ranking
        ]

    # Print summary
    logger.info(f"\nStrategy: {strategy}")
    logger.info(f"Number of folds: {n_folds}")

    if is_participant_within:
        logger.info(f"\n--- Per-Participant Results ---")
        for p_name, p_data in summary['per_participant'].items():
            logger.info(f"  {p_name}: val={p_data['val_accuracy']:.2%}, test={p_data['test_accuracy']:.2%} "
                       f"({p_data['n_experiments']} experiments)")

        logger.info(f"\n--- Participant Ranking (by test accuracy) ---")
        for i, item in enumerate(summary['participant_ranking'], 1):
            logger.info(f"  {i}. {item['participant']}: {item['test_accuracy']:.2%}")

    logger.info(f"\nValidation Accuracy:")
    logger.info(f"  Mean +/- Std: {summary['validation_accuracy']['mean']:.2%} +/- {summary['validation_accuracy']['std']:.2%}")
    logger.info(f"  Range: [{summary['validation_accuracy']['min']:.2%}, {summary['validation_accuracy']['max']:.2%}]")
    logger.info(f"  Per fold: {[f'{a:.2%}' for a in val_accs]}")

    if test_accs:
        logger.info(f"\nTest Accuracy:")
        logger.info(f"  Mean +/- Std: {summary['test_accuracy']['mean']:.2%} +/- {summary['test_accuracy']['std']:.2%}")
        logger.info(f"  Range: [{summary['test_accuracy']['min']:.2%}, {summary['test_accuracy']['max']:.2%}]")
        logger.info(f"  Per fold: {[f'{a:.2%}' for a in test_accs]}")

    logger.info(f"\nPer-Class Test Accuracy:")
    for name, data in summary['per_class_accuracy'].items():
        logger.info(f"  {name}: {data['mean']:.2%} +/- {data['std']:.2%}")

    # Save summary
    summary_path = os.path.join(run_dir, 'cv_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to: {summary_path}")

    # Create summary plot
    create_cv_summary_plot(run_dir, summary, class_names, fold_results)

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_cv_summary_plot(run_dir, summary, class_names, fold_results=None):
    """Create visualization of CV results.

    Args:
        run_dir: Timestamped run directory where plot will be saved
        summary: Aggregated summary statistics
        class_names: List of class name strings
        fold_results: Optional list of fold result dicts (needed for participant labels)
    """
    n_folds = summary['n_folds']
    strategy = summary.get('strategy', 'unknown')
    is_participant_within = strategy == 'participant_within'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy per fold/participant
    ax = axes[0]
    x = range(n_folds)
    val_accs = summary['validation_accuracy']['per_fold']
    test_accs = summary['test_accuracy']['per_fold']

    # Determine x-axis labels based on strategy
    if is_participant_within and fold_results:
        x_labels = []
        for r in fold_results:
            participant = r.get('fold_info', {}).get('participant', f'P{r["fold_idx"]}')
            if len(str(participant)) > 10:
                x_labels.append(str(participant)[:8] + '..')
            else:
                x_labels.append(str(participant))
        xlabel_text = 'Participant'
        title_text = 'Accuracy per Participant'
    else:
        x_labels = [f'Fold {i}' for i in x]
        xlabel_text = 'Fold'
        title_text = 'Accuracy per Fold'

    width = 0.35
    ax.bar([i - width/2 for i in x], val_accs, width, label='Validation', color='steelblue')
    if test_accs:
        ax.bar([i + width/2 for i in x], test_accs, width, label='Test', color='darkorange')

    ax.axhline(y=summary['validation_accuracy']['mean'], color='steelblue', linestyle='--', alpha=0.7)
    if test_accs:
        ax.axhline(y=summary['test_accuracy']['mean'], color='darkorange', linestyle='--', alpha=0.7)

    ax.set_xlabel(xlabel_text)
    ax.set_ylabel('Accuracy')
    ax.set_title(title_text)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45 if is_participant_within else 0, ha='right' if is_participant_within else 'center')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 2: Per-class accuracy
    ax = axes[1]
    plot_class_names = list(summary['per_class_accuracy'].keys())
    means = [summary['per_class_accuracy'][n]['mean'] for n in plot_class_names]
    stds = [summary['per_class_accuracy'][n]['std'] for n in plot_class_names]

    colors = DEFAULT_COLORS[:len(plot_class_names)]
    bars = ax.bar(plot_class_names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Test Accuracy (Mean +/- Std)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.1%}', ha='center', va='bottom', fontsize=10)

    # Plot 3: Summary statistics
    ax = axes[2]
    ax.axis('off')

    # Build summary text based on strategy
    if is_participant_within:
        stats_text = f"""
Within-Subject CV Summary
{'='*30}

Participants: {n_folds}

Validation Accuracy:
  Mean: {summary['validation_accuracy']['mean']:.2%}
  Std:  {summary['validation_accuracy']['std']:.2%}

Test Accuracy:
  Mean: {summary['test_accuracy']['mean']:.2%}
  Std:  {summary['test_accuracy']['std']:.2%}

Ranking (by test acc):
"""
        ranking = summary.get('participant_ranking', [])
        for i, item in enumerate(ranking[:5], 1):  # Top 5
            p_name = item['participant'].replace('participant_', '')
            if len(p_name) > 8:
                p_name = p_name[:6] + '..'
            stats_text += f"  {i}. {p_name}: {item['test_accuracy']:.1%}\n"
        if len(ranking) > 5:
            stats_text += f"  ... ({len(ranking) - 5} more)\n"
    else:
        stats_text = f"""
Cross-Validation Summary
{'='*30}

Folds: {n_folds}

Validation Accuracy:
  Mean: {summary['validation_accuracy']['mean']:.2%}
  Std:  {summary['validation_accuracy']['std']:.2%}

Test Accuracy:
  Mean: {summary['test_accuracy']['mean']:.2%}
  Std:  {summary['test_accuracy']['std']:.2%}

Per-Class (Test):
"""
        for name, data in summary['per_class_accuracy'].items():
            stats_text += f"  {name}: {data['mean']:.2%} +/- {data['std']:.2%}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plot_path = os.path.join(run_dir, 'cv_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def list_training_runs(runs_dir):
    """List all training runs in the runs directory.

    Args:
        runs_dir: Path to the runs/ directory (e.g. {data_root}/runs/)
    """
    if not os.path.exists(runs_dir):
        logger.info(f"No runs directory found: {runs_dir}")
        return []

    # Match any subdirectory (unet_*, cnn_*, etc.)
    run_dirs = sorted([
        d for d in glob.glob(os.path.join(runs_dir, '*'))
        if os.path.isdir(d)
    ])

    if not run_dirs:
        logger.info(f"No training runs found in {runs_dir}")
        return []

    logger.info(f"\nTraining runs in {runs_dir}:")
    logger.info("-" * 60)

    runs = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        # Check for cv_results.json to get summary
        results_path = os.path.join(run_dir, 'cv_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            val_acc = results.get('validation_accuracy', {}).get('mean', 0)
            test_acc = results.get('test_accuracy', {}).get('mean', 0)
            n_folds = results.get('n_folds', '?')
            status = f"Complete ({n_folds} folds) - Val: {val_acc:.1%}, Test: {test_acc:.1%}"
        else:
            # Count completed folds (supports both flat and nested layouts)
            completed_folds = len(glob.glob(os.path.join(run_dir, '**/fold_results.json'), recursive=True))
            status = f"In progress ({completed_folds} folds completed)"

        logger.info(f"  {run_name}: {status}")
        runs.append({'name': run_name, 'path': run_dir, 'status': status})

    return runs


def resolve_cv_directory(config, cv_dir_override=None, auto_precompute=False, config_path=None):
    """Resolve the CV folds directory from config or override.

    Args:
        config: Configuration object
        cv_dir_override: Explicit path (from --cv-dir), takes precedence
        auto_precompute: If True and cv_dir doesn't exist, auto-run precompute_kfolds
        config_path: Path to config YAML (required when auto_precompute=True)

    Returns:
        cv_dir path string

    Raises:
        FileNotFoundError: If the resolved directory does not exist and auto_precompute is False
        RuntimeError: If auto-precomputation fails
    """
    if cv_dir_override:
        cv_dir = cv_dir_override
    else:
        data_root = config.get_train_data_root()
        cv_config = getattr(config, 'cross_validation', None)
        if cv_config:
            output_subdir = getattr(cv_config, 'output_subdir', 'cv_folds')
        else:
            output_subdir = 'cv_folds'
        cv_dir = os.path.join(data_root, 'datasets', output_subdir)

    if not os.path.exists(cv_dir):
        if auto_precompute and config_path:
            logger.info(f"CV directory not found: {cv_dir}")
            logger.info("Auto-precomputing kfold datasets...")
            from src.data.precompute_kfolds import precompute_kfolds
            success = precompute_kfolds(config_path)
            if not success:
                raise RuntimeError("Auto-precomputation of kfold datasets failed")
            if not os.path.exists(cv_dir):
                raise FileNotFoundError(
                    f"CV directory still not found after precomputation: {cv_dir}"
                )
        else:
            raise FileNotFoundError(
                f"CV directory not found: {cv_dir}\n"
                "Run 'python -m src.data.precompute_kfolds --config config/config.yaml' first"
            )

    return cv_dir


def discover_fold_dirs(cv_dir):
    """Discover fold directories within cv_dir.

    Supports three naming conventions (checked in priority order):
      1. fold_*           — flat (experiment_kfold, session_loso, participant_lopo)
      2. P*/fold_*        — nested participant_within (new layout)
      3. P*_fold*         — legacy participant_within (backward compat)

    Returns:
        Sorted list of fold directory paths

    Raises:
        FileNotFoundError: If no fold directories are found
    """
    # 1. Flat fold directories
    fold_dirs = sorted(glob.glob(os.path.join(cv_dir, 'fold_*')))
    if not fold_dirs:
        # 2. Nested participant directories (new layout)
        fold_dirs = sorted(glob.glob(os.path.join(cv_dir, 'P*', 'fold_*')))
    if not fold_dirs:
        # 3. Legacy flat participant_within naming
        fold_dirs = sorted(glob.glob(os.path.join(cv_dir, 'P*_fold*')))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {cv_dir}")
    return fold_dirs


def get_fold_output_dir(cv_dir, fold_dir, run_dir):
    """Mirror a fold's relative path from cv_dir into run_dir.

    Examples:
        cv_dir/P0/fold_0 → run_dir/P0/fold_0
        cv_dir/fold_0    → run_dir/fold_0
    """
    rel_path = os.path.relpath(fold_dir, cv_dir)
    return os.path.join(run_dir, rel_path)


def resolve_run_name(run_name_arg, prefix="cnn"):
    """Resolve the run name from argument or generate a timestamped one.

    Args:
        run_name_arg: User-provided run name (may be None)
        prefix: Prefix for auto-generated names (e.g. "cnn", "unet")

    Returns:
        Resolved run name string
    """
    if run_name_arg:
        return run_name_arg
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def setup_run_directory(data_root, run_name, config_path, extra_metadata=None):
    """Create a timestamped run directory inside {data_root}/runs/ and save metadata.

    Args:
        data_root: Data root directory (train_base_data_path)
        run_name: Name for the run subdirectory
        config_path: Path to config YAML to copy into the run dir
        extra_metadata: Optional dict of extra fields to include in run_config.json

    Returns:
        Absolute path to the created run directory
    """
    run_dir = os.path.join(data_root, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save run config
    run_config = {
        'run_name': run_name,
        'started_at': datetime.now().isoformat(),
        'config_file': os.path.abspath(config_path),
    }
    if extra_metadata:
        run_config.update(extra_metadata)

    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # Copy config file for reproducibility
    config_copy_path = os.path.join(run_dir, 'config.yaml')
    shutil.copy2(os.path.abspath(config_path), config_copy_path)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config saved to: {config_copy_path}")

    return run_dir


# ---------------------------------------------------------------------------
# Training loop orchestration
# ---------------------------------------------------------------------------

def run_cv_training_loop(fold_dirs, train_fold_fn, run_dir, class_names, args,
                         cv_dir, single_fold=None, module_name="train_cv"):
    """Run the CV training loop: iterate folds, aggregate results.

    Args:
        fold_dirs: List of fold directory paths
        train_fold_fn: Callable(fold_dir, fold_idx, fold_output_dir, args) -> fold_result dict
        run_dir: Timestamped run directory
        class_names: List of class name strings for aggregation
        args: Parsed command-line arguments
        cv_dir: CV folds root directory (used to compute fold_output_dir)
        single_fold: If not None, train only this fold index
        module_name: Module name for log messages (e.g. "src.training.train_cnn_cls_cv")

    Returns:
        0 on success, 1 on failure
    """
    n_folds = len(fold_dirs)

    # Mode: single fold
    if single_fold is not None:
        if single_fold < 0 or single_fold >= n_folds:
            logger.error(f"Invalid fold index {single_fold}. Must be 0-{n_folds-1}")
            return 1

        fold_dir = fold_dirs[single_fold]
        fold_output_dir = get_fold_output_dir(cv_dir, fold_dir, run_dir)
        train_fold_fn(fold_dir, single_fold, fold_output_dir, args)

        run_name = os.path.basename(run_dir)
        logger.info(f"\nFold {single_fold} training complete.")
        logger.info(f"Results saved to: {fold_output_dir}")
        logger.info(f"\nTo aggregate all results after training all folds, run:")
        logger.info(f"  python -m {module_name} -c {args.config} --run-name {run_name} --aggregate-only")
        return 0

    # Mode: train all folds sequentially
    logger.info(f"\n{'='*60}")
    logger.info("STARTING K-FOLD CROSS-VALIDATION TRAINING")
    logger.info(f"{'='*60}")

    all_fold_results = []
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            fold_output_dir = get_fold_output_dir(cv_dir, fold_dir, run_dir)
            result = train_fold_fn(fold_dir, fold_idx, fold_output_dir, args)
            all_fold_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if all_fold_results:
        aggregate_cv_results(run_dir, all_fold_results, class_names)
    else:
        logger.error("No folds completed successfully")
        return 1

    logger.info(f"\nCross-validation training complete!")
    logger.info(f"Results saved to: {run_dir}")

    return 0
