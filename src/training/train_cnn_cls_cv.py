"""
CNN Classifier Training with K-Fold Cross-Validation

Trains a direct CNN classifier across multiple CV folds and aggregates results.
Requires pre-computed folds from: python -m src.data.precompute_kfolds

Features:
- Trains all folds sequentially or a single fold (for parallel execution)
- Aggregates results across folds (mean ± std)
- Per-fold checkpoints and metrics
- Combined CV results summary
- Timestamped run directories preserve results across multiple training runs

Usage:
    # Train all folds (creates timestamped run directory)
    python -m src.training.train_cnn_cls_cv --config config/config.yaml

    # Train with custom run name
    python -m src.training.train_cnn_cls_cv --config config/config.yaml --run-name experiment_v1

    # Train specific fold (for parallel/cluster execution)
    python -m src.training.train_cnn_cls_cv --config config/config.yaml --fold 0

    # Use custom CV directory
    python -m src.training.train_cnn_cls_cv --config config/config.yaml --cv-dir /path/to/cv_folds

    # Aggregate results only (after parallel fold training)
    python -m src.training.train_cnn_cls_cv --config config/config.yaml --run-name run_20260131_143052 --aggregate-only

    # List previous training runs
    python -m src.training.train_cnn_cls_cv --config config/config.yaml --list-runs
"""

import os
import sys
import argparse
import pickle
import json
import glob
import yaml
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.models.direct_cnn_classifier import DirectCNNClassifier
from src.training.train_cnn_cls import DirectClassifierTrainer, CLASS_NAMES

import utils.logging_config as logconf
logger = logconf.get_logger("TRAIN_CNN_CV")

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Load class config from centralized config
_label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
with open(_label_config_path) as _f:
    _label_config = yaml.safe_load(_f)
_classes_config = _label_config.get('classes', {})
_INCLUDE_NOISE = _classes_config.get('include_noise', True)
_NUM_CLASSES = 5 if _INCLUDE_NOISE else 4

# When noise excluded, labels are remapped 1,2,3,4 → 0,1,2,3 at training time
# So CLASS_NAMES maps remapped indices to original names
if _INCLUDE_NOISE:
    CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(5)]
else:
    CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(1, 5)]
# Default colors for up to 5 classes
DEFAULT_COLORS = ['#808080', '#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # gray, green, red, blue, orange


def load_fold_datasets(fold_dir):
    """Load train/val/test datasets from a fold directory."""
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

    return train_ds, val_ds, test_ds


def load_fold_info(fold_dir):
    """Load fold metadata."""
    info_path = os.path.join(fold_dir, 'fold_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def create_model(config, train_ds):
    """Create a fresh model instance with dimensions inferred from data."""
    # Expected dimensions from config (used for validation)
    expected_pulses = config.preprocess.tokenization.window

    # Load label config for num_classes
    label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
    with open(label_config_path) as f:
        label_config = yaml.safe_load(f)
    include_noise = label_config['classes'].get('include_noise', True)
    num_classes = 5 if include_noise else 4
    logger.info(f"Label config: include_noise={include_noise}, num_classes={num_classes}")

    # Get CNN config
    cnn_config = config.ml.get('cnn', {})
    width_multiplier = cnn_config.get('width_multiplier', 1)

    # Auto-compute kernel_scale from decimation factor if not explicitly set
    # Formula: kernel_scale = 10 / decimation_factor (to maintain ~10% receptive field)
    kernel_scale = cnn_config.get('kernel_scale', None)
    if kernel_scale is None:
        decimation_factor = config.preprocess.signal.decimation.get('factor', 10)
        valid_decimation_factors = {1, 2, 5, 10}
        if decimation_factor not in valid_decimation_factors:
            raise ValueError(
                f"Invalid decimation factor: {decimation_factor}. "
                f"Must be one of {sorted(valid_decimation_factors)} for kernel_scale auto-computation. "
                f"Alternatively, set ml.cnn.kernel_scale explicitly in config."
            )
        kernel_scale = 10 // decimation_factor
        logger.info(f"Auto-computed kernel_scale={kernel_scale} from decimation_factor={decimation_factor}")

    # Get dropout settings from training.regularization.dropout
    training_config = config.ml.get('training', {})
    reg_config = training_config.get('regularization', {})
    dropout_config = reg_config.get('dropout', {})
    spatial_dropout = dropout_config.get('spatial', 0.1)
    fc_dropout = dropout_config.get('fc', 0.5)

    # Infer input dimensions from data
    sample = train_ds[0]
    if isinstance(sample, dict):
        sample_shape = sample['tokens'].shape
    else:
        sample_shape = sample[0].shape

    # Handle both batched (B, C, Depth, Pulses) and unbatched (C, Depth, Pulses) datasets
    if len(sample_shape) == 4:  # Batched: (B, C, Depth, Pulses)
        in_channels = sample_shape[1]
        input_depth = sample_shape[2]
        input_pulses = sample_shape[3]
    elif len(sample_shape) == 3:  # Unbatched: (C, Depth, Pulses)
        in_channels = sample_shape[0]
        input_depth = sample_shape[1]
        input_pulses = sample_shape[2]
    else:
        raise ValueError(f"Unexpected data shape: {sample_shape}, expected 3D or 4D")

    logger.info(f"Inferred in_channels={in_channels}, input_depth={input_depth}, input_pulses={input_pulses} from data shape {sample_shape}")

    # Validate against config
    if input_pulses != expected_pulses:
        logger.warning(f"Data pulses ({input_pulses}) != config window ({expected_pulses}). Using data value.")

    logger.info(f"CNN config: width_multiplier={width_multiplier}, kernel_scale={kernel_scale}, dropout={fc_dropout}, spatial_dropout={spatial_dropout}")

    model = DirectCNNClassifier(
        in_channels=in_channels,
        input_pulses=input_pulses,
        input_depth=input_depth,
        num_classes=num_classes,
        dropout=fc_dropout,
        spatial_dropout=spatial_dropout,
        width_multiplier=width_multiplier,
        kernel_scale=kernel_scale
    )
    return model


def train_single_fold(config, fold_dir, fold_idx, run_dir, device, args):
    """Train a single fold and return results.

    Args:
        config: Configuration object
        fold_dir: Directory containing fold data (train_ds.pkl, etc.)
        fold_idx: Fold index
        run_dir: Timestamped run directory for outputs
        device: Torch device
        args: Command line arguments
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING FOLD {fold_idx}")
    logger.info(f"{'='*60}")
    logger.info(f"Fold data directory: {fold_dir}")
    logger.info(f"Run output directory: {run_dir}")

    # Load fold info
    fold_info = load_fold_info(fold_dir)
    if fold_info:
        strategy = fold_info.get('strategy', 'unknown')
        if strategy == 'experiment_kfold':
            logger.info(f"Strategy: {strategy}, holdout experiments: {len(fold_info.get('test_val_experiments', []))}")
        elif strategy == 'session_loso':
            logger.info(f"Strategy: {strategy}, holdout session: {fold_info.get('holdout_session')}")
        elif strategy == 'participant_lopo':
            logger.info(f"Strategy: {strategy}, holdout participant: {fold_info.get('holdout_participant')}")
        elif strategy == 'participant_within':
            logger.info(f"Strategy: {strategy}, participant: {fold_info.get('participant')} "
                       f"({fold_info.get('n_experiments', '?')} experiments)")

    # Load datasets
    train_ds, val_ds, test_ds = load_fold_datasets(fold_dir)
    logger.info(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")

    # Create data loaders
    resource_cfg = config.get_resource_config()
    train_loader = DataLoader(
        train_ds, batch_size=None, shuffle=True,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=None, shuffle=False,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )
    test_loader = DataLoader(
        test_ds, batch_size=None, shuffle=False,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )

    # Create fresh model for this fold (infer dimensions from data)
    model = create_model(config, train_ds)
    model.print_architecture()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Output directory for this fold (inside timestamped run directory)
    fold_output_dir = os.path.join(run_dir, f'fold_{fold_idx}')
    os.makedirs(fold_output_dir, exist_ok=True)

    # Create trainer
    trainer = DirectClassifierTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=fold_output_dir,
        use_wandb=not args.no_wandb
    )

    # Override WandB run name to include fold
    if trainer.use_wandb and WANDB_AVAILABLE:
        # Will be set in init_wandb, but we can modify the config
        pass

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        restart=args.restart
    )

    # Collect results
    fold_results = {
        'fold_idx': fold_idx,
        'fold_dir': fold_dir,
        'fold_info': fold_info,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'total_epochs': trainer.current_epoch + 1,
    }

    # Load test predictions for detailed metrics
    test_pred_path = os.path.join(fold_output_dir, 'test_predictions.npz')
    if os.path.exists(test_pred_path):
        test_data = np.load(test_pred_path)
        predictions = test_data['predictions']
        labels = test_data['labels']

        # Compute metrics
        fold_results['test_accuracy'] = (predictions == labels).mean()
        fold_results['test_predictions'] = predictions.tolist()
        fold_results['test_labels'] = labels.tolist()

        # Per-class accuracy (use _NUM_CLASSES for dynamic class count)
        per_class_acc = {}
        for cls in range(_NUM_CLASSES):
            mask = labels == cls
            if mask.sum() > 0:
                per_class_acc[CLASS_NAMES[cls]] = float((predictions[mask] == cls).mean())
            else:
                per_class_acc[CLASS_NAMES[cls]] = 0.0
        fold_results['per_class_accuracy'] = per_class_acc

    # Save fold results
    results_path = os.path.join(fold_output_dir, 'fold_results.json')
    with open(results_path, 'w') as f:
        # Remove non-serializable items
        serializable_results = {k: v for k, v in fold_results.items()
                               if k not in ['test_predictions', 'test_labels']}
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nFold {fold_idx} complete:")
    logger.info(f"  Best val accuracy: {fold_results['best_val_acc']:.2%}")
    if 'test_accuracy' in fold_results:
        logger.info(f"  Test accuracy: {fold_results['test_accuracy']:.2%}")

    return fold_results


def aggregate_cv_results(run_dir, fold_results):
    """Aggregate results across all folds.

    Args:
        run_dir: Timestamped run directory where results will be saved
        fold_results: List of fold result dictionaries
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
    per_class_accs = {name: [] for name in CLASS_NAMES}
    for r in fold_results:
        if 'per_class_accuracy' in r:
            for name in CLASS_NAMES:
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
    create_cv_summary_plot(run_dir, summary, fold_results)

    return summary


def create_cv_summary_plot(run_dir, summary, fold_results):
    """Create visualization of CV results.

    Args:
        run_dir: Timestamped run directory where plot will be saved
        summary: Aggregated summary statistics
        fold_results: List of fold result dictionaries
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
    if is_participant_within:
        # Use participant names from fold_results
        x_labels = []
        for r in fold_results:
            participant = r.get('fold_info', {}).get('participant', f'P{r["fold_idx"]}')
            # Shorten long participant names
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
    class_names = list(summary['per_class_accuracy'].keys())
    means = [summary['per_class_accuracy'][n]['mean'] for n in class_names]
    stds = [summary['per_class_accuracy'][n]['std'] for n in class_names]

    colors = DEFAULT_COLORS[:len(class_names)]
    bars = ax.bar(class_names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Test Accuracy (Mean ± Std)')
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
        # Add top participants from ranking
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
            stats_text += f"  {name}: {data['mean']:.2%} ± {data['std']:.2%}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(run_dir, 'cv_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot saved to: {plot_path}")


def collect_existing_fold_results(run_dir, n_folds):
    """Collect results from already-trained folds in a run directory.

    Args:
        run_dir: Timestamped run directory containing fold_0/, fold_1/, etc.
        n_folds: Number of folds to look for
    """
    fold_results = []

    for fold_idx in range(n_folds):
        results_path = os.path.join(run_dir, f'fold_{fold_idx}', 'fold_results.json')

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                result = json.load(f)
                result['fold_idx'] = fold_idx
                fold_results.append(result)
            logger.info(f"Loaded results for fold {fold_idx}")
        else:
            logger.warning(f"No results found for fold {fold_idx} at {results_path}")

    # Sort by fold index
    fold_results.sort(key=lambda x: x['fold_idx'])

    return fold_results


def list_training_runs(cv_dir):
    """List all training runs in the CV directory."""
    run_dirs = sorted(glob.glob(os.path.join(cv_dir, 'run_*')))

    if not run_dirs:
        logger.info(f"No training runs found in {cv_dir}")
        return []

    logger.info(f"\nTraining runs in {cv_dir}:")
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
            # Count completed folds
            completed_folds = len(glob.glob(os.path.join(run_dir, 'fold_*/fold_results.json')))
            status = f"In progress ({completed_folds} folds completed)"

        logger.info(f"  {run_name}: {status}")
        runs.append({'name': run_name, 'path': run_dir, 'status': status})

    return runs


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN Classifier with K-Fold Cross-Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all folds (creates timestamped run directory)
  python -m src.training.train_cnn_cls_cv --config config/config.yaml

  # Train with custom run name
  python -m src.training.train_cnn_cls_cv --config config/config.yaml --run-name experiment_v1

  # Train specific fold (for parallel execution)
  python -m src.training.train_cnn_cls_cv --config config/config.yaml --fold 0

  # Aggregate results after parallel training (specify run name)
  python -m src.training.train_cnn_cls_cv --config config/config.yaml --run-name run_20260131_143052 --aggregate-only

  # List previous training runs
  python -m src.training.train_cnn_cls_cv --config config/config.yaml --list-runs

  # Custom CV directory
  python -m src.training.train_cnn_cls_cv --config config/config.yaml --cv-dir /path/to/cv_folds
        """
    )
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--cv-dir', type=str, default=None,
                        help='Path to CV folds directory (overrides config)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run (default: run_YYYYMMDD_HHMMSS)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Train only this specific fold (for parallel execution)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate results from existing fold training')
    parser.add_argument('--list-runs', action='store_true',
                        help='List all previous training runs and exit')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--restart', '-r', action='store_true',
                        help='Restart from latest checkpoint')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config, create_dirs=False)
    setup_environment(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Determine CV directory
    if args.cv_dir:
        cv_dir = args.cv_dir
    else:
        data_root = config.get_train_data_root()
        cv_config = getattr(config, 'cross_validation', None)
        if cv_config:
            output_subdir = getattr(cv_config, 'output_subdir', 'cv_folds')
        else:
            output_subdir = 'cv_folds'
        cv_dir = os.path.join(data_root, output_subdir)

    if not os.path.exists(cv_dir):
        logger.error(f"CV directory not found: {cv_dir}")
        logger.error("Run 'python -m src.data.precompute_kfolds --config config/config.yaml' first")
        return 1

    # Mode: list runs
    if args.list_runs:
        list_training_runs(cv_dir)
        return 0

    # Find fold directories (supports both 'fold_*' and 'P*_fold*' naming)
    fold_dirs = sorted(glob.glob(os.path.join(cv_dir, 'fold_*')))
    if not fold_dirs:
        # Try nested CV naming: P{id}_fold{k}
        fold_dirs = sorted(glob.glob(os.path.join(cv_dir, 'P*_fold*')))
    if not fold_dirs:
        logger.error(f"No fold directories found in {cv_dir}")
        return 1

    n_folds = len(fold_dirs)
    logger.info(f"Found {n_folds} folds in {cv_dir}")

    # Load CV config
    cv_config_path = os.path.join(cv_dir, 'cv_config.json')
    if os.path.exists(cv_config_path):
        with open(cv_config_path, 'r') as f:
            cv_run_config = json.load(f)
        logger.info(f"CV Strategy: {cv_run_config.get('strategy', 'unknown')}")

    # Determine run directory
    if args.run_name:
        # Use provided run name (with or without run_ prefix)
        run_name = args.run_name if args.run_name.startswith('run_') else f'run_{args.run_name}'
    else:
        # Generate timestamped run name
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = os.path.join(cv_dir, run_name)

    # Mode: aggregate only
    if args.aggregate_only:
        if not os.path.exists(run_dir):
            logger.error(f"Run directory not found: {run_dir}")
            logger.error("Specify an existing run with --run-name or train first")
            # List available runs
            list_training_runs(cv_dir)
            return 1

        logger.info(f"Aggregating results from: {run_dir}")
        fold_results = collect_existing_fold_results(run_dir, n_folds)
        if fold_results:
            aggregate_cv_results(run_dir, fold_results)
        else:
            logger.error("No fold results found to aggregate")
            return 1
        return 0

    # Create run directory
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Save run config for reference
    run_config_path = os.path.join(run_dir, 'run_config.json')
    with open(run_config_path, 'w') as f:
        json.dump({
            'run_name': run_name,
            'started_at': datetime.now().isoformat(),
            'config_file': os.path.abspath(args.config),
            'n_folds': n_folds,
            'fold': args.fold,
            'restart': args.restart,
            'no_wandb': args.no_wandb,
        }, f, indent=2)

    # Mode: single fold
    if args.fold is not None:
        if args.fold < 0 or args.fold >= n_folds:
            logger.error(f"Invalid fold index {args.fold}. Must be 0-{n_folds-1}")
            return 1

        fold_dir = fold_dirs[args.fold]
        fold_results = [train_single_fold(config, fold_dir, args.fold, run_dir, device, args)]

        logger.info(f"\nFold {args.fold} training complete.")
        logger.info(f"Results saved to: {run_dir}/fold_{args.fold}/")
        logger.info(f"\nTo aggregate all results after training all folds, run:")
        logger.info(f"  python -m src.training.train_cnn_cls_cv -c {args.config} --run-name {run_name} --aggregate-only")
        return 0

    # Mode: train all folds sequentially
    logger.info(f"\n{'='*60}")
    logger.info("STARTING K-FOLD CROSS-VALIDATION TRAINING")
    logger.info(f"{'='*60}")

    all_fold_results = []
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            result = train_single_fold(config, fold_dir, fold_idx, run_dir, device, args)
            all_fold_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if all_fold_results:
        aggregate_cv_results(run_dir, all_fold_results)
    else:
        logger.error("No folds completed successfully")
        return 1

    logger.info(f"\nCross-validation training complete!")
    logger.info(f"Results saved to: {run_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
