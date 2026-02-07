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
import json
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import classification_report

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.models.direct_cnn_classifier import DirectCNNClassifier
from src.training.train_cnn_cls import DirectClassifierTrainer, CLASS_NAMES
from src.training.cv_utils import (
    load_fold_datasets,
    load_fold_info,
    collect_existing_fold_results,
    list_training_runs,
    aggregate_cv_results,
    resolve_cv_directory,
    discover_fold_dirs,
    resolve_run_name,
    setup_run_directory,
    run_cv_training_loop,
)

import utils.logging_config as logconf
logger = logconf.get_logger("TRAIN_CNN_CV")

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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


def train_single_fold(config, fold_dir, fold_idx, fold_output_dir, device, args):
    """Train a single fold and return results.

    Args:
        config: Configuration object
        fold_dir: Directory containing fold data (train_ds.pkl, etc.)
        fold_idx: Fold index
        fold_output_dir: Output directory for this fold's results
        device: Torch device
        args: Command line arguments
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING FOLD {fold_idx}")
    logger.info(f"{'='*60}")
    logger.info(f"Fold data directory: {fold_dir}")
    logger.info(f"Fold output directory: {fold_output_dir}")

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
    num_classes = len(CLASS_NAMES)
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

        # Per-class accuracy
        per_class_acc = {}
        for cls in range(num_classes):
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

    # Resolve CV directory
    try:
        cv_dir = resolve_cv_directory(config, args.cv_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Mode: list runs
    if args.list_runs:
        list_training_runs(cv_dir)
        return 0

    # Discover folds
    try:
        fold_dirs = discover_fold_dirs(cv_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    n_folds = len(fold_dirs)
    logger.info(f"Found {n_folds} folds in {cv_dir}")

    # Load CV config
    cv_config_path = os.path.join(cv_dir, 'cv_config.json')
    if os.path.exists(cv_config_path):
        with open(cv_config_path, 'r') as f:
            cv_run_config = json.load(f)
        logger.info(f"CV Strategy: {cv_run_config.get('strategy', 'unknown')}")

    # Resolve run name
    run_name = resolve_run_name(args.run_name, prefix="run")
    run_dir = os.path.join(cv_dir, run_name)

    # Mode: aggregate only
    if args.aggregate_only:
        if not os.path.exists(run_dir):
            logger.error(f"Run directory not found: {run_dir}")
            logger.error("Specify an existing run with --run-name or train first")
            list_training_runs(cv_dir)
            return 1

        logger.info(f"Aggregating results from: {run_dir}")
        fold_results = collect_existing_fold_results(run_dir)
        if fold_results:
            aggregate_cv_results(run_dir, fold_results, CLASS_NAMES)
        else:
            logger.error("No fold results found to aggregate")
            return 1
        return 0

    # Setup run directory
    run_dir = setup_run_directory(cv_dir, run_name, args.config, extra_metadata={
        'n_folds': n_folds,
        'fold': args.fold,
        'restart': args.restart,
        'no_wandb': args.no_wandb,
    })

    # Create a closure that captures config and device
    def _train_fold(fold_dir, fold_idx, fold_output_dir, args):
        return train_single_fold(config, fold_dir, fold_idx, fold_output_dir, device, args)

    return run_cv_training_loop(
        fold_dirs=fold_dirs,
        train_fold_fn=_train_fold,
        run_dir=run_dir,
        class_names=CLASS_NAMES,
        args=args,
        cv_dir=cv_dir,
        single_fold=args.fold,
        module_name="src.training.train_cnn_cls_cv",
    )


if __name__ == '__main__':
    exit(main())
