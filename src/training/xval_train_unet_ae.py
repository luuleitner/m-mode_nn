"""
UNet Autoencoder Training with K-Fold Cross-Validation

Trains UNet with classification head across multiple CV folds and aggregates results.
Uses unified loss functions: reconstruction (unweighted) + classification (weighted).

Features:
- Trains all folds sequentially or a single fold (for parallel execution)
- Aggregates results across folds (mean ± std)
- Per-fold checkpoints and metrics
- Per-class metrics tracking
- Combined CV results summary
- Timestamped run directories preserve results across multiple training runs

Usage:
    # Train all folds
    python -m src.training.train_unet_ae_cv --config config/config.yaml

    # Train specific fold
    python -m src.training.train_unet_ae_cv --config config/config.yaml --fold 0

    # Aggregate results only
    python -m src.training.train_unet_ae_cv --config config/config.yaml --run-name run_YYYYMMDD --aggregate-only

    # List previous training runs
    python -m src.training.train_unet_ae_cv --config config/config.yaml --list-runs
"""

import os
import sys
import argparse
import json

import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.training.base_trainer import BaseTrainer
from src.training.adapters import CNNAdapter
from src.training.callbacks import (
    CheckpointCallback,
    VisualizationCallback,
    WandBCallback,
    EarlyStoppingCallback
)
from src.training.train_unet_ae import (
    create_model,
    create_adapter,
    compute_class_weights,
    CLASS_NAMES
)
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
logger = logconf.get_logger("TRAIN_UNET_CV")


def create_callbacks(config, results_dir, test_loader=None, num_classes=0):
    """Create training callbacks with enhanced visualization."""
    callbacks = []

    # Checkpoint callback
    checkpoint_config = getattr(config.ml.training, 'checkpointing', None)
    callbacks.append(CheckpointCallback(
        save_dir=results_dir,
        save_best=getattr(checkpoint_config, 'save_best', True) if checkpoint_config else True,
        save_every_n_epochs=getattr(checkpoint_config, 'save_every_n_epochs', 10) if checkpoint_config else 10,
        save_restart_every=getattr(config.ml.training.restart, 'save_restart_every', 5),
        keep_n_checkpoints=3
    ))

    # Visualization callback (with GT/Pred/Diff plots)
    validation_config = getattr(config.ml.training, 'validation', None)
    plot_every = getattr(validation_config, 'plot_every_n_epochs', 10) if validation_config else 10
    callbacks.append(VisualizationCallback(
        save_dir=results_dir,
        plot_every_n_epochs=plot_every,
        test_loader=test_loader,
        class_names=CLASS_NAMES[:num_classes] if num_classes > 0 else CLASS_NAMES
    ))

    # WandB callback for CV (disabled by default to avoid multiple runs)
    # Individual folds can be logged if needed
    if config.wandb.use_wandb:
        from src.training.callbacks import WandBCallback
        loss_weights = config.get_loss_weights()
        wandb_config = {
            'model_type': config.ml.model.type,
            'embedding_dim': config.ml.model.embedding_dim,
            'num_classes': num_classes,
            'cls_weight': loss_weights.get('classification_weight', 0.0),
        }
        callbacks.append(WandBCallback(
            project=config.wandb.project + '_cv',
            config=wandb_config,
            name=None,  # Will be set per fold
            save_dir=results_dir,
            api_key=getattr(config.wandb, 'api_key', None),
            class_names=CLASS_NAMES[:num_classes] if num_classes > 0 else [],
            plot_confusion_every=plot_every
        ))

    # Early stopping callback
    early_stop_config = getattr(config.ml.training, 'early_stopping', None)
    if early_stop_config and getattr(early_stop_config, 'enabled', False):
        callbacks.append(EarlyStoppingCallback(
            patience=getattr(early_stop_config, 'patience', 20),
            min_delta=getattr(early_stop_config, 'min_delta', 1e-5),
            monitor=getattr(early_stop_config, 'monitor', 'val_loss')
        ))

    return callbacks


def train_single_fold(config, fold_dir, fold_idx, fold_output_dir, device, args):
    """Train a single fold and return results."""
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

    # Load datasets (with optional on-the-fly augmentation)
    aug_config = config.get_train_augmentation_config()
    train_ds, val_ds, test_ds = load_fold_datasets(fold_dir, augmentation_config=aug_config)
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

    # Create model and adapter
    model = create_model(config)
    adapter = create_adapter(config)
    num_classes = getattr(model, 'num_classes', 0)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(fold_output_dir, exist_ok=True)

    # Create callbacks with class names
    callbacks = create_callbacks(config, fold_output_dir, test_loader, num_classes=num_classes)

    # Create trainer
    trainer = BaseTrainer(
        model=model,
        adapter=adapter,
        callbacks=callbacks,
        device=device,
        results_dir=fold_output_dir
    )

    # Update callbacks with test loader
    for cb in trainer.callbacks.callbacks:
        if isinstance(cb, VisualizationCallback):
            cb.set_test_loader(test_loader)

    # Get training parameters
    loss_weights = config.get_loss_weights()
    regularization = config.get_regularization_config()

    # Rename classification_weight to cls_weight
    if 'classification_weight' in loss_weights:
        loss_weights['cls_weight'] = loss_weights.pop('classification_weight')

    # Set up class weights
    num_classes = getattr(model, 'num_classes', 0)
    if num_classes > 0:
        trainer.num_classes = num_classes
        class_weights_dict = compute_class_weights(train_loader, num_classes, config, trainer.device)
        if class_weights_dict:
            trainer.set_class_weights(class_weights_dict, num_classes)

    # Build scheduler config from YAML (pass extra params like warmup_epochs, eta_min)
    sched_cfg = config.ml.training.lr_scheduler
    scheduler_config = {
        k: getattr(sched_cfg, k) for k in dir(sched_cfg)
        if not k.startswith('_') and k != 'type'
    } if sched_cfg else {}

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.ml.training.epochs,
        learning_rate=config.ml.training.lr,
        weight_decay=config.ml.training.weight_decay,
        optimizer_type=config.ml.training.optimizer.type,
        scheduler_type=config.ml.training.lr_scheduler.type,
        scheduler_config=scheduler_config,
        loss_weights=loss_weights,
        grad_clip_norm=regularization['grad_clip_norm'],
        restart=args.restart
    )

    # Final evaluation
    test_metrics = trainer.evaluate(test_loader, loss_weights=loss_weights)

    # Plot test confusion matrix (must be done AFTER evaluate() sets _last_test_*)
    test_preds = getattr(trainer, '_last_test_predictions', None)
    test_labels = getattr(trainer, '_last_test_labels', None)
    if test_preds is not None and test_labels is not None:
        final_epoch = len(history['train_loss']) - 1
        for cb in trainer.callbacks.callbacks:
            # Local visualization callback
            if isinstance(cb, VisualizationCallback):
                cb._plot_single_confusion_matrix(
                    epoch=final_epoch,
                    predictions=test_preds,
                    labels=test_labels,
                    split='test',
                    prefix='final_'
                )
            # WandB callback
            if isinstance(cb, WandBCallback) and cb.enabled:
                cb._plot_confusion_matrix(
                    epoch=final_epoch,
                    predictions=test_preds,
                    labels=test_labels,
                    split='test'
                )
        logger.info(f"Test confusion matrix saved for fold {fold_idx}")

    # Collect results
    fold_results = {
        'fold_idx': fold_idx,
        'fold_dir': fold_dir,
        'fold_info': fold_info,
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
        'best_val_acc': max(history['val_accuracy']) if history.get('val_accuracy') else 0.0,
        'total_epochs': len(history['train_loss']),
    }

    if 'test_accuracy' in test_metrics:
        fold_results['test_accuracy'] = test_metrics['test_accuracy']
        fold_results['test_balanced_accuracy'] = test_metrics.get('test_balanced_accuracy', 0.0)

    if 'per_class_accuracy' in test_metrics:
        fold_results['per_class_accuracy'] = {
            CLASS_NAMES[k] if k < len(CLASS_NAMES) else f'class_{k}': v
            for k, v in test_metrics['per_class_accuracy'].items()
        }

    # Save predictions
    if 'predictions' in test_metrics:
        np.savez(
            os.path.join(fold_output_dir, 'test_predictions.npz'),
            predictions=test_metrics['predictions'],
            labels=test_metrics['labels'],
            probabilities=test_metrics.get('probabilities', [])
        )

    # Save fold results
    results_path = os.path.join(fold_output_dir, 'fold_results.json')
    with open(results_path, 'w') as f:
        serializable = {k: v for k, v in fold_results.items()
                       if not isinstance(v, np.ndarray)}
        json.dump(serializable, f, indent=2)

    logger.info(f"\nFold {fold_idx} complete:")
    logger.info(f"  Best val accuracy: {fold_results['best_val_acc']:.2%}")
    if 'test_accuracy' in fold_results:
        logger.info(f"  Test accuracy: {fold_results['test_accuracy']:.2%}")

    return fold_results


def main():
    parser = argparse.ArgumentParser(
        description='Train UNet with K-Fold CV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all folds
  python -m src.training.train_unet_ae_cv --config config/config.yaml

  # Train specific fold
  python -m src.training.train_unet_ae_cv --config config/config.yaml --fold 0

  # Aggregate results after parallel training
  python -m src.training.train_unet_ae_cv --config config/config.yaml --run-name unet_20260131_143022 --aggregate-only

  # List previous training runs
  python -m src.training.train_unet_ae_cv --config config/config.yaml --list-runs
        """
    )
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--cv-dir', type=str, default=None, help='Path to CV folds directory')
    parser.add_argument('--run-name', type=str, default=None, help='Name for this training run')
    parser.add_argument('--fold', type=int, default=None, help='Train only this fold')
    parser.add_argument('--aggregate-only', action='store_true', help='Only aggregate existing results')
    parser.add_argument('--list-runs', action='store_true', help='List all previous training runs and exit')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--restart', '-r', nargs='?', const=True, default=False,
                        help='Restart from checkpoint. Optionally specify path: --restart /path/to/ckpt.pth')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config, create_dirs=False)
    setup_environment(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Resolve CV directory (auto-precompute if missing)
    try:
        cv_dir = resolve_cv_directory(config, args.cv_dir,
                                      auto_precompute=True, config_path=args.config)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return 1

    # Mode: list runs
    if args.list_runs:
        data_root = config.get_train_data_root()
        runs_dir = os.path.join(data_root, 'runs')
        list_training_runs(runs_dir)
        return 0

    # Discover folds
    try:
        fold_dirs = discover_fold_dirs(cv_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    n_folds = len(fold_dirs)
    logger.info(f"Found {n_folds} folds in {cv_dir}")

    # Resolve run name
    data_root = config.get_train_data_root()
    run_name = resolve_run_name(args.run_name, prefix="unet")
    run_dir = os.path.join(data_root, 'runs', run_name)

    # Mode: aggregate only
    if args.aggregate_only:
        if not os.path.exists(run_dir):
            logger.error(f"Run directory not found: {run_dir}")
            logger.error("Specify an existing run with --run-name or train first")
            runs_dir = os.path.join(data_root, 'runs')
            list_training_runs(runs_dir)
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
    run_dir = setup_run_directory(data_root, run_name, args.config, extra_metadata={
        'n_folds': n_folds,
        'fold': args.fold,
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
        module_name="src.training.train_unet_ae_cv",
    )


if __name__ == '__main__':
    exit(main())
