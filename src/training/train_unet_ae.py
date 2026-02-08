"""
UNet Autoencoder Training

Training script for UNetAutoencoder with optional classification head.
Uses unified loss functions: reconstruction (unweighted) + classification (weighted).

Features:
- Joint reconstruction + classification loss
- Class-weighted classification (reconstruction never weighted)
- Per-class metrics tracking
- Soft label support
- Early stopping on balanced accuracy

Usage:
    python -m src.training.train_unet_ae --config config/config.yaml
    python -m src.training.train_unet_ae --config config/config.yaml --restart
"""

import os
import sys
import argparse
import pickle
import yaml
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.data.datasets import create_filtered_split_datasets

from src.training.base_trainer import BaseTrainer
from src.training.adapters import CNNAdapter
from src.training.callbacks import (
    CheckpointCallback,
    VisualizationCallback,
    WandBCallback,
    EarlyStoppingCallback
)

import utils.logging_config as logconf
logger = logconf.get_logger("TRAIN")

# Load class names from centralized config
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


def compute_class_weights(train_loader, num_classes, config, device):
    """
    Compute class weights from training data distribution.

    Mirrors CNN classifier's implementation with multiple methods:
    - inverse_frequency: standard total / (n_classes * count)
    - effective_samples: Class-Balanced Loss (Cui et al., 2019)
    - sqrt_inverse: gentler weighting
    - custom: user-defined weights

    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        config: Configuration object
        device: Torch device

    Returns:
        dict: {class_idx: weight} or None if disabled
    """
    # Get imbalance config
    train_cfg = config.ml.training
    imbalance_cfg = train_cfg.get('imbalance', None) if hasattr(train_cfg, 'get') else getattr(train_cfg, 'imbalance', None)

    if not imbalance_cfg:
        return None

    weights_cfg = imbalance_cfg.get('class_weights', None) if hasattr(imbalance_cfg, 'get') else getattr(imbalance_cfg, 'class_weights', None)
    if not weights_cfg:
        return None

    enabled = weights_cfg.get('enabled', False) if hasattr(weights_cfg, 'get') else getattr(weights_cfg, 'enabled', False)
    if not enabled:
        return None

    method = weights_cfg.get('method', 'inverse_frequency') if hasattr(weights_cfg, 'get') else getattr(weights_cfg, 'method', 'inverse_frequency')
    power = weights_cfg.get('power', 1.0) if hasattr(weights_cfg, 'get') else getattr(weights_cfg, 'power', 1.0)
    custom_weights = weights_cfg.get('custom_weights', None) if hasattr(weights_cfg, 'get') else getattr(weights_cfg, 'custom_weights', None)

    logger.info(f"Computing class weights (method={method}, power={power})...")

    # Count classes
    class_counts = torch.zeros(num_classes)
    for batch in tqdm(train_loader, desc="Counting classes"):
        if isinstance(batch, dict):
            labels = batch['labels']
        else:
            labels = batch[1]

        # Handle soft labels
        if labels.dim() > 1 and labels.shape[-1] > 1:
            hard_labels = labels.argmax(dim=-1)
        else:
            hard_labels = labels.squeeze() if labels.dim() > 1 else labels

        for cls in range(num_classes):
            class_counts[cls] += (hard_labels == cls).sum().item()

    total = class_counts.sum()
    n_classes = len(class_counts)
    max_ratio = class_counts.max() / (class_counts.min() + 1e-6)

    # Compute weights based on method
    if method == "custom" and custom_weights is not None:
        weights = torch.tensor(custom_weights, dtype=torch.float32)
        logger.info(f"Using custom weights: {custom_weights}")
    elif method == "effective_samples":
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        logger.info(f"Using effective_samples method (β={beta})")
    elif method == "sqrt_inverse":
        weights = torch.sqrt(total / (class_counts + 1e-6))
        logger.info("Using sqrt_inverse method")
    else:  # inverse_frequency
        weights = total / (n_classes * class_counts + 1e-6)
        logger.info("Using inverse_frequency method")

    # Apply power scaling
    if power != 1.0:
        weights = weights ** power
        logger.info(f"Applied power scaling: {power}")

    # Normalize (average weight = 1)
    weights = weights / weights.sum() * n_classes

    logger.info(f"Class distribution: {class_counts.long().tolist()} (max/min ratio: {max_ratio:.1f})")
    logger.info(f"Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    # Return as dict for compatibility
    return {i: weights[i].item() for i in range(num_classes)}


def create_model(config):
    """Create model based on config."""
    model_type = config.ml.model.type
    reg_config = config.get_regularization_config()
    use_batchnorm = reg_config.get('batch_norm', True)

    # Check if classification is enabled (classification_weight > 0)
    loss_weights = config.get_loss_weights()
    classification_weight = loss_weights.get('classification_weight', 0)

    # Load num_classes from label config if classification is enabled
    if classification_weight > 0:
        label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
        with open(label_config_path) as f:
            label_config = yaml.safe_load(f)
        include_noise = label_config['classes'].get('include_noise', True)
        num_classes = 5 if include_noise else 4
        logger.info(f"Joint training enabled: classification_weight={classification_weight}, num_classes={num_classes}, include_noise={include_noise}")
    else:
        num_classes = 0

    # Input dimensions after adapter transpose: [B, C, Pulses, Depth]
    # Transpose preserves temporal resolution through encoder pooling
    input_pulses = config.preprocess.tokenization.window  # H dimension (temporal)
    input_depth = 130  # W dimension (spatial, after decimation)

    if model_type == "UNetAutoencoder":
        from src.models.unet_ae import UNetAutoencoder

        model = UNetAutoencoder(
            in_channels=3,
            input_height=input_pulses,  # Pulses (temporal)
            input_width=input_depth,     # Depth (spatial)
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=use_batchnorm,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Created model: {model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def create_adapter(config):
    """Create adapter based on model type."""
    model_type = config.ml.model.type

    if model_type == "UNetAutoencoder":
        return CNNAdapter()
    else:
        raise ValueError(f"No adapter for model type: {model_type}")


def create_callbacks(config, results_dir, test_loader=None, num_classes=0):
    """Create training callbacks with enhanced WandB logging."""
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

    vis_callback = VisualizationCallback(
        save_dir=results_dir,
        plot_every_n_epochs=plot_every,
        test_loader=test_loader,
        class_names=CLASS_NAMES[:num_classes] if num_classes > 0 else CLASS_NAMES
    )
    callbacks.append(vis_callback)

    # WandB callback with enhanced logging
    if config.wandb.use_wandb:
        # Get loss weights for logging
        loss_weights = config.get_loss_weights()

        wandb_config = {
            'model_type': config.ml.model.type,
            'embedding_dim': config.ml.model.embedding_dim,
            'channels': config.ml.model.channels_per_layer,
            'epochs': config.ml.training.epochs,
            'learning_rate': config.ml.training.lr,
            'optimizer': config.ml.training.optimizer.type,
            'scheduler': config.ml.training.lr_scheduler.type,
            'num_classes': num_classes,
            'mse_weight': loss_weights.get('mse_weight', 0.5),
            'l1_weight': loss_weights.get('l1_weight', 0.5),
            'cls_weight': loss_weights.get('classification_weight', 0.0),
        }

        callbacks.append(WandBCallback(
            project=config.wandb.project,
            config=wandb_config,
            name=config.wandb.name,
            save_dir=results_dir,
            api_key=getattr(config.wandb, 'api_key', None),
            class_names=CLASS_NAMES[:num_classes] if num_classes > 0 else [],
            plot_confusion_every=plot_every,
            log_every_n_batches=10
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


def load_or_create_datasets(config):
    """Load datasets from pickle or create new."""
    pickle_path = config.get_train_data_root()
    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    if config.ml.loading.load_data_pickle:
        logger.info("Loading datasets from pickle...")
        with open(train_path, 'rb') as f:
            train_ds = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_ds = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_ds = pickle.load(f)
    else:
        logger.info("Creating datasets from metadata...")
        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            **config.get_dataset_parameters()
        )

        # Save for future use
        logger.info("Saving datasets to pickle...")
        with open(train_path, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_ds, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_ds, f)

    logger.info(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, config):
    """
    Create data loaders for training.

    Note: Class balancing is now handled at dataset construction time
    (balance_classes=True in config), not at DataLoader level.
    The train dataset already contains balanced batches if enabled.
    """
    resource_config = config.get_resource_config()

    # Check if dataset has balanced batches
    if hasattr(train_ds, 'balance_classes') and train_ds.balance_classes:
        logger.info("Using pre-balanced batches (balanced at dataset construction)")

    train_loader = DataLoader(
        train_ds,
        batch_size=None,  # Pre-batched
        shuffle=True,     # Shuffle batch order
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor']
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        shuffle=False,
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor']
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=None,
        shuffle=False,
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor']
    )

    return train_loader, val_loader, test_loader


def main(config_path, restart=False):
    """Main training function."""
    # Load config (create_dirs=True to create checkpoint directories)
    config = load_config(config_path, create_dirs=True)
    setup_environment(config)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data
    train_ds, val_ds, test_ds = load_or_create_datasets(config)

    # Attach on-the-fly augmentation
    aug_config = config.get_train_augmentation_config()
    if aug_config.get('enabled', False):
        from src.data.augmentations import SignalAugmenter
        aug_cfg = {k: v for k, v in aug_config.items() if k != 'enabled'}
        train_ds.set_general_augmenter(SignalAugmenter(config=aug_cfg))
        logger.info("On-the-fly training augmentation enabled")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )

    # Create model and adapter
    model = create_model(config)
    adapter = create_adapter(config)
    num_classes = getattr(model, 'num_classes', 0)

    # Get results directory
    results_base = config.get_checkpoint_path()

    # Create callbacks with class names for WandB
    callbacks = create_callbacks(config, results_base, test_loader, num_classes=num_classes)

    # Create trainer
    trainer = BaseTrainer(
        model=model,
        adapter=adapter,
        callbacks=callbacks,
        device=config.global_setting.run.device,
        results_dir=results_base
    )

    # Update all callbacks with the trainer's timestamped results directory
    # This ensures checkpoints, visualizations, and logs all go to the same run folder
    for cb in trainer.callbacks.callbacks:
        if isinstance(cb, VisualizationCallback):
            cb.set_test_loader(test_loader)
            cb.save_dir = trainer.results_dir
        if isinstance(cb, CheckpointCallback):
            cb.save_dir = trainer.results_dir
        if isinstance(cb, WandBCallback):
            cb.save_dir = trainer.results_dir

    # Get training parameters
    loss_weights = config.get_loss_weights()
    regularization = config.get_regularization_config()

    # Rename classification_weight to cls_weight for unified interface
    if 'classification_weight' in loss_weights:
        loss_weights['cls_weight'] = loss_weights.pop('classification_weight')

    # Determine num_classes from model
    num_classes = getattr(model, 'num_classes', 0)
    if num_classes > 0:
        trainer.num_classes = num_classes

        # Compute class weights using CNN-style computation
        class_weights_dict = compute_class_weights(train_loader, num_classes, config, trainer.device)
        if class_weights_dict:
            trainer.set_class_weights(class_weights_dict, num_classes)
            logger.info(f"Using class-weighted classification loss")

    # Build scheduler config from YAML
    sched_cfg = config.ml.training.lr_scheduler
    scheduler_config = {
        k: getattr(sched_cfg, k) for k in dir(sched_cfg)
        if not k.startswith('_') and k != 'type'
    } if sched_cfg else {}

    # Train
    try:
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
            restart=restart
        )

        # Final evaluation
        logger.info("Running final evaluation...")
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
            logger.info("Test confusion matrix saved")

        # Print summary
        print_summary(history, test_metrics, trainer.results_dir)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        # Emergency save handled by checkpoint callback

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def print_summary(history, test_metrics, results_dir):
    """Print training summary."""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss:   {history['val_loss'][-1]:.6f}")
    print(f"Final Val MSE:    {history['val_mse'][-1]:.6f}")
    print(f"Test MSE:         {test_metrics['test_mse']:.6f}")
    print(f"Test Loss:        {test_metrics.get('test_loss', 0):.6f}")

    # Print classification accuracy if available
    if 'val_accuracy' in history and history['val_accuracy']:
        print(f"Final Val Acc:    {history['val_accuracy'][-1]:.2%}")
    if 'val_balanced_accuracy' in history and history['val_balanced_accuracy']:
        print(f"Final Val BalAcc: {history['val_balanced_accuracy'][-1]:.2%}")
    if 'test_accuracy' in test_metrics:
        print(f"Test Accuracy:    {test_metrics['test_accuracy']:.2%}")
    if 'test_balanced_accuracy' in test_metrics:
        print(f"Test Balanced:    {test_metrics['test_balanced_accuracy']:.2%}")

    # Per-class test accuracy
    if 'per_class_accuracy' in test_metrics:
        print("\nPer-class Test Accuracy:")
        for cls, acc in test_metrics['per_class_accuracy'].items():
            cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            print(f"  {cls_name}: {acc:.2%}")

    print(f"\nTotal Epochs:     {len(history['train_loss'])}")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train autoencoder model')

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--restart', '-r',
        nargs='?', const=True, default=False,
        help='Restart from checkpoint. Optionally specify path: --restart /path/to/ckpt.pth'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config, restart=args.restart)
