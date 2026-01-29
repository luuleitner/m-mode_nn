"""
Training Entry Point

Unified training script for autoencoder models.
Uses config-driven model and adapter selection.

Usage:
    python -m src.training.train --config config/config.yaml
    python -m src.training.train --config config/config.yaml --restart
"""

import os
import sys
import argparse
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader

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


def create_model(config):
    """Create model based on config."""
    model_type = config.ml.model.type
    reg_config = config.get_regularization_config()
    use_batchnorm = reg_config.get('batch_norm', True)

    # Check if classification is enabled (classification_weight > 0)
    loss_weights = config.get_loss_weights()
    classification_weight = loss_weights.get('classification_weight', 0)
    num_classes = 3 if classification_weight > 0 else 0

    if num_classes > 0:
        logger.info(f"Joint training enabled: classification_weight={classification_weight}")

    # Input dimensions after adapter transpose: [B, C, Pulses, Depth]
    # Transpose preserves temporal resolution through encoder pooling
    input_pulses = config.preprocess.tokenization.window  # H dimension (temporal)
    input_depth = 130  # W dimension (spatial, after decimation)

    if model_type == "CNNAutoencoder":
        from src.models.cnn_ae import CNNAutoencoder

        model = CNNAutoencoder(
            in_channels=3,
            input_height=input_pulses,  # Pulses (temporal)
            input_width=input_depth,     # Depth (spatial)
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=use_batchnorm,
            num_classes=num_classes
        )
    elif model_type == "UNetAutoencoder":
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

    if model_type in ["CNNAutoencoder", "UNetAutoencoder"]:
        return CNNAdapter()
    else:
        raise ValueError(f"No adapter for model type: {model_type}")


def create_callbacks(config, results_dir, test_loader=None):
    """Create training callbacks."""
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

    # Visualization callback
    validation_config = getattr(config.ml.training, 'validation', None)
    plot_every = getattr(validation_config, 'plot_every_n_epochs', 10) if validation_config else 10

    vis_callback = VisualizationCallback(
        save_dir=results_dir,
        plot_every_n_epochs=plot_every,
        test_loader=test_loader
    )
    callbacks.append(vis_callback)

    # WandB callback
    if config.wandb.use_wandb:
        wandb_config = {
            'model_type': config.ml.model.type,
            'embedding_dim': config.ml.model.embedding_dim,
            'channels': config.ml.model.channels_per_layer,
            'epochs': config.ml.training.epochs,
            'learning_rate': config.ml.training.lr,
            'optimizer': config.ml.training.optimizer.type,
            'scheduler': config.ml.training.lr_scheduler.type
        }

        callbacks.append(WandBCallback(
            project=config.wandb.project,
            config=wandb_config,
            name=config.wandb.name,
            save_dir=results_dir
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
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )

    # Create model and adapter
    model = create_model(config)
    adapter = create_adapter(config)

    # Get results directory
    results_base = config.get_checkpoint_path()

    # Create callbacks
    # Note: results_dir will be set by trainer with timestamp
    callbacks = create_callbacks(config, results_base, test_loader)

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

    # Add class weights for weighted loss if enabled
    balancing_config = getattr(config.ml.training, 'class_balancing', None)
    use_weighted_loss = (
        balancing_config is not None and
        getattr(balancing_config, 'enabled', False) and
        getattr(balancing_config, 'method', '') in ['weighted_loss', 'both']
    )

    if use_weighted_loss:
        class_weights = train_ds.get_class_weights()
        loss_weights['class_weights'] = class_weights
        logger.info(f"Using weighted loss with class weights: {class_weights}")

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
            loss_weights=loss_weights,
            grad_clip_norm=regularization['grad_clip_norm'],
            restart=restart
        )

        # Final evaluation
        logger.info("Running final evaluation...")
        test_metrics = trainer.evaluate(test_loader)

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
    print(f"Test MAE:         {test_metrics['test_mae']:.6f}")

    # Print classification accuracy if available
    if 'val_accuracy' in history:
        print(f"Final Val Acc:    {history['val_accuracy'][-1]:.2%}")
    if 'test_accuracy' in test_metrics:
        print(f"Test Accuracy:    {test_metrics['test_accuracy']:.2%}")

    print(f"Total Epochs:     {len(history['train_loss'])}")
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
        action='store_true',
        help='Restart training from latest checkpoint'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config, restart=args.restart)
