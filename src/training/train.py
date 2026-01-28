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
    WandBCallback
)

import utils.logging_config as logconf
logger = logconf.get_logger("TRAIN")


def create_model(config):
    """Create model based on config."""
    model_type = config.ml.model.type

    if model_type == "CNNAutoencoder":
        from src.models.cnn_ae import CNNAutoencoder

        model = CNNAutoencoder(
            in_channels=3,
            input_height=130,
            input_width=config.preprocess.tokenization.window,
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Created model: {model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def create_adapter(config):
    """Create adapter based on model type."""
    model_type = config.ml.model.type

    if model_type == "CNNAutoencoder":
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
    """Create data loaders."""
    resource_config = config.get_resource_config()

    train_loader = DataLoader(
        train_ds,
        batch_size=None,  # Pre-batched
        shuffle=True,
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
    # Load config
    config = load_config(config_path)
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

    # Update visualization callback with actual test_loader
    for cb in trainer.callbacks.callbacks:
        if isinstance(cb, VisualizationCallback):
            cb.set_test_loader(test_loader)
        if isinstance(cb, CheckpointCallback):
            cb.save_dir = trainer.results_dir

    # Get training parameters
    loss_weights = config.get_loss_weights()
    regularization = config.get_regularization_config()

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
