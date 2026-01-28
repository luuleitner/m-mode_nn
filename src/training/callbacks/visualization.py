"""
Visualization Callback - Training curves and reconstruction visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from .base_callback import Callback

import logging
logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Generates visualizations during training."""

    def __init__(self, save_dir, plot_every_n_epochs=10, test_loader=None):
        self.save_dir = save_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        self.test_loader = test_loader
        os.makedirs(save_dir, exist_ok=True)

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_every_n_epochs == 0:
            self._plot_training_curves(epoch)
            if self.test_loader is not None:
                self._plot_reconstructions(epoch)

    def on_train_end(self, logs=None):
        epoch = logs.get('epoch', len(self.trainer.history['train_loss']) - 1)
        self._plot_training_curves(epoch, prefix='final_')
        if self.test_loader is not None:
            self._plot_reconstructions(epoch, prefix='final_')
            self._plot_embeddings(epoch, prefix='final_')

    def _plot_training_curves(self, epoch, prefix=''):
        """Plot training and validation loss curves."""
        history = self.trainer.history

        if not history['train_loss']:
            logger.warning("No training history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs_range = range(1, len(history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE curve
        axes[0, 1].plot(epochs_range, history['val_mse'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Validation MSE')
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        if history['learning_rates']:
            axes[1, 0].plot(epochs_range, history['learning_rates'], 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        # Overfitting monitor
        if len(history['train_loss']) > 1:
            loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
            axes[1, 1].plot(epochs_range, loss_diff, 'orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss - Train Loss')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{prefix}training_curves_epoch_{epoch+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {save_path}")

    def _plot_reconstructions(self, epoch, prefix=''):
        """Plot reconstruction comparisons using adapter."""
        if self.test_loader is None:
            return

        self.trainer.model.eval()
        with torch.no_grad():
            batch = next(iter(self.test_loader))
            data = self.trainer.adapter.prepare_batch(batch, self.trainer.device)
            reconstruction, _ = self.trainer.model(data)

        save_path = os.path.join(self.save_dir, f'{prefix}reconstructions_epoch_{epoch+1}.png')
        self.trainer.adapter.visualize_reconstruction(data, reconstruction, save_path)

        sample_path = os.path.join(self.save_dir, f'{prefix}samples_epoch_{epoch+1}.png')
        self.trainer.adapter.visualize_samples(data, reconstruction, sample_path)
        logger.info(f"Reconstruction plots saved: {save_path}")

    def _plot_embeddings(self, epoch, prefix='', max_batches=10):
        """Plot embedding distribution analysis."""
        if self.test_loader is None:
            return

        self.trainer.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= max_batches:
                    break
                data = self.trainer.adapter.prepare_batch(batch, self.trainer.device)
                _, embedding = self.trainer.model(data)
                all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            return

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Embedding norm distribution
        embedding_norms = np.linalg.norm(all_embeddings, axis=1)
        axes[0].hist(embedding_norms, bins=30, density=True, alpha=0.7, color='blue')
        axes[0].set_xlabel('Embedding L2 Norm')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Embedding Norm Distribution')
        axes[0].grid(True, alpha=0.3)

        # Embedding value distribution
        axes[1].hist(all_embeddings.flatten(), bins=50, density=True, alpha=0.7, color='green')
        axes[1].set_xlabel('Embedding Value')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Embedding Value Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{prefix}embeddings_epoch_{epoch+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Embedding analysis saved: {save_path}")
