"""
Visualization Callback - Training curves, reconstruction, and confusion matrix visualizations.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from .base_callback import Callback

import logging
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Load class config from centralized label config
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_label_config_path = os.path.join(_project_root, 'preprocessing/label_logic/label_config.yaml')
try:
    with open(_label_config_path) as _f:
        _label_config = yaml.safe_load(_f)
    _classes_config = _label_config.get('classes', {})
    _INCLUDE_NOISE = _classes_config.get('include_noise', True)
    _DEFAULT_NUM_CLASSES = 5 if _INCLUDE_NOISE else 4
    # When noise excluded, labels are remapped 1,2,3,4 → 0,1,2,3
    if _INCLUDE_NOISE:
        _DEFAULT_CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(5)]
    else:
        _DEFAULT_CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(1, 5)]
except Exception:
    _DEFAULT_NUM_CLASSES = 4
    _DEFAULT_CLASS_NAMES = ['Up', 'Down', 'Left', 'Right']


class VisualizationCallback(Callback):
    """Generates visualizations during training."""

    def __init__(self, save_dir, plot_every_n_epochs=1, test_loader=None, log_to_wandb=True, class_names=None):
        """
        Args:
            save_dir: Directory to save plots
            plot_every_n_epochs: Generate plots every N epochs (default: 1 = every epoch)
            test_loader: Optional test data loader for reconstruction plots
            log_to_wandb: Log plots to WandB if available
            class_names: List of class names for confusion matrix labels (default: from label_config)
        """
        self.save_dir = save_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        self.test_loader = test_loader
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.class_names = class_names if class_names is not None else _DEFAULT_CLASS_NAMES
        os.makedirs(save_dir, exist_ok=True)

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    def _log_to_wandb(self, name, image_path, epoch=None):
        """Log image to wandb if available and enabled."""
        if not self.log_to_wandb or not WANDB_AVAILABLE:
            return
        try:
            if wandb.run is not None and os.path.exists(image_path):
                log_data = {name: wandb.Image(image_path)}
                if epoch is not None:
                    log_data['epoch'] = epoch
                wandb.log(log_data)
        except Exception as e:
            logger.warning(f"Failed to log image to WandB: {e}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_every_n_epochs == 0:
            self._plot_training_curves(epoch)
            if self.test_loader is not None:
                self._plot_reconstructions(epoch)
                self._plot_embeddings(epoch)
            # Plot confusion matrices for train and val
            self._plot_confusion_matrices(epoch)

    def on_train_end(self, logs=None):
        logs = logs or {}
        epoch = logs.get('epoch', len(self.trainer.history['train_loss']) - 1)
        self._plot_training_curves(epoch, prefix='final_')
        if self.test_loader is not None:
            self._plot_reconstructions(epoch, prefix='final_')
            self._plot_embeddings(epoch, prefix='final_')

        # Plot final confusion matrices
        self._plot_confusion_matrices(epoch, prefix='final_')

        # Plot test confusion matrix
        test_preds = getattr(self.trainer, '_last_test_predictions', None)
        test_labels = getattr(self.trainer, '_last_test_labels', None)
        if test_preds is not None and test_labels is not None:
            self._plot_single_confusion_matrix(epoch, test_preds, test_labels, split='test', prefix='final_')

    def _plot_training_curves(self, epoch, prefix=''):
        """Plot training and validation loss/accuracy curves."""
        history = self.trainer.history

        if not history['train_loss']:
            logger.warning("No training history to plot")
            return

        # Determine if we have classification metrics
        has_accuracy = history.get('train_accuracy') and history.get('val_accuracy')

        if has_accuracy:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        epochs_range = range(1, len(history['train_loss']) + 1)

        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MSE curve
        ax = axes[0, 1]
        if history['val_mse']:
            ax.plot(epochs_range, history['val_mse'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Validation MSE (Reconstruction)')
        ax.grid(True, alpha=0.3)

        if has_accuracy:
            # Accuracy curves
            ax = axes[0, 2]
            ax.plot(epochs_range, history['train_accuracy'], 'b-', label='Train', linewidth=2)
            ax.plot(epochs_range, history['val_accuracy'], 'r-', label='Val', linewidth=2)
            if history.get('val_balanced_accuracy'):
                ax.plot(epochs_range, history['val_balanced_accuracy'], 'r--',
                       label='Val Balanced', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 0]
        if history['learning_rates']:
            ax.plot(epochs_range, history['learning_rates'], 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        # Overfitting monitor (loss gap)
        ax = axes[1, 1]
        if len(history['train_loss']) > 1:
            loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
            ax.plot(epochs_range, loss_diff, 'orange', linewidth=2, label='Loss Gap')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val - Train')
            ax.set_title('Overfitting Monitor')
            ax.grid(True, alpha=0.3)

        if has_accuracy:
            # Per-class accuracy over time
            ax = axes[1, 2]
            if history.get('per_class_acc') and history['per_class_acc']:
                # Get class indices from first entry
                first_entry = history['per_class_acc'][0]
                if first_entry:
                    for cls_idx in sorted(first_entry.keys()):
                        accs = [ep.get(cls_idx, 0) for ep in history['per_class_acc']]
                        ax.plot(epochs_range[:len(accs)], accs, label=f'Class {cls_idx}', linewidth=1.5)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Per-Class Val Accuracy')
                    ax.set_ylim(0, 1)
                    ax.legend(loc='lower right')
                    ax.grid(True, alpha=0.3)

        plt.suptitle(f'Training Progress - Epoch {epoch + 1}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{prefix}training_curves_epoch_{epoch+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {save_path}")
        self._log_to_wandb('training/curves', save_path, epoch)

    def _plot_reconstructions(self, epoch, prefix=''):
        """Plot reconstruction comparisons: GT / Prediction / Diff."""
        if self.test_loader is None:
            return

        self.trainer.model.eval()
        with torch.no_grad():
            batch = next(iter(self.test_loader))
            data, labels = self.trainer.adapter.prepare_batch(batch, self.trainer.device)
            outputs = self.trainer.model(data)
            reconstruction = outputs[0]

        # Use adapter's visualization if available
        save_path = os.path.join(self.save_dir, f'{prefix}reconstructions_epoch_{epoch+1}.png')
        if hasattr(self.trainer.adapter, 'visualize_reconstruction'):
            self.trainer.adapter.visualize_reconstruction(data, reconstruction, save_path)
            self._log_to_wandb('reconstructions/heatmaps', save_path, epoch)

        # GT / Prediction / Diff visualization
        diff_path = os.path.join(self.save_dir, f'{prefix}gt_pred_diff_epoch_{epoch+1}.png')
        self._plot_gt_pred_diff(data, reconstruction, labels, diff_path, epoch)

        # Sample visualization
        sample_path = os.path.join(self.save_dir, f'{prefix}samples_epoch_{epoch+1}.png')
        if hasattr(self.trainer.adapter, 'visualize_samples'):
            self.trainer.adapter.visualize_samples(data, reconstruction, sample_path)
            self._log_to_wandb('reconstructions/samples', sample_path, epoch)

        logger.info(f"Reconstruction plots saved to {self.save_dir}")

    def _plot_gt_pred_diff(self, data, reconstruction, labels, save_path, epoch, n_samples=4):
        """
        Plot Ground Truth / Prediction / Difference for multiple samples.

        Shows reconstruction quality per sample with difference highlighting errors.
        """
        # Move to CPU and convert to numpy
        data_np = data.cpu().numpy()
        recon_np = reconstruction.cpu().numpy()

        # Get labels for display
        if labels is not None:
            labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            if len(labels_np.shape) > 1 and labels_np.shape[-1] > 1:
                labels_np = labels_np.argmax(axis=-1)
        else:
            labels_np = None

        n_samples = min(n_samples, data_np.shape[0])

        # Create figure: n_samples rows x 3 columns (GT, Pred, Diff)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Use first channel for visualization (or mean of channels)
            if data_np.shape[1] == 1:
                gt = data_np[i, 0]
                pred = recon_np[i, 0]
            else:
                # Use mean across channels or first channel
                gt = data_np[i, 0]  # First channel (typically RF data)
                pred = recon_np[i, 0]

            diff = gt - pred

            # Compute metrics for this sample
            mse = np.mean((gt - pred) ** 2)
            mae = np.mean(np.abs(gt - pred))

            # Determine colormap limits
            vmin, vmax = gt.min(), gt.max()
            diff_abs_max = max(abs(diff.min()), abs(diff.max()))

            # Ground Truth
            im0 = axes[i, 0].imshow(gt, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            label_str = f" (class {int(labels_np[i])})" if labels_np is not None else ""
            axes[i, 0].set_title(f'Ground Truth{label_str}')
            axes[i, 0].set_ylabel(f'Sample {i}')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

            # Prediction
            im1 = axes[i, 1].imshow(pred, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'Prediction')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            # Difference (with symmetric colormap)
            im2 = axes[i, 2].imshow(diff, aspect='auto', cmap='RdBu_r',
                                    vmin=-diff_abs_max, vmax=diff_abs_max)
            axes[i, 2].set_title(f'Difference (MSE={mse:.4f}, MAE={mae:.4f})')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        plt.suptitle(f'Reconstruction Quality - Epoch {epoch + 1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self._log_to_wandb('reconstructions/gt_pred_diff', save_path, epoch)

    def _plot_confusion_matrices(self, epoch, prefix=''):
        """Plot confusion matrices for train and val."""
        if not SKLEARN_AVAILABLE:
            return

        # Train confusion matrix
        train_preds = getattr(self.trainer, '_last_train_predictions', None)
        train_labels = getattr(self.trainer, '_last_train_labels', None)
        if train_preds is not None and train_labels is not None:
            self._plot_single_confusion_matrix(
                epoch, train_preds, train_labels, split='train', prefix=prefix
            )

        # Val confusion matrix
        val_preds = getattr(self.trainer, '_last_val_predictions', None)
        val_labels = getattr(self.trainer, '_last_val_labels', None)
        if val_preds is not None and val_labels is not None:
            self._plot_single_confusion_matrix(
                epoch, val_preds, val_labels, split='val', prefix=prefix
            )

    def _plot_single_confusion_matrix(self, epoch, predictions, labels, split='val', prefix=''):
        """Plot a single confusion matrix."""
        if predictions is None or labels is None or len(predictions) == 0:
            return

        try:
            num_classes = self.trainer.num_classes if hasattr(self.trainer, 'num_classes') else _DEFAULT_NUM_CLASSES
            if num_classes is None or num_classes == 0:
                return

            cm = confusion_matrix(labels, predictions, labels=range(num_classes))
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

            # Compute accuracy
            accuracy = (predictions == labels).mean()

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            # Use class names for labels
            class_labels = [self.class_names[i] if i < len(self.class_names) else f'Class {i}'
                           for i in range(num_classes)]
            ax.set(
                xticks=np.arange(num_classes),
                yticks=np.arange(num_classes),
                xticklabels=class_labels,
                yticklabels=class_labels,
                ylabel='True Label',
                xlabel='Predicted Label',
                title=f'{split.capitalize()} Confusion Matrix - Epoch {epoch+1} (Acc: {accuracy:.1%})'
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Add text annotations
            thresh = cm_norm.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    text = f'{cm_norm[i, j]:.1%}\n({cm[i, j]})'
                    ax.text(j, i, text, ha='center', va='center',
                            color='white' if cm_norm[i, j] > thresh else 'black', fontsize=9)

            plt.tight_layout()

            save_path = os.path.join(self.save_dir, f'{prefix}confusion_matrix_{split}_epoch_{epoch+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            self._log_to_wandb(f'{split}/confusion_matrix', save_path, epoch)
            logger.info(f"{split.capitalize()} confusion matrix saved: {save_path}")

        except Exception as e:
            logger.warning(f"Failed to plot {split} confusion matrix: {e}")

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
                data, _ = self.trainer.adapter.prepare_batch(batch, self.trainer.device)
                outputs = self.trainer.model(data)
                embedding = outputs[1]  # Handle both 2 and 3 output formats
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
        self._log_to_wandb('embeddings/distribution', save_path, epoch)
