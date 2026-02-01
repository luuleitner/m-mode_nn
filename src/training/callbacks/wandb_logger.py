"""
WandB Callback - Weights & Biases logging integration.

Enhanced with per-class metrics, confusion matrices, and comprehensive logging
matching the CNN classifier's logging capabilities.
"""

import os
import yaml
import numpy as np
from .base_callback import Callback

import logging
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. WandB logging disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

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


class WandBCallback(Callback):
    """
    Weights & Biases logging callback with comprehensive metrics.

    Features:
    - Per-class accuracy, precision, recall, F1
    - Confusion matrix plotting
    - Train-val gap tracking
    - Gradient norm logging
    - Reconstruction quality metrics
    """

    def __init__(self, project, config=None, name=None, save_dir=None,
                 log_every_n_batches=10, watch_model=True, api_key=None,
                 class_names=None, plot_confusion_every=5):
        self.project = project
        self.config = config or {}
        self.name = name
        self.save_dir = save_dir
        self.log_every_n_batches = log_every_n_batches
        self.watch_model = watch_model
        self.api_key = api_key
        self.class_names = class_names if class_names is not None else _DEFAULT_CLASS_NAMES
        self.plot_confusion_every = plot_confusion_every

        self.enabled = WANDB_AVAILABLE
        self._run = None
        self._batch_step = 0

    def on_train_begin(self, logs=None):
        if not self.enabled:
            return

        try:
            # Login with API key if provided
            if self.api_key:
                wandb.login(key=self.api_key)

            self._run = wandb.init(
                project=self.project,
                config=self.config,
                name=self.name,
                dir=self.save_dir,
                reinit=True
            )

            # Define separate x-axes for batch-level and epoch-level metrics
            # This prevents step conflicts between batch and epoch logging
            wandb.define_metric("batch_step")
            wandb.define_metric("epoch")
            wandb.define_metric("batch/*", step_metric="batch_step")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

            if self.watch_model and hasattr(self.trainer, 'model'):
                wandb.watch(self.trainer.model, log='gradients', log_freq=100)

            logger.info(f"WandB initialized: {self.project}")

        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.enabled = False

    def on_train_end(self, logs=None):
        if not self.enabled or self._run is None:
            return

        logs = logs or {}

        try:
            if hasattr(self.trainer, 'history'):
                history = self.trainer.history

                if history['val_loss']:
                    wandb.run.summary['best_val_loss'] = min(history['val_loss'])
                    wandb.run.summary['final_val_loss'] = history['val_loss'][-1]
                    wandb.run.summary['final_train_loss'] = history['train_loss'][-1]
                    wandb.run.summary['total_epochs'] = len(history['train_loss'])

                if history.get('val_accuracy'):
                    wandb.run.summary['best_val_accuracy'] = max(history['val_accuracy'])
                    wandb.run.summary['final_val_accuracy'] = history['val_accuracy'][-1]

                if history.get('val_balanced_accuracy'):
                    wandb.run.summary['best_val_balanced_accuracy'] = max(history['val_balanced_accuracy'])
                    wandb.run.summary['final_val_balanced_accuracy'] = history['val_balanced_accuracy'][-1]

                if history.get('val_mse'):
                    wandb.run.summary['best_val_mse'] = min(history['val_mse'])
                    wandb.run.summary['final_val_mse'] = history['val_mse'][-1]

            # Log test confusion matrix if available
            test_preds = getattr(self.trainer, '_last_test_predictions', None)
            test_labels = getattr(self.trainer, '_last_test_labels', None)
            if test_preds is not None and test_labels is not None:
                epoch = logs.get('epoch', len(self.trainer.history['train_loss']) - 1)
                self._plot_confusion_matrix(epoch, test_preds, test_labels, split='test')

                # Log test metrics to summary
                test_acc = (test_preds == test_labels).mean()
                wandb.run.summary['test_accuracy'] = test_acc

            wandb.finish()
            logger.info("WandB run finished")

        except Exception as e:
            logger.warning(f"Error finishing WandB run: {e}")

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return

        logs = logs or {}
        try:
            # Core metrics
            log_dict = {
                'epoch': epoch,
                'train/loss': logs.get('train_loss'),
                'val/loss': logs.get('val_loss'),
                'val/mse': logs.get('val_mse'),
                'train/learning_rate': logs.get('learning_rate')
            }

            # Classification metrics
            if 'train_accuracy' in logs:
                log_dict['train/accuracy'] = logs.get('train_accuracy')
            if 'val_accuracy' in logs:
                log_dict['val/accuracy'] = logs.get('val_accuracy')
            if 'val_balanced_accuracy' in logs:
                log_dict['val/balanced_accuracy'] = logs.get('val_balanced_accuracy')

            # Train-val gap (overfitting indicators)
            train_loss = logs.get('train_loss')
            val_loss = logs.get('val_loss')
            if train_loss is not None and val_loss is not None:
                log_dict['gap/loss'] = train_loss - val_loss

            train_acc = logs.get('train_accuracy')
            val_acc = logs.get('val_accuracy')
            if train_acc is not None and val_acc is not None:
                log_dict['gap/accuracy'] = train_acc - val_acc

            # Per-class accuracy from logs
            per_class_acc = logs.get('per_class_acc', {})
            for cls_idx, acc in per_class_acc.items():
                cls_name = self._get_class_name(cls_idx)
                log_dict[f'val/class_{cls_name}_acc'] = acc

            # Per-class metrics (precision, recall, F1) from trainer history
            if hasattr(self.trainer, 'history'):
                history = self.trainer.history
                # Check if per_class_acc is stored in history with detailed metrics
                if 'per_class_acc' in history and history['per_class_acc']:
                    latest_per_class = history['per_class_acc'][-1] if history['per_class_acc'] else {}
                    for cls_idx, acc in latest_per_class.items():
                        cls_name = self._get_class_name(cls_idx)
                        log_dict[f'val/class_{cls_name}_acc'] = acc

            wandb.log(log_dict)

            # Plot confusion matrices for train and val every epoch
            self._plot_all_confusion_matrices(epoch)

        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    def _plot_all_confusion_matrices(self, epoch):
        """Plot confusion matrices for train and val."""
        if not PLOTTING_AVAILABLE:
            return

        # Train confusion matrix
        train_preds = getattr(self.trainer, '_last_train_predictions', None)
        train_labels = getattr(self.trainer, '_last_train_labels', None)
        if train_preds is not None and train_labels is not None:
            self._plot_confusion_matrix(epoch, train_preds, train_labels, split='train')

        # Val confusion matrix
        val_preds = getattr(self.trainer, '_last_val_predictions', None)
        val_labels = getattr(self.trainer, '_last_val_labels', None)
        if val_preds is not None and val_labels is not None:
            self._plot_confusion_matrix(epoch, val_preds, val_labels, split='val')

    def _get_class_name(self, cls_idx):
        """Get class name from index."""
        if isinstance(cls_idx, str):
            return cls_idx
        if self.class_names and cls_idx < len(self.class_names):
            return self.class_names[cls_idx]
        return f"class_{cls_idx}"

    def _plot_confusion_matrix(self, epoch, predictions, labels, split='val'):
        """
        Plot and log confusion matrix.

        Args:
            epoch: Current epoch
            predictions: Array of predictions
            labels: Array of true labels
            split: 'train', 'val', or 'test'
        """
        if not PLOTTING_AVAILABLE:
            return

        if predictions is None or labels is None or len(predictions) == 0:
            return

        try:
            num_classes = self.trainer.num_classes if hasattr(self.trainer, 'num_classes') else _DEFAULT_NUM_CLASSES
            cm = confusion_matrix(labels, predictions, labels=range(num_classes))
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

            # Compute overall accuracy for title
            accuracy = (predictions == labels).mean()

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            class_labels = [self._get_class_name(i) for i in range(num_classes)]
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

            # Save and log
            save_path = os.path.join(self.save_dir, f'confusion_matrix_{split}_epoch_{epoch+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            wandb.log({f'{split}/confusion_matrix': wandb.Image(save_path), 'epoch': epoch})
            logger.debug(f"Logged {split} confusion matrix for epoch {epoch+1}")

        except Exception as e:
            logger.warning(f"Failed to plot {split} confusion matrix: {e}")

    def on_batch_end(self, batch, logs=None):
        if not self.enabled or batch % self.log_every_n_batches != 0:
            return

        logs = logs or {}
        try:
            # Log batch-level metrics with 'batch_step' as the x-axis
            # Keys from base_trainer.compute_loss: total_loss, mse_loss, l1_loss, embedding_reg, cls_accuracy
            log_dict = {
                'batch_step': self._batch_step,
                'batch/loss': logs.get('total_loss'),
                'batch/mse_loss': logs.get('mse_loss'),
                'batch/l1_loss': logs.get('l1_loss'),
                'batch/embedding_reg': logs.get('embedding_reg'),
            }

            # Add classification metrics if available
            if 'cls_loss' in logs:
                log_dict['batch/cls_loss'] = logs.get('cls_loss')
            if 'cls_accuracy' in logs:
                log_dict['batch/cls_accuracy'] = logs.get('cls_accuracy')

            wandb.log(log_dict)
            self._batch_step += 1
        except Exception as e:
            logger.warning(f"Failed to log batch to WandB: {e}")

    def log_image(self, name, image_path):
        if not self.enabled:
            return
        try:
            if os.path.exists(image_path):
                wandb.log({name: wandb.Image(image_path)})
        except Exception as e:
            logger.warning(f"Failed to log image to WandB: {e}")

    def save_artifact(self, path, artifact_type='model'):
        if not self.enabled:
            return
        try:
            if os.path.exists(path):
                wandb.save(path)
        except Exception as e:
            logger.warning(f"Failed to save artifact to WandB: {e}")
