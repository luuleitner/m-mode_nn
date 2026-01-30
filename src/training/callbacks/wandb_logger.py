"""
WandB Callback - Weights & Biases logging integration.
"""

import os
from .base_callback import Callback

import logging
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. WandB logging disabled.")


class WandBCallback(Callback):
    """Weights & Biases logging callback."""

    def __init__(self, project, config=None, name=None, save_dir=None,
                 log_every_n_batches=1, watch_model=True, api_key=None):
        self.project = project
        self.config = config or {}
        self.name = name
        self.save_dir = save_dir
        self.log_every_n_batches = log_every_n_batches
        self.watch_model = watch_model
        self.api_key = api_key

        self.enabled = WANDB_AVAILABLE
        self._run = None
        self._batch_step = 0  # Global batch counter across all epochs

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

        try:
            if hasattr(self.trainer, 'history'):
                history = self.trainer.history
                if history['val_loss']:
                    wandb.run.summary['best_val_loss'] = min(history['val_loss'])
                    wandb.run.summary['final_val_loss'] = history['val_loss'][-1]
                    wandb.run.summary['final_train_loss'] = history['train_loss'][-1]
                    wandb.run.summary['total_epochs'] = len(history['train_loss'])

            wandb.finish()
            logger.info("WandB run finished")

        except Exception as e:
            logger.warning(f"Error finishing WandB run: {e}")

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return

        logs = logs or {}
        try:
            # Log epoch-level metrics with 'epoch' as the x-axis
            log_dict = {
                'epoch': epoch,
                'train/loss': logs.get('train_loss'),
                'val/loss': logs.get('val_loss'),
                'val/mse': logs.get('val_mse'),
                'train/learning_rate': logs.get('learning_rate')
            }

            # Add classification metrics if available
            if 'train_accuracy' in logs:
                log_dict['train/accuracy'] = logs.get('train_accuracy')
            if 'val_accuracy' in logs:
                log_dict['val/accuracy'] = logs.get('val_accuracy')

            wandb.log(log_dict)
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

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
