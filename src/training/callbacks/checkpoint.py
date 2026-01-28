"""
Checkpoint Callback - Saves best, latest, and restart checkpoints.
"""

import os
import glob
import torch
from datetime import datetime
from .base_callback import Callback

import logging
logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Saves model checkpoints during training."""

    def __init__(self, save_dir, save_best=True, save_every_n_epochs=10,
                 save_restart_every=5, keep_n_checkpoints=3):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.save_restart_every = save_restart_every
        self.keep_n_checkpoints = keep_n_checkpoints

        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None

        os.makedirs(save_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        """Initialize best loss from history if resuming."""
        if hasattr(self.trainer, 'history') and self.trainer.history['val_loss']:
            self.best_val_loss = min(self.trainer.history['val_loss'])
            logger.info(f"Checkpoint callback: best val_loss from history = {self.best_val_loss:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoints based on epoch and validation loss."""
        logs = logs or {}
        val_loss = logs.get('val_loss', float('inf'))

        if self.save_best and val_loss < self.best_val_loss:
            self._save_best_checkpoint(epoch, val_loss)

        if (epoch + 1) % self.save_every_n_epochs == 0:
            self._save_latest_checkpoint(epoch, val_loss)

        if (epoch + 1) % self.save_restart_every == 0:
            self._save_restart_checkpoint(epoch, val_loss)

    def on_train_end(self, logs=None):
        """Save final checkpoint."""
        logs = logs or {}
        epoch = logs.get('epoch', 0)
        val_loss = logs.get('val_loss', float('inf'))

        self._save_checkpoint(
            os.path.join(self.save_dir, 'final_checkpoint.pth'),
            epoch, val_loss, 'final'
        )
        logger.info("Final checkpoint saved")

    def _save_best_checkpoint(self, epoch, val_loss):
        """Save new best model and remove previous best."""
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            try:
                os.remove(self.best_checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to remove old best checkpoint: {e}")

        self.best_val_loss = val_loss
        self.best_checkpoint_path = os.path.join(
            self.save_dir, f'best_checkpoint_epoch_{epoch:04d}_loss_{val_loss:.6f}.pth'
        )

        self._save_checkpoint(self.best_checkpoint_path, epoch, val_loss, 'best')
        logger.info(f"New best model saved: loss {val_loss:.6f} at epoch {epoch + 1}")

    def _save_latest_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.save_dir, f'latest_checkpoint_{epoch:04d}.pth')
        self._save_checkpoint(path, epoch, val_loss, 'latest')
        self._cleanup_old_checkpoints('latest_checkpoint_*.pth')

    def _save_restart_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.save_dir, f'restart_checkpoint_epoch_{epoch:03d}.pth')
        self._save_checkpoint(path, epoch, val_loss, 'restart')
        self._cleanup_old_checkpoints('restart_checkpoint_*.pth')

    def _save_checkpoint(self, path, epoch, val_loss, checkpoint_type):
        """Save checkpoint with all training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict() if self.trainer.optimizer else None,
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
            'history': self.trainer.history,
            'val_loss': val_loss,
            'checkpoint_type': checkpoint_type,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'class': self.trainer.model.__class__.__name__,
                'total_params': sum(p.numel() for p in self.trainer.model.parameters())
            }
        }
        torch.save(checkpoint, path)

    def _cleanup_old_checkpoints(self, pattern):
        """Keep only the N most recent checkpoints matching pattern."""
        checkpoints = glob.glob(os.path.join(self.save_dir, pattern))

        if len(checkpoints) > self.keep_n_checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            for old_ckpt in checkpoints[:-self.keep_n_checkpoints]:
                try:
                    os.remove(old_ckpt)
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_ckpt}: {e}")

    def get_best_checkpoint_path(self):
        return self.best_checkpoint_path

    def find_latest_checkpoint(self):
        patterns = ['restart_checkpoint_*.pth', 'latest_checkpoint_*.pth', '*.pth']
        for pattern in patterns:
            checkpoints = glob.glob(os.path.join(self.save_dir, pattern))
            if checkpoints:
                return max(checkpoints, key=os.path.getmtime)
        return None
