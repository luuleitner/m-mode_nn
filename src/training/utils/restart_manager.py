"""
Restart Manager - Simplified checkpoint loading for training resumption.
"""

import os
import glob
import torch

import logging
logger = logging.getLogger(__name__)


class RestartManager:
    """Handles loading checkpoints for training restart."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def find_latest_checkpoint(self):
        """Find most recent checkpoint. Priority: restart > latest > best > any."""
        if not os.path.exists(self.checkpoint_dir):
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return None

        patterns = [
            'restart_checkpoint_*.pth',
            'latest_checkpoint_*.pth',
            'best_checkpoint_*.pth',
            '*.pth'
        ]

        for pattern in patterns:
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
            if checkpoints:
                latest = max(checkpoints, key=os.path.getmtime)
                logger.info(f"Found checkpoint: {latest}")
                return latest

        logger.info(f"No checkpoints found in: {self.checkpoint_dir}")
        return None

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, device=None, strict=True):
        """Load checkpoint and restore training state. Returns (start_epoch, history).

        Use strict=False for transfer learning where model architecture may differ.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if missing:
            logger.warning(f"Missing keys (randomly initialized): {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {unexpected}")
        logger.info("Model state loaded")

        # Load optimizer state
        if optimizer and checkpoint.get('optimizer_state_dict'):
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        epoch = checkpoint.get('epoch', 0)
        history = checkpoint.get('history', {
            'train_loss': [], 'val_loss': [], 'val_mse': [], 'learning_rates': []
        })

        logger.info(f"Checkpoint loaded from epoch {epoch}")
        if checkpoint.get('val_loss'):
            logger.info(f"Validation loss at checkpoint: {checkpoint['val_loss']:.6f}")

        return epoch + 1, history

    def can_restart(self):
        return self.find_latest_checkpoint() is not None
