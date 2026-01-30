"""
Early Stopping Callback - Stop training when validation loss stops improving.
"""

from .base_callback import Callback

import utils.logging_config as logconf
logger = logconf.get_logger("EARLY_STOP")


class EarlyStoppingCallback(Callback):
    """
    Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        monitor: Metric to monitor (default: 'val_loss')
    """

    def __init__(self, patience=20, min_delta=1e-5, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        logger.info(f"Early stopping enabled: patience={self.patience}, min_delta={self.min_delta}")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"Early stopping: '{self.monitor}' not found in logs")
            return

        if current_value < self.best_value - self.min_delta:
            # Improvement found
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch + 1}"
                )
                # Signal to stop training
                if hasattr(self, 'trainer') and hasattr(self.trainer, 'callbacks'):
                    self.trainer.callbacks.stop_training = True