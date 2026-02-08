"""
Early Stopping Callback - Stop training when a monitored metric stops improving.
"""

from .base_callback import Callback

import utils.logging_config as logconf
logger = logconf.get_logger("EARLY_STOP")

# Metrics where higher is better (maximize); everything else is minimized
_MAX_METRICS = {'accuracy', 'acc', 'balanced_accuracy', 'f1', 'auc', 'precision', 'recall'}


def _infer_mode(monitor):
    """Infer min/max mode from metric name.

    Returns 'max' for accuracy-like metrics, 'min' for loss-like metrics.
    """
    # Check if any maximize-keyword appears in the monitor name
    name_lower = monitor.lower()
    for keyword in _MAX_METRICS:
        if keyword in name_lower:
            return 'max'
    return 'min'


class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric stops improving.

    Mode is auto-detected from the metric name: accuracy/f1/auc metrics
    are maximized, loss/error metrics are minimized. Can be overridden
    with the ``mode`` parameter.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min', 'max', or 'auto' (inferred from monitor name)
    """

    def __init__(self, patience=20, min_delta=1e-5, monitor='val_loss', mode='auto'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode if mode != 'auto' else _infer_mode(monitor)
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0

    def _is_improvement(self, current, best):
        if self.mode == 'min':
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_train_begin(self, logs=None):
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        logger.info(
            f"Early stopping enabled: patience={self.patience}, "
            f"min_delta={self.min_delta}, monitor={self.monitor}, mode={self.mode}"
        )

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"Early stopping: '{self.monitor}' not found in logs")
            return

        if self._is_improvement(current_value, self.best_value):
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