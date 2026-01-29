"""
Base Trainer - Core training loop with callback and adapter support.
"""

import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .adapters.base_adapter import BaseAdapter
from .callbacks.base_callback import CallbackList
from .utils.restart_manager import RestartManager

import logging
logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Generic trainer for autoencoder models.
    Uses adapters for model-specific data handling and callbacks for hooks.
    """

    def __init__(self, model, adapter, callbacks=None, device='cuda', results_dir='results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.adapter = adapter

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)

        # Timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(results_dir, f'training_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history = {
            'train_loss': [],
            'val_loss': [],           # Weighted (consistent with training)
            'val_loss_unweighted': [], # Unweighted (true reconstruction quality)
            'val_mse': [],
            'learning_rates': []
        }

        self.restart_manager = RestartManager(self.results_dir)

        logger.info(f"Trainer initialized on: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Adapter: {adapter.__class__.__name__} - {adapter.input_format}")

    def setup_optimizer(self, optimizer_type='adamw', learning_rate=1e-3, weight_decay=1e-4):
        """Setup optimizer."""
        optimizer_type = optimizer_type.lower()

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"Optimizer: {type(self.optimizer).__name__}")

    def setup_scheduler(self, scheduler_type='plateau', epochs=100, steps_per_epoch=None, **kwargs):
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise RuntimeError("Setup optimizer before scheduler")

        scheduler_type = scheduler_type.lower()

        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'onecycle' and steps_per_epoch:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                epochs=epochs,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs // 3, gamma=0.1)
        elif scheduler_type == 'none':
            self.scheduler = None
        else:
            logger.warning(f"Unknown scheduler: {scheduler_type}, using none")
            self.scheduler = None

        if self.scheduler:
            logger.info(f"Scheduler: {type(self.scheduler).__name__}")

    def _to_hard_labels(self, labels):
        """
        Convert soft labels [B, num_classes] to hard labels [B].

        Handles multiple label formats:
        - Soft labels: [B, num_classes] -> argmax to get class index
        - Hard labels: [B, 1] -> squeeze
        - Hard labels: [B] -> pass through
        """
        if labels.dim() > 1 and labels.shape[-1] > 1:
            hard_labels = labels.argmax(dim=-1)
            if hard_labels.dim() > 1:
                hard_labels = hard_labels[:, 0]
        elif labels.dim() > 1:
            hard_labels = labels.squeeze(-1)
        else:
            hard_labels = labels
        return hard_labels.long()

    def compute_loss(self, reconstruction, target, embedding, labels=None, logits=None, loss_weights=None):
        """
        Compute combined loss: reconstruction (MSE + L1) + embedding regularization + classification.

        Args:
            reconstruction: Reconstructed input
            target: Original input
            embedding: Latent embedding
            labels: Class labels (soft or hard)
            logits: Classification logits from classifier head
            loss_weights: Dict with loss component weights
        """
        if loss_weights is None:
            loss_weights = {'mse_weight': 0.8, 'l1_weight': 0.2, 'embedding_reg': 0.001}

        class_weights = loss_weights.get('class_weights', None)
        classification_weight = loss_weights.get('classification_weight', 0.0)

        # Apply class-weighted loss if labels and class_weights are provided
        if labels is not None and class_weights is not None:
            hard_labels = self._to_hard_labels(labels)

            # Compute per-sample weights based on class
            sample_weights = torch.tensor(
                [class_weights.get(int(lbl.item()), 1.0) for lbl in hard_labels],
                device=reconstruction.device, dtype=reconstruction.dtype
            )

            # Compute per-sample losses, then weight
            # MSE per sample: mean over (C, H, W) dims
            mse_per_sample = F.mse_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
            l1_per_sample = F.l1_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))

            # Weighted mean
            mse_loss = (sample_weights * mse_per_sample).mean()
            l1_loss = (sample_weights * l1_per_sample).mean()
        else:
            mse_loss = F.mse_loss(reconstruction, target)
            l1_loss = F.l1_loss(reconstruction, target)

        embedding_reg = embedding.pow(2).mean()

        total_loss = (
            loss_weights['mse_weight'] * mse_loss +
            loss_weights['l1_weight'] * l1_loss +
            loss_weights['embedding_reg'] * embedding_reg
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'embedding_reg': embedding_reg.item()
        }

        # Classification loss (if classifier head is used)
        if logits is not None and labels is not None and classification_weight > 0:
            hard_labels = self._to_hard_labels(labels)

            cls_loss = F.cross_entropy(logits, hard_labels)
            total_loss = total_loss + classification_weight * cls_loss

            loss_dict['cls_loss'] = cls_loss.item()
            loss_dict['total_loss'] = total_loss.item()

            # Track classification accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == hard_labels).float().mean().item()
                loss_dict['cls_accuracy'] = accuracy

        return total_loss, loss_dict

    def train_epoch(self, train_loader, epoch, loss_weights=None, grad_clip_norm=1.0):
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        has_classifier = False

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

        for batch_idx, batch in enumerate(pbar):
            data, labels = self.adapter.prepare_batch(batch, self.device)

            # Forward - handle both 2 and 3 output formats
            outputs = self.model(data)
            if len(outputs) == 3:
                reconstruction, embedding, logits = outputs
                has_classifier = True
            else:
                reconstruction, embedding = outputs
                logits = None

            loss, loss_dict = self.compute_loss(
                reconstruction, data, embedding, labels, logits, loss_weights
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            if 'cls_accuracy' in loss_dict:
                total_accuracy += loss_dict['cls_accuracy']

            # Update progress bar
            postfix = {'loss': f'{loss.item():.4f}'}
            if 'cls_accuracy' in loss_dict:
                postfix['acc'] = f'{loss_dict["cls_accuracy"]:.2%}'
            pbar.set_postfix(postfix)

            self.callbacks.on_batch_end(batch_idx, loss_dict)

        avg_loss = total_loss / len(train_loader)
        if has_classifier:
            avg_accuracy = total_accuracy / len(train_loader)
            return avg_loss, avg_accuracy
        return avg_loss

    def validate_epoch(self, val_loader, epoch, loss_weights=None):
        """
        Validate for one epoch.

        Returns:
            dict with keys:
                - weighted_loss: Loss with class weighting (consistent with training)
                - mse: Pure MSE reconstruction error
                - unweighted_loss: Loss without class weighting (true reconstruction quality)
                - cls_accuracy: Classification accuracy (if classifier enabled)
                - cls_loss: Classification loss (if classifier enabled)
        """
        self.model.eval()
        total_weighted_loss = 0
        total_unweighted_loss = 0
        total_mse = 0
        total_cls_accuracy = 0
        total_cls_loss = 0
        has_classifier = False

        with torch.no_grad():
            for batch in val_loader:
                data, labels = self.adapter.prepare_batch(batch, self.device)

                # Forward - handle both 2 and 3 output formats
                outputs = self.model(data)
                if len(outputs) == 3:
                    reconstruction, embedding, logits = outputs
                    has_classifier = True
                else:
                    reconstruction, embedding = outputs
                    logits = None

                # Weighted loss (consistent with training objective)
                _, weighted_dict = self.compute_loss(
                    reconstruction, data, embedding, labels, logits, loss_weights
                )

                # Unweighted loss (true reconstruction quality across all classes)
                _, unweighted_dict = self.compute_loss(
                    reconstruction, data, embedding
                )

                mse = F.mse_loss(reconstruction, data)

                total_weighted_loss += weighted_dict['total_loss']
                total_unweighted_loss += unweighted_dict['total_loss']
                total_mse += mse.item()

                if 'cls_accuracy' in weighted_dict:
                    total_cls_accuracy += weighted_dict['cls_accuracy']
                    total_cls_loss += weighted_dict.get('cls_loss', 0)

        n_batches = len(val_loader)
        result = {
            'weighted_loss': total_weighted_loss / n_batches,
            'mse': total_mse / n_batches,
            'unweighted_loss': total_unweighted_loss / n_batches
        }

        if has_classifier:
            result['cls_accuracy'] = total_cls_accuracy / n_batches
            result['cls_loss'] = total_cls_loss / n_batches

        return result

    def fit(self, train_loader, val_loader, epochs=100, learning_rate=1e-3, weight_decay=1e-4,
            optimizer_type='adamw', scheduler_type='plateau', loss_weights=None,
            grad_clip_norm=1.0, restart=False):
        """Main training loop. Returns training history dict."""
        # Setup
        self.setup_optimizer(optimizer_type, learning_rate, weight_decay)
        self.setup_scheduler(scheduler_type, epochs, len(train_loader))

        # Restart if requested
        start_epoch = 0
        if restart:
            checkpoint_path = self.restart_manager.find_latest_checkpoint()
            if checkpoint_path:
                start_epoch, self.history = self.restart_manager.load_checkpoint(
                    checkpoint_path, self.model, self.optimizer, self.scheduler, self.device
                )
                logger.info(f"Restarting from epoch {start_epoch}")

        # Log input info
        sample_batch = next(iter(train_loader))
        sample_data, _ = self.adapter.prepare_batch(sample_batch, self.device)
        logger.info(f"Input info: {self.adapter.get_input_info(sample_data)}")

        self.callbacks.on_train_begin({'epochs': epochs, 'start_epoch': start_epoch})

        # Training loop
        for epoch in range(start_epoch, epochs):
            self.callbacks.on_epoch_begin(epoch)

            # Train epoch - may return (loss,) or (loss, accuracy)
            train_result = self.train_epoch(train_loader, epoch, loss_weights, grad_clip_norm)
            if isinstance(train_result, tuple):
                avg_train_loss, train_accuracy = train_result
            else:
                avg_train_loss = train_result
                train_accuracy = None

            # Validate epoch - returns dict
            val_metrics = self.validate_epoch(val_loader, epoch, loss_weights)
            avg_val_loss = val_metrics['weighted_loss']
            avg_val_mse = val_metrics['mse']
            avg_val_loss_unweighted = val_metrics['unweighted_loss']
            val_accuracy = val_metrics.get('cls_accuracy')

            # Update scheduler (use weighted val loss for consistency with training)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                elif not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_loss_unweighted'].append(avg_val_loss_unweighted)
            self.history['val_mse'].append(avg_val_mse)
            self.history['learning_rates'].append(current_lr)

            # Track classification metrics if available
            if train_accuracy is not None:
                if 'train_accuracy' not in self.history:
                    self.history['train_accuracy'] = []
                    self.history['val_accuracy'] = []
                self.history['train_accuracy'].append(train_accuracy)
                self.history['val_accuracy'].append(val_accuracy)

            callback_logs = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_loss_unweighted': avg_val_loss_unweighted,
                'val_mse': avg_val_mse,
                'learning_rate': current_lr,
                'epoch': epoch
            }
            if train_accuracy is not None:
                callback_logs['train_accuracy'] = train_accuracy
                callback_logs['val_accuracy'] = val_accuracy

            self.callbacks.on_epoch_end(epoch, callback_logs)

            # Log progress
            log_msg = (
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} (unw={avg_val_loss_unweighted:.4f}), "
                f"MSE={avg_val_mse:.4f}, LR={current_lr:.2e}"
            )
            if train_accuracy is not None:
                log_msg += f", Acc={val_accuracy:.2%}"
            logger.info(log_msg)

            # Check for early stopping
            if self.callbacks.stop_training:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.callbacks.on_train_end({
            'epoch': epochs - 1,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
        })

        logger.info("Training completed!")
        return self.history

    def evaluate(self, test_loader):
        """Evaluate on test set. Returns metrics dict."""
        self.model.eval()
        total_mse = total_mae = total_samples = 0
        total_correct = 0
        has_classifier = False

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                data, labels = self.adapter.prepare_batch(batch, self.device)

                # Handle both 2 and 3 output formats
                outputs = self.model(data)
                if len(outputs) == 3:
                    reconstruction, embedding, logits = outputs
                    has_classifier = True

                    # Compute accuracy
                    if labels is not None:
                        hard_labels = self._to_hard_labels(labels)
                        preds = logits.argmax(dim=-1)
                        total_correct += (preds == hard_labels).sum().item()
                else:
                    reconstruction, embedding = outputs

                mse = F.mse_loss(reconstruction, data, reduction='sum')
                mae = F.l1_loss(reconstruction, data, reduction='sum')

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += data.size(0)

        metrics = {
            'test_mse': total_mse / total_samples,
            'test_mae': total_mae / total_samples,
            'test_samples': total_samples
        }

        if has_classifier:
            metrics['test_accuracy'] = total_correct / total_samples

        log_msg = f"Test Results - MSE: {metrics['test_mse']:.6f}, MAE: {metrics['test_mae']:.6f}"
        if has_classifier:
            log_msg += f", Accuracy: {metrics['test_accuracy']:.2%}"
        logger.info(log_msg)

        return metrics

    def get_results_path(self):
        return self.results_dir
