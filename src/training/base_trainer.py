"""
Base Trainer - Core training loop with callback and adapter support.

Refactored to use unified loss functions from losses.py:
- Reconstruction loss is NEVER class-weighted
- Classification loss CAN be class-weighted (training only)
- Validation uses unweighted loss for fair comparison
"""

import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from .adapters.base_adapter import BaseAdapter
from .callbacks.base_callback import CallbackList
from .utils.restart_manager import RestartManager
from .utils.losses import compute_joint_train_loss, compute_joint_eval_loss

import logging
logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Generic trainer for autoencoder models.
    Uses adapters for model-specific data handling and callbacks for hooks.
    """

    def __init__(self, model, adapter, callbacks=None, device='cuda', results_dir='results',
                 create_subdir=True):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            adapter: Data adapter for batch preparation
            callbacks: List of callbacks
            device: Device to use ('cuda' or 'cpu')
            results_dir: Base directory for results
            create_subdir: If True, create timestamped subdirectory. Set False for CV folds.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.adapter = adapter

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)

        # Results directory - optionally create timestamped subdirectory
        if create_subdir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = os.path.join(results_dir, f'training_{timestamp}')
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.class_weights = None  # Pre-computed tensor for efficiency
        self.num_classes = None
        self.history = {
            'train_loss': [],
            'val_loss': [],           # Unweighted (for fair comparison)
            'val_mse': [],
            'learning_rates': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_balanced_accuracy': [],
            'per_class_acc': []
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

    def _prepare_labels(self, labels):
        """
        Prepare labels for loss computation.

        Args:
            labels: Raw labels from dataset (soft or hard)

        Returns:
            tuple: (soft_labels, hard_labels) where soft_labels may be None
        """
        if labels is None:
            return None, None

        labels = labels.to(self.device)

        if labels.dim() > 1 and labels.shape[-1] > 1:
            # Soft labels: (B, num_classes)
            soft_labels = labels.float()
            hard_labels = labels.argmax(dim=-1).long()
        else:
            # Hard labels
            hard_labels = labels.squeeze() if labels.dim() > 1 else labels
            hard_labels = hard_labels.long()
            soft_labels = None

        return soft_labels, hard_labels

    def set_class_weights(self, class_weights_dict, num_classes):
        """
        Set class weights from dict to tensor for efficient GPU computation.

        Args:
            class_weights_dict: Dict mapping class_idx -> weight
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        if class_weights_dict is not None:
            weights = torch.tensor(
                [class_weights_dict.get(i, 1.0) for i in range(num_classes)],
                dtype=torch.float32
            )
            self.class_weights = weights.to(self.device)
            logger.info(f"Class weights set: {weights.tolist()}")
        else:
            self.class_weights = None

    def compute_loss(self, reconstruction, target, embedding, labels=None, logits=None,
                     loss_weights=None, is_training=True):
        """
        Compute joint loss using unified loss functions.

        Key design:
        - Reconstruction loss is NEVER class-weighted
        - Classification loss is weighted during training only

        Args:
            reconstruction: Reconstructed input
            target: Original input
            embedding: Latent embedding (for regularization)
            labels: Class labels (soft or hard)
            logits: Classification logits from classifier head
            loss_weights: Dict with mse_weight, l1_weight, cls_weight, embedding_reg
            is_training: If True, apply class weights to classification

        Returns:
            tuple: (total_loss, loss_dict)
        """
        if loss_weights is None:
            loss_weights = {'mse_weight': 0.5, 'l1_weight': 0.5, 'cls_weight': 0.0, 'embedding_reg': 0.001}

        soft_labels, hard_labels = self._prepare_labels(labels)

        # Use unified joint loss (reconstruction never weighted, classification weighted for training)
        cls_weight = loss_weights.get('cls_weight', loss_weights.get('classification_weight', 0.0))
        joint_weights = {
            'mse_weight': loss_weights.get('mse_weight', 0.5),
            'l1_weight': loss_weights.get('l1_weight', 0.5),
            'cls_weight': cls_weight
        }

        if is_training:
            total_loss, loss_dict = compute_joint_train_loss(
                reconstruction, target, logits, soft_labels, hard_labels,
                self.class_weights, joint_weights
            )
        else:
            total_loss, loss_dict = compute_joint_eval_loss(
                reconstruction, target, logits, soft_labels, hard_labels,
                joint_weights
            )

        # Add embedding regularization
        embedding_reg_weight = loss_weights.get('embedding_reg', 0.001)
        if embedding_reg_weight > 0:
            embedding_reg = embedding.pow(2).mean()
            total_loss = total_loss + embedding_reg_weight * embedding_reg
            loss_dict['embedding_reg'] = embedding_reg.item()
            loss_dict['total'] = total_loss.item()

        # Rename for backward compatibility
        loss_dict['total_loss'] = loss_dict.get('total', total_loss.item())
        loss_dict['mse_loss'] = loss_dict.get('mse', 0.0)
        loss_dict['l1_loss'] = loss_dict.get('l1', 0.0)
        if 'accuracy' in loss_dict:
            loss_dict['cls_accuracy'] = loss_dict['accuracy']

        return total_loss, loss_dict

    def train_epoch(self, train_loader, epoch, loss_weights=None, grad_clip_norm=1.0):
        """Train for one epoch. Returns (avg_loss, avg_accuracy, per_class_acc)."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        has_classifier = False
        all_preds = []
        all_labels = []

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
                reconstruction, data, embedding, labels, logits, loss_weights, is_training=True
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Log gradient norm before clipping
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
            else:
                grad_norm = 0.0

            self.optimizer.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if 'cls_accuracy' in loss_dict and logits is not None:
                soft_labels, hard_labels = self._prepare_labels(labels)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == hard_labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(hard_labels.cpu().numpy())

            # Update progress bar
            postfix = {'loss': f'{loss.item():.4f}'}
            if 'cls_accuracy' in loss_dict:
                postfix['acc'] = f'{loss_dict["cls_accuracy"]:.2%}'
            pbar.set_postfix(postfix)

            self.callbacks.on_batch_end(batch_idx, loss_dict)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if has_classifier and total_samples > 0:
            avg_accuracy = total_correct / total_samples
            # Compute per-class accuracy
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            per_class_acc = {}
            if self.num_classes:
                for cls in range(self.num_classes):
                    mask = all_labels == cls
                    if mask.sum() > 0:
                        per_class_acc[cls] = float((all_preds[mask] == cls).mean())
                    else:
                        per_class_acc[cls] = 0.0

            # Store for confusion matrix plotting
            self._last_train_predictions = all_preds
            self._last_train_labels = all_labels

            return avg_loss, avg_accuracy, per_class_acc

        return avg_loss, None, {}

    def validate_epoch(self, val_loader, epoch, loss_weights=None):
        """
        Validate for one epoch. Uses UNWEIGHTED loss for fair comparison.

        Returns:
            dict with comprehensive metrics including per-class breakdown
        """
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_samples = 0
        total_correct = 0
        has_classifier = False
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                data, labels = self.adapter.prepare_batch(batch, self.device)
                batch_size = data.size(0)

                # Forward
                outputs = self.model(data)
                if len(outputs) == 3:
                    reconstruction, embedding, logits = outputs
                    has_classifier = True
                else:
                    reconstruction, embedding = outputs
                    logits = None

                # Unweighted loss for fair evaluation
                loss, loss_dict = self.compute_loss(
                    reconstruction, data, embedding, labels, logits, loss_weights, is_training=False
                )

                mse = F.mse_loss(reconstruction, data)

                total_loss += loss.item() * batch_size
                total_mse += mse.item() * batch_size
                total_samples += batch_size

                if logits is not None and labels is not None:
                    soft_labels, hard_labels = self._prepare_labels(labels)
                    preds = logits.argmax(dim=-1)
                    probs = F.softmax(logits, dim=-1)

                    total_correct += (preds == hard_labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(hard_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

        result = {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'mse': total_mse / total_samples if total_samples > 0 else 0.0,
        }

        if has_classifier and all_preds:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            result['accuracy'] = total_correct / total_samples
            result['cls_accuracy'] = result['accuracy']  # Backward compat

            # Per-class accuracy
            per_class_acc = {}
            if self.num_classes:
                for cls in range(self.num_classes):
                    mask = all_labels == cls
                    if mask.sum() > 0:
                        per_class_acc[cls] = float((all_preds[mask] == cls).mean())
                    else:
                        per_class_acc[cls] = 0.0
                result['per_class_acc'] = per_class_acc

                # Balanced accuracy (mean of per-class recalls)
                result['balanced_accuracy'] = np.mean(list(per_class_acc.values()))

                # Per-class precision, recall, F1
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_labels, all_preds,
                    labels=list(range(self.num_classes)),
                    zero_division=0
                )
                result['per_class_metrics'] = {
                    cls: {'precision': precision[cls], 'recall': recall[cls],
                          'f1': f1[cls], 'support': support[cls]}
                    for cls in range(self.num_classes)
                }

                # Mean confidence per class
                mean_confidence = {}
                for cls in range(self.num_classes):
                    mask = all_preds == cls
                    if mask.sum() > 0:
                        mean_confidence[cls] = float(all_probs[mask, cls].mean())
                    else:
                        mean_confidence[cls] = 0.0
                result['mean_confidence'] = mean_confidence

            result['predictions'] = all_preds
            result['labels'] = all_labels
            result['probabilities'] = all_probs

            # Store for WandB callback confusion matrix
            self._last_val_predictions = all_preds
            self._last_val_labels = all_labels

        # Backward compatibility keys
        result['weighted_loss'] = result['loss']
        result['unweighted_loss'] = result['loss']

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

            # Train epoch - returns (loss, accuracy, per_class_acc)
            avg_train_loss, train_accuracy, train_per_class = self.train_epoch(
                train_loader, epoch, loss_weights, grad_clip_norm
            )

            # Validate epoch - returns dict with comprehensive metrics
            val_metrics = self.validate_epoch(val_loader, epoch, loss_weights)
            avg_val_loss = val_metrics['loss']
            avg_val_mse = val_metrics['mse']
            val_accuracy = val_metrics.get('accuracy')
            val_balanced_acc = val_metrics.get('balanced_accuracy', 0.0)
            val_per_class = val_metrics.get('per_class_acc', {})

            # Update scheduler (use val loss)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                elif not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_mse'].append(avg_val_mse)
            self.history['learning_rates'].append(current_lr)

            # Track classification metrics if available
            if train_accuracy is not None:
                self.history['train_accuracy'].append(train_accuracy)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['val_balanced_accuracy'].append(val_balanced_acc)
                self.history['per_class_acc'].append(val_per_class)

            callback_logs = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_mse': avg_val_mse,
                'learning_rate': current_lr,
                'epoch': epoch
            }
            if train_accuracy is not None:
                callback_logs['train_accuracy'] = train_accuracy
                callback_logs['val_accuracy'] = val_accuracy
                callback_logs['val_balanced_accuracy'] = val_balanced_acc
                callback_logs['per_class_acc'] = val_per_class

            self.callbacks.on_epoch_end(epoch, callback_logs)

            # Log progress
            log_msg = (
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
                f"MSE={avg_val_mse:.4f}, LR={current_lr:.2e}"
            )
            if train_accuracy is not None:
                log_msg += f", TrainAcc={train_accuracy:.2%}, ValAcc={val_accuracy:.2%}"
                if val_balanced_acc > 0:
                    log_msg += f", BalAcc={val_balanced_acc:.2%}"
            print(log_msg)
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

    def evaluate(self, test_loader, loss_weights=None):
        """Evaluate on test set. Returns comprehensive metrics dict."""
        # Use validate_epoch for consistent metrics
        val_metrics = self.validate_epoch(test_loader, epoch=0, loss_weights=loss_weights)

        metrics = {
            'test_mse': val_metrics['mse'],
            'test_mae': val_metrics['mse'],  # Approx, use loss for MAE
            'test_loss': val_metrics['loss'],
            'test_samples': len(val_metrics.get('predictions', []))
        }

        if 'accuracy' in val_metrics:
            metrics['test_accuracy'] = val_metrics['accuracy']
            metrics['test_balanced_accuracy'] = val_metrics.get('balanced_accuracy', 0.0)

        if 'per_class_acc' in val_metrics:
            metrics['per_class_accuracy'] = val_metrics['per_class_acc']

        if 'per_class_metrics' in val_metrics:
            metrics['per_class_metrics'] = val_metrics['per_class_metrics']

        if 'predictions' in val_metrics:
            metrics['predictions'] = val_metrics['predictions']
            metrics['labels'] = val_metrics['labels']
            metrics['probabilities'] = val_metrics['probabilities']

            # Store for WandB callback to plot test confusion matrix
            self._last_test_predictions = val_metrics['predictions']
            self._last_test_labels = val_metrics['labels']

        log_msg = f"Test Results - MSE: {metrics['test_mse']:.6f}, Loss: {metrics['test_loss']:.6f}"
        if 'test_accuracy' in metrics:
            log_msg += f", Accuracy: {metrics['test_accuracy']:.2%}"
            if metrics.get('test_balanced_accuracy', 0) > 0:
                log_msg += f", Balanced: {metrics['test_balanced_accuracy']:.2%}"
        logger.info(log_msg)

        return metrics

    def get_results_path(self):
        return self.results_dir
