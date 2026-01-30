"""
CNN Classifier Training

Trains a direct CNN classifier (no autoencoder) for M-mode classification.
Based on colleague's proven architecture adapted for decimated input.

Features:
- Full WandB integration with metrics, plots, and model tracking
- Checkpoint saving (best + periodic)
- Training visualizations (confusion matrix, loss curves, per-class metrics)
- Early stopping
- Class-weighted loss for imbalanced data
- Restart from checkpoint support

Usage:
    python -m src.training.train_cnn_cls --config config/config.yaml
    python -m src.training.train_cnn_cls --config config/config.yaml --data-dir /path/to/data
    python -m src.training.train_cnn_cls --config config/config.yaml --restart
"""

import os
import sys
import argparse
import pickle
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import numpy as np

# Use non-interactive backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.models.direct_cnn_classifier import DirectCNNClassifier
from src.data.datasets import create_filtered_split_datasets
import utils.logging_config as logconf

logger = logconf.get_logger("TRAIN_DIRECT_CLS")

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. WandB logging disabled.")

# Load class names from centralized config
_label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
with open(_label_config_path) as _f:
    _label_config = yaml.safe_load(_f)
_classes_config = _label_config.get('classes', {})
CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(_classes_config.get('num_classes', 3))]
CLASS_COLORS = _classes_config.get('colors', {})


def focal_loss(logits, targets, weight=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance.

    Down-weights easy examples (high confidence correct predictions)
    and focuses on hard examples (misclassifications, minority classes).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw model outputs (B, num_classes)
        targets: Ground truth labels (B,)
        weight: Per-class weights (num_classes,)
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to cross-entropy.
               gamma=2 is the default from the paper.
        reduction: 'mean', 'sum', or 'none'
    """
    ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')

    # Get probability of correct class
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    focal_loss = focal_weight * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


class DirectClassifierTrainer:
    """
    Full-featured trainer for direct CNN classifier.

    Includes:
    - WandB logging with plots
    - Checkpoint management
    - Early stopping
    - Training visualizations
    - Class-weighted loss
    """

    def __init__(self, model, config, device, output_dir, use_wandb=True):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        # Training config
        train_cfg = config.ml.training
        self.epochs = train_cfg.epochs
        self.lr = train_cfg.lr
        self.weight_decay = getattr(train_cfg, 'weight_decay', 0.01)
        self.grad_clip = train_cfg.regularization.get('grad_clip_norm', 1.0)

        # Scheduler config
        self.scheduler_type = train_cfg.lr_scheduler.get('type', 'cosine')

        # Class imbalance handling (class weights and/or focal loss)
        self.class_weights = None
        self.use_weighted_loss = False
        self.use_focal_loss = True  # Default
        self.focal_gamma = 2.0

        imbalance_cfg = train_cfg.get('imbalance', None) if hasattr(train_cfg, 'get') else getattr(train_cfg, 'imbalance', None)
        if imbalance_cfg:
            # Class weights config
            weights_cfg = imbalance_cfg.get('class_weights', None) if hasattr(imbalance_cfg, 'get') else getattr(imbalance_cfg, 'class_weights', None)
            if weights_cfg:
                self.use_weighted_loss = weights_cfg.get('enabled', False) if hasattr(weights_cfg, 'get') else getattr(weights_cfg, 'enabled', False)

            # Focal loss config
            focal_cfg = imbalance_cfg.get('focal_loss', None) if hasattr(imbalance_cfg, 'get') else getattr(imbalance_cfg, 'focal_loss', None)
            if focal_cfg:
                self.use_focal_loss = focal_cfg.get('enabled', True) if hasattr(focal_cfg, 'get') else getattr(focal_cfg, 'enabled', True)
                self.focal_gamma = focal_cfg.get('gamma', 2.0) if hasattr(focal_cfg, 'get') else getattr(focal_cfg, 'gamma', 2.0)

        logger.info(f"Imbalance config: class_weights={self.use_weighted_loss}, focal_loss={self.use_focal_loss}")

        # Early stopping config (use OmegaConf-compatible access)
        es_cfg = train_cfg.get('early_stopping', None) if hasattr(train_cfg, 'get') else getattr(train_cfg, 'early_stopping', None)
        if es_cfg is not None:
            self.early_stopping_enabled = es_cfg.get('enabled', True) if hasattr(es_cfg, 'get') else getattr(es_cfg, 'enabled', True)
            self.early_stopping_patience = es_cfg.get('patience', 20) if hasattr(es_cfg, 'get') else getattr(es_cfg, 'patience', 20)
            self.early_stopping_min_delta = es_cfg.get('min_delta', 1e-5) if hasattr(es_cfg, 'get') else getattr(es_cfg, 'min_delta', 1e-5)
            self.early_stopping_monitor = es_cfg.get('monitor', 'val_loss') if hasattr(es_cfg, 'get') else getattr(es_cfg, 'monitor', 'val_loss')
        else:
            self.early_stopping_enabled = False
            self.early_stopping_patience = 20
            self.early_stopping_min_delta = 1e-5
            self.early_stopping_monitor = 'val_loss'

        # Checkpoint config
        ckpt_cfg = getattr(train_cfg, 'checkpointing', None)
        self.save_every_n_epochs = getattr(ckpt_cfg, 'save_every_n_epochs', 10) if ckpt_cfg else 10

        # Visualization config
        val_cfg = getattr(train_cfg, 'validation', None)
        self.plot_every_n_epochs = getattr(val_cfg, 'plot_every_n_epochs', 5) if val_cfg else 5

        # History tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_balanced_acc': [],
            'lr': [],
            'per_class_acc': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_balanced_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        # Will be set during training
        self.optimizer = None
        self.scheduler = None
        self._wandb_run = None

    def _create_optimizer(self):
        """Create optimizer based on config."""
        opt_type = self.config.ml.training.optimizer.get('type', 'adamw').lower()

        if opt_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=tuple(self.config.ml.training.optimizer.get('betas', [0.9, 0.999]))
            )
        elif opt_type == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=tuple(self.config.ml.training.optimizer.get('betas', [0.9, 0.999]))
            )
        elif opt_type == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.config.ml.training.optimizer.get('momentum', 0.9),
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        logger.info(f"Optimizer: {opt_type}, LR: {self.lr}, Weight Decay: {self.weight_decay}")

    def _create_scheduler(self, train_loader):
        """Create learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif self.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
            )
        elif self.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.lr * 10,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader)
            )
        else:
            self.scheduler = None

        logger.info(f"Scheduler: {self.scheduler_type}")

    def compute_class_weights(self, train_loader):
        """Compute class weights from training data distribution."""
        logger.info("Computing class weights from training data...")
        class_counts = torch.zeros(self.model.num_classes)

        for batch in tqdm(train_loader, desc="Counting classes"):
            if isinstance(batch, dict):
                labels = batch['labels']
            else:
                labels = batch[1]

            # Handle soft labels
            if labels.dim() > 1 and labels.shape[-1] > 1:
                hard_labels = labels.argmax(dim=-1)
            else:
                hard_labels = labels.squeeze() if labels.dim() > 1 else labels

            for cls in range(self.model.num_classes):
                class_counts[cls] += (hard_labels == cls).sum().item()

        # Inverse frequency weighting (more aggressive for extreme imbalance)
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts + 1e-6)

        # For extreme imbalance, apply power scaling (but not too aggressive to avoid oscillation)
        # Power=1.0 is standard inverse frequency, power=1.5 was too aggressive
        max_ratio = class_counts.max() / (class_counts.min() + 1e-6)
        weight_power = 1.0  # Standard inverse frequency (was 1.5, too aggressive)
        if max_ratio > 10:
            logger.info(f"Extreme imbalance detected (ratio={max_ratio:.1f}), using weight power={weight_power}")
            weights = weights ** weight_power

        weights = weights / weights.sum() * len(class_counts)  # Normalize

        self.class_weights = weights.to(self.device)

        # Log class distribution
        logger.info(f"Class distribution: {class_counts.long().tolist()}")
        logger.info(f"Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        logger.info(f"Using focal loss: {self.use_focal_loss} (gamma={self.focal_gamma})")

        # Log to WandB
        if self.use_wandb and self._wandb_run:
            class_dist = {f'class_distribution/{CLASS_NAMES[i]}': class_counts[i].item()
                          for i in range(len(class_counts))}
            wandb.log(class_dist)

        return class_counts

    def _prepare_batch(self, batch):
        """Prepare batch for forward pass."""
        if isinstance(batch, dict):
            data = batch['tokens'].to(self.device)
            labels = batch['labels']
        else:
            data = batch[0].to(self.device)
            labels = batch[1]

        # Transpose: [B, C, Depth, Pulses] -> [B, C, Pulses, Depth]
        data = data.permute(0, 1, 3, 2)

        # Handle soft labels
        if labels.dim() > 1 and labels.shape[-1] > 1:
            hard_labels = labels.argmax(dim=-1)
        else:
            hard_labels = labels.squeeze() if labels.dim() > 1 else labels
        hard_labels = hard_labels.long().to(self.device)

        return data, hard_labels

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            data, hard_labels = self._prepare_batch(batch)
            batch_size = data.size(0)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(data)

            # Loss with class weights and focal loss
            if self.use_focal_loss:
                loss = focal_loss(logits, hard_labels, weight=self.class_weights, gamma=self.focal_gamma)
            elif self.class_weights is not None:
                loss = F.cross_entropy(logits, hard_labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, hard_labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (returns the total norm before clipping)
            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                # Compute gradient norm without clipping
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]))

            self.optimizer.step()

            # OneCycle scheduler steps per batch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Metrics
            preds = logits.argmax(dim=-1)
            correct = (preds == hard_labels).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/batch_size:.1%}'
            })

            # Log batch metrics to WandB
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/accuracy': correct / batch_size,
                    'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'batch/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    'batch/grad_clipped': int(grad_norm > self.grad_clip) if self.grad_clip > 0 else 0
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class_acc = {}
        for cls in range(self.model.num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_acc[cls] = (all_preds[mask] == cls).mean()
            else:
                per_class_acc[cls] = 0.0

        return avg_loss, avg_acc, per_class_acc

    @torch.no_grad()
    def validate(self, val_loader, desc="Validating"):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(val_loader, desc=desc):
            data, hard_labels = self._prepare_batch(batch)
            batch_size = data.size(0)

            # Forward
            logits = self.model(data)
            probs = F.softmax(logits, dim=-1)

            # Loss (same as training for consistent metrics)
            if self.use_focal_loss:
                loss = focal_loss(logits, hard_labels, weight=self.class_weights, gamma=self.focal_gamma)
            elif self.class_weights is not None:
                loss = F.cross_entropy(logits, hard_labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, hard_labels)

            # Metrics
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * batch_size
            total_correct += (preds == hard_labels).sum().item()
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        per_class_acc = {}
        for cls in range(self.model.num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_acc[cls] = (all_preds[mask] == cls).mean()
            else:
                per_class_acc[cls] = 0.0

        # Per-class precision, recall, F1 (critical for imbalanced data)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds,
            labels=list(range(self.model.num_classes)),
            zero_division=0
        )
        per_class_metrics = {
            cls: {'precision': precision[cls], 'recall': recall[cls], 'f1': f1[cls], 'support': support[cls]}
            for cls in range(self.model.num_classes)
        }

        # Mean confidence per class (for calibration analysis)
        mean_confidence = {}
        for cls in range(self.model.num_classes):
            mask = all_preds == cls
            if mask.sum() > 0:
                mean_confidence[cls] = all_probs[mask, cls].mean()
            else:
                mean_confidence[cls] = 0.0

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'per_class_acc': per_class_acc,
            'per_class_metrics': per_class_metrics,
            'mean_confidence': mean_confidence,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

    def save_checkpoint(self, epoch, is_best=False, is_periodic=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'class_weights': self.class_weights.cpu() if self.class_weights is not None else None,
            'model_config': {
                'in_channels': self.model.in_channels,
                'input_pulses': self.model.input_pulses,
                'input_depth': self.model.input_depth,
                'num_classes': self.model.num_classes,
                'flatten_dim': self.model.flatten_dim
            }
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        ckpt_dir = os.path.join(self.output_dir, 'checkpoints')

        # Save latest
        latest_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model (epoch {epoch+1}, val_acc={self.best_val_acc:.2%})")

            # Also save to wandb
            if self.use_wandb:
                wandb.save(best_path)

        # Save periodic
        if is_periodic:
            periodic_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            logger.info(f"Saved periodic checkpoint: epoch {epoch+1}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = checkpoint.get('history', self.history)
        self.current_epoch = checkpoint.get('epoch', 0) + 1

        if checkpoint.get('class_weights') is not None:
            self.class_weights = checkpoint['class_weights'].to(self.device)

        logger.info(f"Resumed from epoch {self.current_epoch}, best_val_acc={self.best_val_acc:.2%}")
        return self.current_epoch

    def plot_training_curves(self, epoch):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax.plot(self.history['val_loss'], label='Val Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[0, 1]
        ax.plot(self.history['train_acc'], label='Train Acc', color='blue')
        ax.plot(self.history['val_acc'], label='Val Acc', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Learning rate
        ax = axes[1, 0]
        ax.plot(self.history['lr'], color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Per-class accuracy over time
        ax = axes[1, 1]
        if self.history['per_class_acc']:
            for cls in range(self.model.num_classes):
                accs = [p.get(cls, 0) for p in self.history['per_class_acc']]
                ax.plot(accs, label=CLASS_NAMES[cls])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        plt.suptitle(f'Training Progress - Epoch {epoch+1}', fontsize=14)
        plt.tight_layout()

        # Save
        plot_path = os.path.join(self.output_dir, 'plots', f'training_curves_epoch_{epoch+1}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log to WandB
        if self.use_wandb:
            wandb.log({'plots/training_curves': wandb.Image(plot_path)})

        return plot_path

    def plot_confusion_matrix(self, predictions, labels, epoch, split='val'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, predictions, labels=range(self.model.num_classes))

        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(CLASS_NAMES)),
            yticks=np.arange(len(CLASS_NAMES)),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=f'Confusion Matrix ({split}) - Epoch {epoch+1}'
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add text annotations
        thresh = cm_norm.max() / 2.
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                text = f'{cm_norm[i, j]:.1%}\n({cm[i, j]})'
                ax.text(j, i, text, ha='center', va='center',
                        color='white' if cm_norm[i, j] > thresh else 'black',
                        fontsize=10)

        plt.tight_layout()

        # Save
        plot_path = os.path.join(self.output_dir, 'plots', f'confusion_matrix_{split}_epoch_{epoch+1}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log to WandB
        if self.use_wandb:
            wandb.log({f'plots/confusion_matrix_{split}': wandb.Image(plot_path)})

        return plot_path

    def init_wandb(self):
        """Initialize WandB logging."""
        if not self.use_wandb:
            return

        wandb_config = {
            'model': type(self.model).__name__,
            'model_variant': 'standard',
            'flatten_dim': self.model.flatten_dim,
            'num_classes': self.model.num_classes,
            'input_shape': (self.model.in_channels, self.model.input_pulses, self.model.input_depth),
            'epochs': self.epochs,
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'optimizer': self.config.ml.training.optimizer.get('type', 'adamw'),
            'scheduler': self.scheduler_type,
            'grad_clip': self.grad_clip,
            'use_weighted_loss': self.use_weighted_loss,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'early_stopping_patience': self.early_stopping_patience
        }

        # Get WandB config from yaml
        wandb_cfg = self.config.wandb
        project = getattr(wandb_cfg, 'project', 'm-mode_nn') + '_direct_cnn'
        name = f"direct_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        api_key = getattr(wandb_cfg, 'api_key', None)

        if api_key:
            wandb.login(key=api_key)

        self._wandb_run = wandb.init(
            project=project,
            name=name,
            config=wandb_config,
            dir=self.output_dir,
            reinit=True
        )

        # Define metrics
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

        # Watch model
        wandb.watch(self.model, log='gradients', log_freq=100)

        logger.info(f"WandB initialized: {project}/{name}")

    def train(self, train_loader, val_loader, test_loader=None, restart=False):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Feature dimension: {self.model.flatten_dim}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Early stopping: enabled={self.early_stopping_enabled}, monitor={self.early_stopping_monitor}, patience={self.early_stopping_patience}")

        # Create optimizer and scheduler
        self._create_optimizer()
        self._create_scheduler(train_loader)

        # Initialize WandB
        self.init_wandb()

        # Compute class weights if needed
        if self.use_weighted_loss:
            self.compute_class_weights(train_loader)

        # Restart from checkpoint if requested
        start_epoch = 0
        if restart:
            ckpt_path = os.path.join(self.output_dir, 'checkpoints', 'checkpoint_latest.pth')
            if os.path.exists(ckpt_path):
                start_epoch = self.load_checkpoint(ckpt_path)
            else:
                logger.warning(f"No checkpoint found at {ckpt_path}, starting from scratch")

        # Training loop
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc, train_per_class = self.train_epoch(train_loader, epoch)

            # Validate
            val_results = self.validate(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            val_per_class = val_results['per_class_acc']
            val_per_class_metrics = val_results['per_class_metrics']
            val_mean_confidence = val_results['mean_confidence']

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['per_class_acc'].append(val_per_class)

            # Compute balanced accuracy (mean of per-class recalls)
            val_balanced_acc = np.mean(list(val_per_class.values())) if val_per_class else 0.0
            self.history['val_balanced_acc'].append(val_balanced_acc)

            # Check for improvement
            is_best = False
            if self.early_stopping_monitor == 'val_loss':
                improved = val_loss < self.best_val_loss - self.early_stopping_min_delta
            elif self.early_stopping_monitor == 'val_balanced_accuracy':
                improved = val_balanced_acc > self.best_val_balanced_acc + self.early_stopping_min_delta
            else:  # val_accuracy
                improved = val_acc > self.best_val_acc + self.early_stopping_min_delta

            if improved:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_val_balanced_acc = val_balanced_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

            # Save checkpoints
            is_periodic = (epoch + 1) % self.save_every_n_epochs == 0
            self.save_checkpoint(epoch, is_best=is_best, is_periodic=is_periodic)

            # Generate plots
            if (epoch + 1) % self.plot_every_n_epochs == 0 or is_best:
                self.plot_training_curves(epoch)
                self.plot_confusion_matrix(
                    val_results['predictions'],
                    val_results['labels'],
                    epoch, split='val'
                )

            # Print epoch summary
            class_acc_str = " | ".join([f"{CLASS_NAMES[c]}:{a:.1%}" for c, a in val_per_class.items()])
            class_f1_str = " | ".join([f"{CLASS_NAMES[c]}:{m['f1']:.2f}" for c, m in val_per_class_metrics.items()])
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%}, balanced_acc={val_balanced_acc:.2%}")
            print(f"  Per-class acc: {class_acc_str}")
            print(f"  Per-class F1:  {class_f1_str}")
            print(f"  Train-Val gap: loss={train_loss - val_loss:+.4f}, acc={train_acc - val_acc:+.2%}")
            if self.early_stopping_monitor == 'val_balanced_accuracy':
                print(f"  LR: {current_lr:.2e} | Best balanced: {self.best_val_balanced_acc:.2%} (epoch {self.best_epoch+1})")
            else:
                print(f"  LR: {current_lr:.2e} | Best: {self.best_val_acc:.2%} (epoch {self.best_epoch+1})")
            print(f"{'='*60}")

            # Log to WandB
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/balanced_accuracy': val_balanced_acc,
                    'train/learning_rate': current_lr,
                    # Train-val gap (primary overfitting indicators)
                    'gap/loss': train_loss - val_loss,
                    'gap/accuracy': train_acc - val_acc,
                }
                # Per-class accuracy
                for cls, acc in val_per_class.items():
                    log_dict[f'val/class_{CLASS_NAMES[cls]}_acc'] = acc
                for cls, acc in train_per_class.items():
                    log_dict[f'train/class_{CLASS_NAMES[cls]}_acc'] = acc
                # Per-class precision/recall/F1 (critical for imbalanced data)
                for cls, metrics in val_per_class_metrics.items():
                    log_dict[f'val/class_{CLASS_NAMES[cls]}_precision'] = metrics['precision']
                    log_dict[f'val/class_{CLASS_NAMES[cls]}_recall'] = metrics['recall']
                    log_dict[f'val/class_{CLASS_NAMES[cls]}_f1'] = metrics['f1']
                # Mean confidence per class (calibration indicator)
                for cls, conf in val_mean_confidence.items():
                    log_dict[f'val/class_{CLASS_NAMES[cls]}_confidence'] = conf
                wandb.log(log_dict)

            # Early stopping
            if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {self.early_stopping_patience} epochs)")
                break

        # Final evaluation on test set
        if test_loader is not None:
            logger.info("\nRunning final evaluation on test set...")
            test_results = self.validate(test_loader, desc="Testing")

            # Plot test confusion matrix
            self.plot_confusion_matrix(
                test_results['predictions'],
                test_results['labels'],
                self.current_epoch, split='test'
            )

            # Log test metrics
            if self.use_wandb:
                test_log = {
                    'test/loss': test_results['loss'],
                    'test/accuracy': test_results['accuracy'],
                }
                for cls, acc in test_results['per_class_acc'].items():
                    test_log[f'test/class_{CLASS_NAMES[cls]}_acc'] = acc
                # Per-class precision/recall/F1
                for cls, metrics in test_results['per_class_metrics'].items():
                    test_log[f'test/class_{CLASS_NAMES[cls]}_precision'] = metrics['precision']
                    test_log[f'test/class_{CLASS_NAMES[cls]}_recall'] = metrics['recall']
                    test_log[f'test/class_{CLASS_NAMES[cls]}_f1'] = metrics['f1']
                # Confidence
                for cls, conf in test_results['mean_confidence'].items():
                    test_log[f'test/class_{CLASS_NAMES[cls]}_confidence'] = conf
                wandb.log(test_log)

            # Print test results
            print("\n" + "=" * 60)
            print("TEST SET RESULTS")
            print("=" * 60)
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test Accuracy: {test_results['accuracy']:.2%}")
            print("\nPer-class Accuracy:")
            for cls, acc in test_results['per_class_acc'].items():
                print(f"  {CLASS_NAMES[cls]}: {acc:.2%}")
            print("\nClassification Report:")
            print(classification_report(
                test_results['labels'],
                test_results['predictions'],
                target_names=CLASS_NAMES,
                digits=3
            ))
            print("=" * 60)

            # Save test predictions
            np.savez(
                os.path.join(self.output_dir, 'test_predictions.npz'),
                predictions=test_results['predictions'],
                labels=test_results['labels'],
                probabilities=test_results['probabilities']
            )

        # Finish WandB
        if self.use_wandb:
            wandb.run.summary['best_val_accuracy'] = self.best_val_acc
            wandb.run.summary['best_val_balanced_accuracy'] = self.best_val_balanced_acc
            wandb.run.summary['best_val_loss'] = self.best_val_loss
            wandb.run.summary['best_epoch'] = self.best_epoch + 1
            wandb.run.summary['total_epochs'] = self.current_epoch + 1
            if test_loader is not None:
                wandb.run.summary['test_accuracy'] = test_results['accuracy']
            wandb.finish()

        logger.info(f"\nTraining complete. Best val acc: {self.best_val_acc:.2%} at epoch {self.best_epoch+1}")
        logger.info(f"Results saved to: {self.output_dir}")

        return self.history


def load_datasets(config, data_dir=None):
    """Load or create train/val/test datasets."""
    # Use provided data_dir or fall back to config
    if data_dir:
        pickle_path = data_dir
    else:
        pickle_path = config.get_train_data_root().strip()

    if not pickle_path or not os.path.isdir(pickle_path):
        raise ValueError(
            f"Invalid data directory: '{pickle_path}'\n"
            "Please specify --data-dir or set train_base_data_path in config.yaml"
        )

    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    # Check if we should load from pickle or create from H5
    load_from_pickle = config.ml.loading.get('load_data_pickle', True) if hasattr(config.ml.loading, 'get') else getattr(config.ml.loading, 'load_data_pickle', True)

    if load_from_pickle:
        # Load existing pickle files
        for path, name in [(train_path, 'Train'), (val_path, 'Validation'), (test_path, 'Test')]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} dataset not found: {path}\n"
                    "Run 'python -m src.data.precompute_datasets --config config.yaml' first,\n"
                    "or set 'load_data_pickle: false' in config to create datasets automatically."
                )

        logger.info(f"Loading datasets from pickle: {pickle_path}")
        with open(train_path, 'rb') as f:
            train_ds = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_ds = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_ds = pickle.load(f)
    else:
        # Create datasets from H5 files
        logger.info("Creating datasets from H5 files...")
        train_ds, test_ds, val_ds = create_filtered_split_datasets(
            **config.get_dataset_parameters()
        )

        # Save for future use
        logger.info(f"Saving datasets to pickle: {pickle_path}")
        with open(train_path, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_ds, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_ds, f)
        logger.info("Datasets saved. Set 'load_data_pickle: true' to reuse them.")

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser(description='Train Direct CNN Classifier')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data-dir', '-d', type=str, default=None,
                        help='Directory with train_ds.pkl/val_ds.pkl/test_ds.pkl (overrides config)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--restart', '-r', action='store_true', help='Restart from latest checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, create_dirs=False)
    setup_environment(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Input dimensions from config
    input_pulses = config.preprocess.tokenization.window  # 10
    input_depth = 130  # After decimation

    # Load label config for class definitions
    label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
    with open(label_config_path) as f:
        label_config = yaml.safe_load(f)
    num_classes = label_config['classes']['num_classes']
    class_names = label_config['classes']['names']

    # Get dropout settings from config
    dropout_config = config.ml.training.regularization.get('dropout', {})
    spatial_dropout = dropout_config.get('spatial', 0.1)
    fc_dropout = dropout_config.get('fc', 0.5)

    # Create model
    model = DirectCNNClassifier(
        in_channels=3,
        input_pulses=input_pulses,
        input_depth=input_depth,
        num_classes=num_classes,
        dropout=fc_dropout,
        spatial_dropout=spatial_dropout
    )

    model.print_architecture()

    # Load datasets
    data_dir = args.data_dir
    train_ds, val_ds, test_ds = load_datasets(config, data_dir=data_dir)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = data_dir if data_dir else config.get_train_data_root().strip()
        output_dir = os.path.join(base_dir, f'direct_cnn_{timestamp}')

    # Create data loaders
    resource_cfg = config.get_resource_config()
    train_loader = DataLoader(
        train_ds, batch_size=None, shuffle=True,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=None, shuffle=False,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )
    test_loader = DataLoader(
        test_ds, batch_size=None, shuffle=False,
        num_workers=resource_cfg['num_workers'],
        pin_memory=resource_cfg['pin_memory']
    )

    # Create trainer
    trainer = DirectClassifierTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        use_wandb=not args.no_wandb
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        restart=args.restart
    )

    return 0


if __name__ == '__main__':
    exit(main())
