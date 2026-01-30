"""
Direct CNN Classifier Training Script

Trains a direct CNN classifier (no autoencoder) for M-mode classification.
Based on colleague's proven architecture adapted for decimated input.

Usage:
    python -m src.training.train_direct_classifier --config config/config.yaml
    python -m src.training.train_direct_classifier --config config/config.yaml --variant large
"""

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config
from src.models.direct_cnn_classifier import DirectCNNClassifier, DirectCNNClassifierLarge
import utils.logging_config as logconf

logger = logconf.get_logger("TRAIN_DIRECT_CLS")

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DirectClassifierTrainer:
    """Trainer for direct CNN classifier."""

    def __init__(self, model, config, device, output_dir, use_wandb=False):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Training config
        train_cfg = config.ml.training
        self.epochs = train_cfg.epochs
        self.lr = train_cfg.lr
        self.weight_decay = train_cfg.get('weight_decay', 0.01)
        self.grad_clip = train_cfg.regularization.get('grad_clip_norm', 1.0)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Scheduler
        scheduler_type = train_cfg.lr_scheduler.get('type', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        else:
            self.scheduler = None

        # Class weights for imbalanced data
        self.class_weights = None
        if train_cfg.class_balancing.get('enabled', False):
            # Will be set after seeing class distribution
            pass

        # Early stopping
        es_cfg = train_cfg.early_stopping
        self.early_stopping_patience = es_cfg.get('patience', 20)
        self.early_stopping_min_delta = es_cfg.get('min_delta', 1e-5)

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def compute_class_weights(self, train_loader):
        """Compute class weights from training data distribution."""
        class_counts = torch.zeros(self.model.num_classes)

        for batch in train_loader:
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

        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts + 1e-6)
        weights = weights / weights.sum() * len(class_counts)  # Normalize

        self.class_weights = weights.to(self.device)
        logger.info(f"Class distribution: {class_counts.long().tolist()}")
        logger.info(f"Class weights: {weights.tolist()}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Get data
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

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(data)

            # Loss with class weights
            if self.class_weights is not None:
                loss = F.cross_entropy(logits, hard_labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, hard_labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Metrics
            preds = logits.argmax(dim=-1)
            correct = (preds == hard_labels).sum().item()
            batch_size = data.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/batch_size:.2%}'
            })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validating"):
            if isinstance(batch, dict):
                data = batch['tokens'].to(self.device)
                labels = batch['labels']
            else:
                data = batch[0].to(self.device)
                labels = batch[1]

            # Transpose
            data = data.permute(0, 1, 3, 2)

            # Handle soft labels
            if labels.dim() > 1 and labels.shape[-1] > 1:
                hard_labels = labels.argmax(dim=-1)
            else:
                hard_labels = labels.squeeze() if labels.dim() > 1 else labels
            hard_labels = hard_labels.long().to(self.device)

            # Forward
            logits = self.model(data)

            # Loss
            if self.class_weights is not None:
                loss = F.cross_entropy(logits, hard_labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, hard_labels)

            # Metrics
            preds = logits.argmax(dim=-1)
            batch_size = data.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == hard_labels).sum().item()
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())

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

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model (val_acc={self.best_val_acc:.2%})")

    def train(self, train_loader, val_loader):
        """Full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")

        # Compute class weights
        if self.config.ml.training.class_balancing.get('enabled', False):
            self.compute_class_weights(train_loader)

        # WandB init
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb.project + "_direct_cnn",
                name=f"direct_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model': 'DirectCNNClassifier',
                    'epochs': self.epochs,
                    'lr': self.lr,
                    'flatten_dim': self.model.flatten_dim,
                    'num_classes': self.model.num_classes
                }
            )
            wandb.watch(self.model, log='gradients', log_freq=100)

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, per_class_acc = self.validate(val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Check improvement
            is_best = False
            if val_acc > self.best_val_acc + self.early_stopping_min_delta:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Print epoch summary
            class_acc_str = " | ".join([f"C{c}:{a:.1%}" for c, a in per_class_acc.items()])
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%}")
            print(f"  Per-class: {class_acc_str}")
            print(f"  LR: {current_lr:.2e} | Best: {self.best_val_acc:.2%}")

            # WandB logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'lr': current_lr
                }
                for cls, acc in per_class_acc.items():
                    log_dict[f'val/class_{cls}_acc'] = acc
                wandb.log(log_dict)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Finish
        if self.use_wandb:
            wandb.finish()

        logger.info(f"Training complete. Best val acc: {self.best_val_acc:.2%}")
        return self.history


def load_datasets(config, data_dir=None):
    """Load train/val datasets from pickle."""
    import pickle

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

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    logger.info(f"Loading datasets from: {pickle_path}")

    with open(train_path, 'rb') as f:
        train_ds = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_ds = pickle.load(f)

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description='Train Direct CNN Classifier')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data-dir', '-d', type=str, default=None,
                        help='Directory with train_ds.pkl/val_ds.pkl (overrides config)')
    parser.add_argument('--variant', '-v', type=str, default='standard',
                        choices=['standard', 'large'], help='Model variant')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, create_dirs=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Input dimensions from config
    input_pulses = config.preprocess.tokenization.window  # 10
    input_depth = 130  # After decimation

    # Create model
    num_classes = 3  # noise, upward, downward
    if args.variant == 'large':
        model = DirectCNNClassifierLarge(
            in_channels=3,
            input_pulses=input_pulses,
            input_depth=input_depth,
            num_classes=num_classes
        )
    else:
        model = DirectCNNClassifier(
            in_channels=3,
            input_pulses=input_pulses,
            input_depth=input_depth,
            num_classes=num_classes
        )

    model.print_architecture()

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = args.data_dir if args.data_dir else config.get_train_data_root().strip()
        output_dir = os.path.join(
            base_dir,
            f'direct_cnn_{args.variant}_{timestamp}'
        )
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_ds, val_ds = load_datasets(config, data_dir=args.data_dir)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=4)

    # Create trainer
    trainer = DirectClassifierTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        use_wandb=not args.no_wandb
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    # Final evaluation on validation set
    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)
    val_loss, val_acc, per_class_acc = trainer.validate(val_loader)
    print(f"Validation Accuracy: {val_acc:.2%}")
    print(f"Validation Loss: {val_loss:.4f}")
    print("\nPer-class Accuracy:")
    class_names = ['noise', 'upward', 'downward']
    for cls, acc in per_class_acc.items():
        print(f"  {class_names[cls]}: {acc:.2%}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
