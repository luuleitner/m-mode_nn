#!/usr/bin/env python3
"""
Vanilla CNN Trainer - Minimal training loop for debugging.

No wandb, no weighting, no fancy features.
Just pure training with train/val/test loss tracking.

Usage:
    python vanilla_trainer.py --data-dir /path/to/cv_folds/fold_0
    python vanilla_trainer.py  # Uses default path
"""

import os
import sys
import argparse
import pickle
import json
import time
from datetime import datetime

# Add project root to path (needed to unpickle dataset objects)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

# Import the vanilla model
from vanilla_cnn import VanillaCNN


def load_datasets(data_dir):
    """Load train/val/test pickle files."""
    train_path = os.path.join(data_dir, 'train_ds.pkl')
    val_path = os.path.join(data_dir, 'val_ds.pkl')
    test_path = os.path.join(data_dir, 'test_ds.pkl')

    for path, name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} dataset not found: {path}")

    with open(train_path, 'rb') as f:
        train_ds = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_ds = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_ds = pickle.load(f)

    return train_ds, val_ds, test_ds


def compute_metrics(model, loader, criterion, device):
    """Compute loss and accuracy on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            # Handle dict format from FilteredSplitH5Dataset
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                soft_labels = batch['labels'].to(device)
                labels = soft_labels.argmax(dim=-1)  # Convert to hard labels
            else:
                data, soft_labels = batch
                data = data.to(device)
                soft_labels = soft_labels.to(device)
                labels = soft_labels.argmax(dim=-1)

            # Forward
            logits = model(data)

            # Loss (plain cross-entropy with hard labels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * data.size(0)

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def ascii_plot(history, width=60, height=15):
    """Create ASCII plot of losses."""
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    test_losses = history['test_loss']

    if len(train_losses) == 0:
        return "No data to plot"

    # Find min/max for scaling
    all_losses = train_losses + val_losses + test_losses
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1

    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    def plot_series(losses, char):
        for i, loss in enumerate(losses):
            x = int(i * (width - 1) / max(len(losses) - 1, 1))
            y = int((1 - (loss - min_loss) / loss_range) * (height - 1))
            y = max(0, min(height - 1, y))
            x = max(0, min(width - 1, x))
            grid[y][x] = char

    plot_series(train_losses, 'T')
    plot_series(val_losses, 'V')
    plot_series(test_losses, 'X')

    # Build output
    lines = []
    lines.append(f"Loss Plot (T=train, V=val, X=test) | Epochs: {len(train_losses)}")
    lines.append(f"{max_loss:.4f} |" + "-" * width)

    for row in grid:
        lines.append("        |" + ''.join(row))

    lines.append(f"{min_loss:.4f} |" + "-" * width)
    lines.append("         " + "0" + " " * (width - 5) + f"{len(train_losses)}")

    return '\n'.join(lines)


def print_epoch_summary(epoch, epochs, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, elapsed):
    """Print formatted epoch summary."""
    print(f"\nEpoch {epoch+1:3d}/{epochs}")
    print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}")
    print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%}")
    print(f"  Test:  loss={test_loss:.4f}, acc={test_acc:.2%}")
    print(f"  Time:  {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Vanilla CNN Trainer')
    parser.add_argument('--data-dir', type=str,
                        default='/vol/data/2026_wristus_wiicontroller_sgambato/day002/processed/Dataset_Envelope_CNN/Window10_Stride05_Labels_soft/run_20260131_021838/cv_folds/fold_0',
                        help='Directory containing train_ds.pkl, val_ds.pkl, test_ds.pkl')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("VANILLA CNN TRAINER")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Load data
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds = load_datasets(args.data_dir)
    print(f"  Train: {len(train_ds)} batches")
    print(f"  Val:   {len(val_ds)} batches")
    print(f"  Test:  {len(test_ds)} batches")

    # Create data loaders (batch_size=None since data is pre-batched)
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=4, pin_memory=True)

    # Check data shape
    sample = next(iter(train_loader))
    if isinstance(sample, dict):
        data_shape = sample['tokens'].shape
        label_shape = sample['labels'].shape
    else:
        data_shape = sample[0].shape
        label_shape = sample[1].shape
    print(f"\nData shape: {data_shape}")
    print(f"Label shape: {label_shape}")

    # Create model
    model = VanillaCNN(in_channels=data_shape[1], num_classes=label_shape[-1])
    model = model.to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer - PLAIN, no weights
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_val_acc = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                soft_labels = batch['labels'].to(device)
                labels = soft_labels.argmax(dim=-1)
            else:
                data, soft_labels = batch
                data = data.to(device)
                soft_labels = soft_labels.to(device)
                labels = soft_labels.argmax(dim=-1)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += data.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        val_loss, val_acc = compute_metrics(model, val_loader, criterion, device)

        # Test
        test_loss, test_acc = compute_metrics(model, test_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        epoch_time = time.time() - epoch_start
        print_epoch_summary(epoch, args.epochs, train_loss, train_acc,
                           val_loss, val_acc, test_loss, test_acc, epoch_time)

    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val accuracy: {best_val_acc:.2%}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.2%}")

    # ASCII plot
    print(f"\n{'='*60}")
    print("LOSS CURVES")
    print(f"{'='*60}")
    print(ascii_plot(history))

    # Per-class accuracy on test set
    print(f"\n{'='*60}")
    print("PER-CLASS TEST ACCURACY")
    print(f"{'='*60}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                soft_labels = batch['labels'].to(device)
                labels = soft_labels.argmax(dim=-1)
            else:
                data, soft_labels = batch
                data = data.to(device)
                soft_labels = soft_labels.to(device)
                labels = soft_labels.argmax(dim=-1)

            logits = model(data)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = ['noise', 'up', 'down', 'left', 'right']

    for cls in range(5):
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls).mean()
            print(f"  Class {cls} ({class_names[cls]:>5}): {cls_acc:.2%} ({mask.sum()} samples)")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'args': vars(args),
        'history': history,
        'best_val_acc': best_val_acc,
        'final_test_acc': history['test_acc'][-1],
        'total_time_seconds': total_time,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Also save a simple loss CSV for easy plotting
    csv_path = os.path.join(output_dir, 'losses.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n')
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},"
                    f"{history['test_loss'][i]:.6f},{history['train_acc'][i]:.6f},"
                    f"{history['val_acc'][i]:.6f},{history['test_acc'][i]:.6f}\n")
    print(f"Losses CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
