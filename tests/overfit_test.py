"""
Overfit Test - Can the CNN memorize a tiny subset?

If the model can't reach ~100% accuracy on 50 samples,
either the architecture can't learn the patterns or the data has issues.

Usage:
    python -m tests.overfit_test --data-dir /path/to/data
    python -m tests.overfit_test --data-dir /path/to/data --samples 100
"""

import os
import sys
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.models.direct_cnn_classifier import DirectCNNClassifier


def main():
    parser = argparse.ArgumentParser(description='Overfit test on tiny subset')
    parser.add_argument('data', type=str,
                        help='Path to pickle file (train_ds.pkl) or directory containing it')
    parser.add_argument('--samples', '-n', type=int, default=50,
                        help='Number of samples to use (default: 50)')
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset - accept file or directory
    if args.data.endswith('.pkl'):
        train_path = args.data
        data_dir = os.path.dirname(args.data)
    else:
        train_path = os.path.join(args.data, 'train_ds.pkl')
        data_dir = args.data

    print(f"Loading: {train_path}")
    with open(train_path, 'rb') as f:
        train_ds = pickle.load(f)

    # Get a balanced subset (equal samples per class if possible)
    print(f"\nSelecting {args.samples} samples...")

    # Check if dataset is pre-batched
    sample = train_ds[0]
    if isinstance(sample, dict):
        sample_labels = sample['labels']
    else:
        sample_labels = sample[1]

    # Detect if batched: labels have shape (batch_size, num_classes) for soft labels
    is_batched = sample_labels.dim() == 2 and sample_labels.shape[-1] <= 10  # soft labels have <=10 classes
    if is_batched:
        num_classes = sample_labels.shape[-1]
        print(f"Dataset is PRE-BATCHED with soft labels, {num_classes} classes")
    else:
        num_classes = 5  # Default
        print(f"Dataset has individual samples, assuming {num_classes} classes")

    if is_batched:
        # For pre-batched dataset, concatenate samples from first few batches
        batch = train_ds[0]
        if isinstance(batch, dict):
            all_tokens = batch['tokens']
            all_labels = batch['labels']
        else:
            all_tokens = batch[0]
            all_labels = batch[1]

        batch_size = all_tokens.shape[0]
        print(f"First batch has {batch_size} samples")

        # Concatenate more batches if needed
        batch_idx = 1
        while all_tokens.shape[0] < args.samples and batch_idx < len(train_ds):
            batch = train_ds[batch_idx]
            if isinstance(batch, dict):
                all_tokens = torch.cat([all_tokens, batch['tokens']], dim=0)
                all_labels = torch.cat([all_labels, batch['labels']], dim=0)
            else:
                all_tokens = torch.cat([all_tokens, batch[0]], dim=0)
                all_labels = torch.cat([all_labels, batch[1]], dim=0)
            batch_idx += 1

        # Take only requested samples
        all_tokens = all_tokens[:args.samples]
        all_labels = all_labels[:args.samples]
        n_samples = all_tokens.shape[0]
        print(f"Using {n_samples} samples from {batch_idx} batch(es)")

        # Class distribution
        hard_labels = all_labels.argmax(dim=-1)
        print("Subset class distribution:")
        for cls in range(num_classes):
            count = (hard_labels == cls).sum().item()
            print(f"  Class {cls}: {count} samples")

        # Wrapper for iteration
        class SimpleDataset:
            def __init__(self, tokens, labels):
                self.tokens = tokens
                self.labels = labels
            def __len__(self):
                return self.tokens.shape[0]
            def __getitem__(self, idx):
                return {'tokens': self.tokens[idx], 'labels': self.labels[idx]}

        subset = SimpleDataset(all_tokens, all_labels)
        sample_shape = all_tokens.shape[1:]
        in_channels, input_depth, input_pulses = sample_shape

    else:
        # Non-batched dataset - original logic
        class_indices = {i: [] for i in range(num_classes)}
        for idx in range(len(train_ds)):
            sample = train_ds[idx]
            label = sample['labels'] if isinstance(sample, dict) else sample[1]
            cls = int(label.item()) if label.dim() == 0 else int(label.squeeze().item())
            class_indices[cls].append(idx)

        print("Full dataset class distribution:")
        for cls, indices in class_indices.items():
            print(f"  Class {cls}: {len(indices)} samples")

        samples_per_class = args.samples // num_classes
        selected_indices = []
        for cls in range(num_classes):
            available = class_indices[cls]
            n_select = min(samples_per_class, len(available))
            selected_indices.extend(available[:n_select])

        subset = Subset(train_ds, selected_indices[:args.samples])
        print(f"Using {len(subset)} samples")

        sample = subset[0]
        sample_shape = sample['tokens'].shape if isinstance(sample, dict) else sample[0].shape
        in_channels, input_depth, input_pulses = sample_shape
    print(f"\nInput shape: {sample_shape} (C, Depth, Pulses)")

    # Create model with NO regularization
    model = DirectCNNClassifier(
        in_channels=in_channels,
        input_depth=input_depth,
        input_pulses=input_pulses,
        num_classes=num_classes,
        dropout=0.0,           # No dropout
        spatial_dropout=0.0,   # No spatial dropout
        width_multiplier=2     # Same as main training
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Regularization: DISABLED (dropout=0, spatial_dropout=0)")

    # Create dataloader (no shuffle - we want to see exact same samples)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

    # Optimizer with no weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Training loop
    print(f"\n{'='*60}")
    print(f"OVERFIT TEST: Can model memorize {args.samples} samples?")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Target: ~100% train accuracy\n")

    history = {'loss': [], 'acc': []}

    for epoch in range(args.epochs):
        model.train()

        for batch in loader:
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                labels = batch['labels']
            else:
                data = batch[0].to(device)
                labels = batch[1]

            # Handle soft labels
            if labels.dim() > 1 and labels.shape[-1] > 1:
                hard_labels = labels.argmax(dim=-1)
            else:
                hard_labels = labels.squeeze() if labels.dim() > 1 else labels
            hard_labels = hard_labels.long().to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, hard_labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == hard_labels).float().mean().item()

        history['loss'].append(loss.item())
        history['acc'].append(acc)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or acc >= 0.99:
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:.1%}")

        # Early exit if we've memorized
        if acc >= 0.99:
            print(f"\n✓ SUCCESS: Model memorized data at epoch {epoch+1}")
            break

    # Final assessment
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Final accuracy: {history['acc'][-1]:.1%}")
    print(f"Final loss: {history['loss'][-1]:.4f}")

    if history['acc'][-1] >= 0.95:
        print("\n✓ PASS: Model CAN learn the patterns")
        print("  → Issue is likely overfitting/generalization, not signal quality")
    elif history['acc'][-1] >= 0.60:
        print("\n⚠ PARTIAL: Model learns something but struggles")
        print("  → Signal may be weak or architecture suboptimal")
    else:
        print("\n✗ FAIL: Model CANNOT memorize even 50 samples")
        print("  → Possible issues:")
        print("    - Labels may be incorrect/noisy")
        print("    - Signal doesn't contain discriminative information")
        print("    - Architecture fundamentally wrong for this data")

    # Plot learning curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['acc'])
    ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Target (100%)')
    ax2.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Random (20%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Overfit Test: {args.samples} samples, {args.epochs} epochs')
    plt.tight_layout()

    plot_path = os.path.join(data_dir, 'overfit_test.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")

    return 0 if history['acc'][-1] >= 0.95 else 1


if __name__ == '__main__':
    exit(main())
