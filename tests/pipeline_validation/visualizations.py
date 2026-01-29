"""
Visualization utilities for pipeline validation.

Generates plots for:
1. Raw synthetic data samples (one per class)
2. Autoencoder reconstructions (input vs output)
3. Latent space visualization (t-SNE with class labels)
4. Confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


# Class names and colors
CLASS_NAMES = ['noise', 'upward', 'downward']
CLASS_COLORS = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red


def plot_raw_samples(
    dataset,
    n_samples_per_class: int = 2,
    save_path: str = None,
    figsize: tuple = (16, 12)
):
    """
    Plot raw synthetic samples for each class.

    Shows all 3 channels for each sample to visualize the distinct patterns.
    """
    n_classes = 3

    # Collect samples by class
    samples_by_class = {i: [] for i in range(n_classes)}

    # Get samples from the base dataset (unbatched)
    base_ds = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset

    for idx in range(len(base_ds)):
        label = int(base_ds.labels[idx])
        if len(samples_by_class[label]) < n_samples_per_class:
            samples_by_class[label].append(base_ds.data[idx])

        # Check if we have enough samples
        if all(len(v) >= n_samples_per_class for v in samples_by_class.values()):
            break

    # Create figure
    fig, axes = plt.subplots(
        n_classes * n_samples_per_class, 3,
        figsize=figsize
    )

    fig.suptitle('Raw Synthetic Data Samples\n(Each row = 1 sample, columns = 3 channels)',
                 fontsize=14, fontweight='bold')

    row_idx = 0
    for class_idx in range(n_classes):
        for sample_idx, sample in enumerate(samples_by_class[class_idx]):
            for ch in range(3):
                ax = axes[row_idx, ch]

                # Plot the channel data as an image
                im = ax.imshow(
                    sample[ch],
                    aspect='auto',
                    cmap='viridis',
                    vmin=0, vmax=1
                )

                # Labels
                if ch == 0:
                    ax.set_ylabel(f'{CLASS_NAMES[class_idx]}\n(sample {sample_idx+1})',
                                  fontsize=10, fontweight='bold',
                                  color=CLASS_COLORS[class_idx])
                if row_idx == 0:
                    ax.set_title(f'Channel {ch+1}', fontsize=11)

                ax.set_xlabel('Temporal (width)' if row_idx == len(axes)-1 else '')
                if ch == 0:
                    ax.set_ylabel(ax.get_ylabel() + '\nDepth', fontsize=9)

                # Add colorbar for last column
                if ch == 2:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            row_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_reconstructions(
    model,
    dataloader,
    device: str = 'cpu',
    n_samples: int = 6,
    save_path: str = None,
    figsize: tuple = (16, 14)
):
    """
    Plot autoencoder input vs reconstruction side by side.

    Shows original and reconstructed data for samples from each class.
    """
    import torch

    model.eval()

    # Collect samples and reconstructions by class
    originals = {i: [] for i in range(3)}
    reconstructions = {i: [] for i in range(3)}
    samples_per_class = n_samples // 3

    with torch.no_grad():
        for batch in dataloader:
            data = batch['tokens'].to(device)
            labels = batch['labels']

            # Get reconstructions
            recon, _ = model(data)

            # Convert soft labels to hard
            if labels.dim() > 1:
                hard_labels = labels.argmax(dim=1)
            else:
                hard_labels = labels

            # Collect samples
            for i in range(len(data)):
                label = int(hard_labels[i])
                if len(originals[label]) < samples_per_class:
                    originals[label].append(data[i].cpu().numpy())
                    reconstructions[label].append(recon[i].cpu().numpy())

            # Check if done
            if all(len(v) >= samples_per_class for v in originals.values()):
                break

    # Create figure: rows = samples, cols = original ch0, recon ch0, diff ch0
    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)

    fig.suptitle('Autoencoder Reconstructions\n(Channel 0 shown: Original | Reconstruction | Difference | Error Map)',
                 fontsize=14, fontweight='bold')

    row = 0
    for class_idx in range(3):
        for sample_idx in range(len(originals[class_idx])):
            orig = originals[class_idx][sample_idx][0]  # Channel 0
            recon = reconstructions[class_idx][sample_idx][0]
            diff = orig - recon
            error = np.abs(diff)

            # Original
            ax = axes[row, 0]
            im = ax.imshow(orig, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            if row == 0:
                ax.set_title('Original', fontsize=11)
            ax.set_ylabel(f'{CLASS_NAMES[class_idx]}', fontsize=10,
                         fontweight='bold', color=CLASS_COLORS[class_idx])

            # Reconstruction
            ax = axes[row, 1]
            im = ax.imshow(recon, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            if row == 0:
                ax.set_title('Reconstruction', fontsize=11)

            # Difference
            ax = axes[row, 2]
            im = ax.imshow(diff, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
            if row == 0:
                ax.set_title('Difference', fontsize=11)

            # Error magnitude
            ax = axes[row, 3]
            im = ax.imshow(error, aspect='auto', cmap='hot', vmin=0, vmax=0.3)
            if row == 0:
                ax.set_title('|Error|', fontsize=11)

            # Add MSE annotation
            mse = np.mean(error**2)
            ax.text(1.05, 0.5, f'MSE:\n{mse:.4f}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='center')

            row += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_latent_space(
    embeddings: dict,
    save_path: str = None,
    figsize: tuple = (14, 5),
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Plot t-SNE visualization of the latent space embeddings.

    Shows train, val, and test sets side by side with class coloring.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Latent Space Visualization (t-SNE)', fontsize=14, fontweight='bold')

    cmap = ListedColormap(CLASS_COLORS)

    for ax, (split_name, X_key, y_key) in zip(
        axes,
        [('Train', 'X_train', 'y_train'),
         ('Validation', 'X_val', 'y_val'),
         ('Test', 'X_test', 'y_test')]
    ):
        X = embeddings[X_key]
        y = embeddings[y_key]

        # t-SNE projection
        if len(X) > 2:  # Need at least 3 samples
            # Adjust perplexity if needed
            perp = min(perplexity, len(X) - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
            X_2d = tsne.fit_transform(X)

            # Scatter plot
            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=y, cmap=cmap,
                alpha=0.7, edgecolors='white', linewidths=0.5,
                s=50
            )

            ax.set_title(f'{split_name} Set (n={len(X)})', fontsize=11)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

    # Add legend
    handles = [plt.scatter([], [], c=CLASS_COLORS[i], label=CLASS_NAMES[i], s=50)
               for i in range(3)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.95))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    figsize: tuple = (8, 7),
    normalize: str = None
):
    """
    Plot confusion matrix with detailed annotations.

    Args:
        normalize: None, 'true' (rows), 'pred' (cols), or 'all'
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)

    # Create display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

    # Add per-class metrics
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_class_distribution(
    embeddings: dict,
    save_path: str = None,
    figsize: tuple = (12, 4)
):
    """
    Plot class distribution across train/val/test splits.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Class Distribution Across Splits', fontsize=14, fontweight='bold')

    for ax, (split_name, y_key) in zip(
        axes,
        [('Train', 'y_train'), ('Validation', 'y_val'), ('Test', 'y_test')]
    ):
        y = embeddings[y_key]

        # Count classes
        unique, counts = np.unique(y, return_counts=True)

        bars = ax.bar(
            [CLASS_NAMES[int(u)] for u in unique],
            counts,
            color=[CLASS_COLORS[int(u)] for u in unique],
            edgecolor='black'
        )

        ax.set_title(f'{split_name} (n={len(y)})', fontsize=11)
        ax.set_ylabel('Count')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)

        ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_training_curves(
    history: dict,
    save_path: str = None,
    figsize: tuple = (10, 4)
):
    """
    Plot training and validation loss curves.
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Autoencoder Training Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add final values annotation
    ax.text(0.98, 0.95,
            f'Final Train: {history["train_loss"][-1]:.4f}\nFinal Val: {history["val_loss"][-1]:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def generate_all_plots(
    train_dataset,
    model,
    train_loader,
    embeddings: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: dict,
    output_dir: str,
    device: str = 'cpu'
):
    """
    Generate all validation plots and save to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots to: {output_dir}")
    print("-" * 50)

    # 1. Raw data samples
    plot_raw_samples(
        train_dataset,
        n_samples_per_class=2,
        save_path=output_dir / '01_raw_samples.png'
    )

    # 2. Reconstructions
    plot_reconstructions(
        model, train_loader, device=device,
        n_samples=6,
        save_path=output_dir / '02_reconstructions.png'
    )

    # 3. Training curves
    plot_training_curves(
        history,
        save_path=output_dir / '03_training_curves.png'
    )

    # 4. Class distribution
    plot_class_distribution(
        embeddings,
        save_path=output_dir / '04_class_distribution.png'
    )

    # 5. Latent space t-SNE
    plot_latent_space(
        embeddings,
        save_path=output_dir / '05_latent_space_tsne.png'
    )

    # 6. Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=output_dir / '06_confusion_matrix.png'
    )

    print("-" * 50)
    print(f"All plots saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    # Quick test of visualizations
    print("Testing visualization functions...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tests.pipeline_validation.synthetic_dataset import create_synthetic_splits

    train_ds, val_ds, test_ds = create_synthetic_splits(
        n_train=100, n_val=50, n_test=50, batch_size=25
    )

    # Test raw samples plot
    fig = plot_raw_samples(train_ds, n_samples_per_class=1)
    plt.show()

    print("Visualization test complete!")