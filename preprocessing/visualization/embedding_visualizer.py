"""
Embedding Space Visualizer

Visualize autoencoder embeddings to diagnose classification issues.
Shows whether embeddings are discriminative (classes separable) or not.

Usage:
    python visualization/embedding_visualizer.py --embeddings path/to/embeddings.npz
    python visualization/embedding_visualizer.py --embeddings path/to/embeddings.npz --method umap
    python visualization/embedding_visualizer.py --embeddings path/to/embeddings.npz --save
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CLASS_NAMES = ['noise', 'upward', 'downward']
CLASS_COLORS = ['#808080', '#2ecc71', '#e74c3c']  # gray, green, red


def load_embeddings(embeddings_path):
    """Load embeddings from npz file."""
    data = np.load(embeddings_path)

    result = {}
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'
        if X_key in data and y_key in data:
            result[split] = {
                'X': data[X_key],
                'y': data[y_key].astype(int)
            }

    return result


def reduce_dimensions(X, method='tsne', n_components=2, random_state=42):
    """Reduce embedding dimensions for visualization."""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(X) - 1),
            max_iter=1000  # renamed from n_iter in newer sklearn
        )
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1
            )
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            print("Falling back to t-SNE...")
            return reduce_dimensions(X, method='tsne', n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)


def compute_class_statistics(X, y):
    """Compute per-class statistics for embeddings."""
    stats = {}

    for cls in np.unique(y):
        mask = y == cls
        X_cls = X[mask]
        stats[cls] = {
            'count': mask.sum(),
            'mean': X_cls.mean(axis=0),
            'std': X_cls.std(axis=0).mean(),
            'norm_mean': np.linalg.norm(X_cls, axis=1).mean(),
            'norm_std': np.linalg.norm(X_cls, axis=1).std()
        }

    # Compute inter-class distances
    class_indices = sorted([k for k in stats.keys() if isinstance(k, (int, np.integer))])
    class_means = [stats[cls]['mean'] for cls in class_indices]
    for i, cls_i in enumerate(class_indices):
        for j, cls_j in enumerate(class_indices):
            if i < j:
                dist = np.linalg.norm(class_means[i] - class_means[j])
                stats[f'dist_{cls_i}_{cls_j}'] = dist

    return stats


def compute_separability_score(X, y):
    """
    Compute a simple separability score using Fisher's criterion.
    Higher score = more separable classes.
    """
    classes = np.unique(y)

    # Overall mean
    overall_mean = X.mean(axis=0)

    # Between-class scatter
    Sb = 0
    for cls in classes:
        mask = y == cls
        n_cls = mask.sum()
        mean_cls = X[mask].mean(axis=0)
        diff = mean_cls - overall_mean
        Sb += n_cls * np.outer(diff, diff)

    # Within-class scatter
    Sw = 0
    for cls in classes:
        mask = y == cls
        X_cls = X[mask]
        mean_cls = X_cls.mean(axis=0)
        for x in X_cls:
            diff = x - mean_cls
            Sw += np.outer(diff, diff)

    # Fisher score: trace(Sb) / trace(Sw)
    trace_Sb = np.trace(Sb)
    trace_Sw = np.trace(Sw)

    if trace_Sw > 0:
        fisher_score = trace_Sb / trace_Sw
    else:
        fisher_score = 0

    return fisher_score


def plot_embedding_scatter(X_2d, y, title, ax=None, show_legend=True):
    """Plot 2D scatter of embeddings colored by class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=CLASS_COLORS[cls],
            label=f'{CLASS_NAMES[cls]} (n={mask.sum()})',
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)

    if show_legend:
        ax.legend(loc='best')

    return ax


def plot_class_distributions(X, y, ax=None):
    """Plot distribution of embedding norms per class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    norms = np.linalg.norm(X, axis=1)

    for cls in np.unique(y):
        mask = y == cls
        ax.hist(
            norms[mask],
            bins=50,
            alpha=0.5,
            label=f'{CLASS_NAMES[cls]}',
            color=CLASS_COLORS[cls]
        )

    ax.set_xlabel('Embedding L2 Norm')
    ax.set_ylabel('Count')
    ax.set_title('Embedding Norm Distribution per Class')
    ax.legend()

    return ax


def plot_dimension_distributions(X, y, top_n=6):
    """Plot distributions of top N embedding dimensions per class."""
    # Find dimensions with highest variance
    variances = X.var(axis=0)
    top_dims = np.argsort(variances)[-top_n:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dim in enumerate(top_dims):
        ax = axes[idx]
        for cls in np.unique(y):
            mask = y == cls
            ax.hist(
                X[mask, dim],
                bins=50,
                alpha=0.5,
                label=CLASS_NAMES[cls],
                color=CLASS_COLORS[cls]
            )
        ax.set_title(f'Dimension {dim} (var={variances[dim]:.4f})')
        ax.legend()

    plt.suptitle('Top Variance Embedding Dimensions by Class')
    plt.tight_layout()

    return fig


def create_full_visualization(embeddings_dict, method='tsne', output_dir=None):
    """Create comprehensive embedding visualization."""

    # Combine all splits for overall view
    all_X = np.vstack([embeddings_dict[s]['X'] for s in embeddings_dict])
    all_y = np.hstack([embeddings_dict[s]['y'] for s in embeddings_dict])

    print("=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(all_X)}")
    print(f"Embedding dimension: {all_X.shape[1]}")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(all_y)))}")

    # Compute separability score
    fisher_score = compute_separability_score(all_X, all_y)
    print(f"\nFisher Separability Score: {fisher_score:.4f}")
    if fisher_score < 0.1:
        print("  -> VERY LOW: Classes highly overlapped - embeddings not discriminative")
    elif fisher_score < 0.5:
        print("  -> LOW: Some overlap - classification will be challenging")
    elif fisher_score < 1.0:
        print("  -> MODERATE: Partial separation - reasonable classification possible")
    else:
        print("  -> GOOD: Classes reasonably separated")

    # Compute class statistics
    print("\nPer-class embedding statistics:")
    stats = compute_class_statistics(all_X, all_y)
    for cls in range(3):
        if cls in stats:
            s = stats[cls]
            print(f"  {CLASS_NAMES[cls]}: n={s['count']}, "
                  f"norm={s['norm_mean']:.3f}+/-{s['norm_std']:.3f}, "
                  f"spread={s['std']:.4f}")

    print("\nInter-class centroid distances:")
    for key, val in stats.items():
        if isinstance(key, str) and key.startswith('dist_'):
            cls_i, cls_j = key.replace('dist_', '').split('_')
            print(f"  {CLASS_NAMES[int(cls_i)]} <-> {CLASS_NAMES[int(cls_j)]}: {val:.4f}")

    # Dimensionality reduction
    print(f"\nReducing dimensions with {method.upper()}...")
    X_2d = reduce_dimensions(all_X, method=method)

    # Create visualization
    fig = plt.figure(figsize=(20, 15))

    # Main scatter plot
    ax1 = fig.add_subplot(2, 2, 1)
    plot_embedding_scatter(X_2d, all_y, f'Embedding Space ({method.upper()})', ax=ax1)

    # Per-split scatter
    ax2 = fig.add_subplot(2, 2, 2)
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    for split, data in embeddings_dict.items():
        # Get indices for this split
        start_idx = 0
        for s in embeddings_dict:
            if s == split:
                break
            start_idx += len(embeddings_dict[s]['X'])
        end_idx = start_idx + len(data['X'])

        ax2.scatter(
            X_2d[start_idx:end_idx, 0],
            X_2d[start_idx:end_idx, 1],
            alpha=0.3,
            s=10,
            label=f'{split} (n={len(data["X"])})'
        )
    ax2.set_title('Colored by Split')
    ax2.legend()

    # Norm distributions
    ax3 = fig.add_subplot(2, 2, 3)
    plot_class_distributions(all_X, all_y, ax=ax3)

    # Class centroids
    ax4 = fig.add_subplot(2, 2, 4)
    for cls in np.unique(all_y):
        mask = all_y == cls
        centroid = X_2d[mask].mean(axis=0)
        ax4.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=CLASS_COLORS[cls], alpha=0.3, s=10
        )
        ax4.scatter(
            centroid[0], centroid[1],
            c=CLASS_COLORS[cls], s=200, marker='X',
            edgecolors='black', linewidths=2,
            label=f'{CLASS_NAMES[cls]} centroid'
        )
    ax4.set_title('Class Centroids')
    ax4.legend()

    plt.suptitle(f'Embedding Visualization (Fisher Score: {fisher_score:.4f})', fontsize=14)
    plt.tight_layout()

    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'embedding_visualization_{method}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    # Also create dimension distribution plot
    fig2 = plot_dimension_distributions(all_X, all_y)
    if output_dir:
        output_path2 = os.path.join(output_dir, 'embedding_dimensions.png')
        fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path2}")

    return fig, fisher_score


def main():
    parser = argparse.ArgumentParser(description='Visualize embedding space')
    parser.add_argument(
        '--embeddings', '-e',
        type=str,
        required=True,
        help='Path to embeddings .npz file'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='tsne',
        choices=['tsne', 'umap', 'pca'],
        help='Dimensionality reduction method (default: tsne)'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save plots to file instead of showing'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for plots (default: same as embeddings file)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file not found: {args.embeddings}")
        return 1

    # Load embeddings
    print(f"Loading embeddings from: {args.embeddings}")
    embeddings = load_embeddings(args.embeddings)

    if not embeddings:
        print("Error: No embeddings found in file")
        return 1

    # Set output directory
    output_dir = args.output_dir
    if args.save and output_dir is None:
        output_dir = os.path.dirname(args.embeddings)

    # Create visualization
    fig, score = create_full_visualization(
        embeddings,
        method=args.method,
        output_dir=output_dir if args.save else None
    )

    if not args.save:
        plt.show()

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if score < 0.1:
        print("PROBLEM: Embeddings are NOT discriminative!")
        print("The autoencoder learned reconstruction features, not classification features.")
        print("\nRecommended actions:")
        print("  1. Add classification loss to AE training")
        print("  2. Use supervised contrastive learning")
        print("  3. Switch to end-to-end CNN classifier")
    elif score < 0.5:
        print("WARNING: Embeddings have limited discriminative power.")
        print("\nRecommended actions:")
        print("  1. Try different AE architecture/embedding size")
        print("  2. Add classification head with fine-tuning")
        print("  3. Tune XGBoost hyperparameters aggressively")
    else:
        print("Embeddings show reasonable class separation.")
        print("Classification issues may be due to:")
        print("  1. XGBoost hyperparameters")
        print("  2. Class imbalance handling")
        print("  3. Train/test distribution shift")

    return 0


if __name__ == '__main__':
    exit(main())