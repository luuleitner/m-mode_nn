"""
Classification Metrics - Evaluation utilities for classification tasks.

Provides functions for computing metrics and generating visualizations
for multi-class classification on embeddings.

Includes extended metrics for imbalanced/anomaly detection problems:
- MCC (Matthews Correlation Coefficient)
- G-mean (Geometric mean of per-class recalls)
- Balanced accuracy
- Intention detection metrics (binary: noise vs intention)
- Average Precision (PR-AUC) per class
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_proba: Predicted probabilities (N, num_classes), optional
        class_names: List of class names for display

    Returns:
        Dictionary with all computed metrics
    """
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'num_samples': len(y_true)
    }

    # ROC-AUC if probabilities provided
    if y_proba is not None:
        roc_auc_per_class = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                roc_auc_per_class.append(
                    auc(*roc_curve(y_true_binary, y_proba[:, i])[:2])
                )
            else:
                roc_auc_per_class.append(np.nan)

        metrics['roc_auc_per_class'] = roc_auc_per_class
        metrics['roc_auc_macro'] = np.nanmean(roc_auc_per_class)

    return metrics


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    noise_class: int = 0
) -> Dict:
    """
    Compute extended metrics for imbalanced/anomaly detection problems.

    Assumes class 0 is the majority "noise" class and classes 1+ are
    minority "intention" classes to detect.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_proba: Predicted probabilities (N, num_classes), optional
        class_names: List of class names for display
        noise_class: Index of the noise/majority class (default: 0)

    Returns:
        Dictionary with anomaly-focused metrics:
        - mcc: Matthews Correlation Coefficient
        - balanced_accuracy: Macro-averaged recall
        - g_mean: Geometric mean of per-class recalls
        - minority_f1: Average F1 of minority classes
        - ap_per_class: Average Precision per class
        - ap_macro: Macro-averaged AP
        - intention_detection: Binary detection metrics (noise vs intention)
    """
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    metrics = {}

    # Matthews Correlation Coefficient - handles imbalance well
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Balanced accuracy (macro-averaged recall)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Per-class recall for G-mean calculation
    recalls = []
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() > 0:
            recall_i = (y_pred[mask] == i).sum() / mask.sum()
            recalls.append(recall_i)
        else:
            recalls.append(0.0)
    metrics['recall_per_class'] = recalls

    # G-mean: geometric mean of per-class recalls
    # Penalizes models that sacrifice minority class performance
    if all(r > 0 for r in recalls):
        metrics['g_mean'] = np.power(np.prod(recalls), 1.0 / len(recalls))
    else:
        metrics['g_mean'] = 0.0

    # Minority class F1 (average F1 of non-noise classes)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    minority_indices = [i for i in range(num_classes) if i != noise_class]
    if minority_indices:
        metrics['minority_f1'] = np.mean([f1_per_class[i] for i in minority_indices])
        metrics['f1_per_class'] = f1_per_class.tolist()
    else:
        metrics['minority_f1'] = 0.0

    # Average Precision (PR-AUC) per class - better than ROC-AUC for imbalanced
    if y_proba is not None:
        ap_per_class = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            if y_true_binary.sum() > 0 and y_true_binary.sum() < len(y_true_binary):
                ap = average_precision_score(y_true_binary, y_proba[:, i])
                ap_per_class.append(ap)
            else:
                ap_per_class.append(np.nan)

        metrics['ap_per_class'] = ap_per_class
        metrics['ap_macro'] = np.nanmean(ap_per_class)

        # Minority AP (average AP of non-noise classes)
        minority_ap = [ap_per_class[i] for i in minority_indices if not np.isnan(ap_per_class[i])]
        metrics['minority_ap'] = np.mean(minority_ap) if minority_ap else 0.0

    return metrics


def compute_intention_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    noise_class: int = 0
) -> Dict:
    """
    Compute binary intention detection metrics (noise vs any intention).

    Useful when the primary goal is detecting that an intention occurred,
    with direction classification being secondary.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_proba: Predicted probabilities (N, num_classes), optional
        noise_class: Index of the noise class (default: 0)

    Returns:
        Dictionary with binary detection metrics:
        - detection_precision: Precision for detecting intentions
        - detection_recall: Recall for detecting intentions
        - detection_f1: F1 score for intention detection
        - detection_ap: Average Precision for intention detection
        - confusion_binary: 2x2 confusion matrix [noise vs intention]
    """
    # Convert to binary: noise (0) vs intention (1)
    y_true_binary = (y_true != noise_class).astype(int)
    y_pred_binary = (y_pred != noise_class).astype(int)

    metrics = {}

    # Binary confusion matrix
    tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
    tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()

    metrics['confusion_binary'] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    # Precision, recall, F1 for intention detection
    metrics['detection_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['detection_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if metrics['detection_precision'] + metrics['detection_recall'] > 0:
        metrics['detection_f1'] = (
            2 * metrics['detection_precision'] * metrics['detection_recall'] /
            (metrics['detection_precision'] + metrics['detection_recall'])
        )
    else:
        metrics['detection_f1'] = 0.0

    # Average Precision for intention detection
    if y_proba is not None:
        # P(intention) = 1 - P(noise) = sum of all non-noise probabilities
        p_intention = 1.0 - y_proba[:, noise_class]
        metrics['detection_ap'] = average_precision_score(y_true_binary, p_intention)
    else:
        metrics['detection_ap'] = None

    # Additional stats
    metrics['n_true_intentions'] = int(y_true_binary.sum())
    metrics['n_pred_intentions'] = int(y_pred_binary.sum())
    metrics['n_total'] = len(y_true)

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate and print sklearn classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Classification report string
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure (optional)
        normalize: Whether to normalize by true labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if normalize:
                text = f'{cm_display[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification (one-vs-rest).

    Args:
        y_true: Ground truth labels (N,)
        y_proba: Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == i).astype(int)

        if len(np.unique(y_true_binary)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class classification.

    Args:
        y_true: Ground truth labels (N,)
        y_proba: Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == i).astype(int)

        if len(np.unique(y_true_binary)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
        ap = average_precision_score(y_true_binary, y_proba[:, i])

        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AP = {ap:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot t-SNE visualization of embeddings colored by class.

    Args:
        embeddings: Embedding vectors (N, embedding_dim)
        labels: Class labels (N,)
        class_names: List of class names
        save_path: Path to save figure (optional)
        perplexity: t-SNE perplexity parameter
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE

    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'{name} (n={mask.sum()})',
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Embedding Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig