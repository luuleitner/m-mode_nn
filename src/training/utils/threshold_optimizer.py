"""
Threshold Optimization for Classification.

Utilities for finding optimal classification thresholds based on
precision-recall trade-offs. Particularly useful for imbalanced
datasets where the default 0.5 threshold is suboptimal.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_recall_curve, f1_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str = 'f1',
    target_value: Optional[float] = None,
    noise_class: int = 0
) -> Dict:
    """
    Find optimal threshold for intention detection (binary: noise vs intention).

    Args:
        y_true: Ground truth labels (N,) - multi-class
        y_proba: Predicted probabilities (N, num_classes)
        target_metric: Optimization target:
            - 'f1': Maximize F1 score
            - 'precision': Find threshold achieving target precision
            - 'recall': Find threshold achieving target recall
            - 'balanced': Balance precision and recall
        target_value: Target value for precision/recall (e.g., 0.95)
        noise_class: Index of noise class (default: 0)

    Returns:
        Dict with:
            - threshold: Optimal threshold
            - precision: Precision at optimal threshold
            - recall: Recall at optimal threshold
            - f1: F1 at optimal threshold
            - all_thresholds: Array of all thresholds evaluated
            - all_precisions: Precision at each threshold
            - all_recalls: Recall at each threshold
    """
    # Convert to binary: noise (0) vs intention (1)
    y_true_binary = (y_true != noise_class).astype(int)

    # P(intention) = 1 - P(noise)
    p_intention = 1.0 - y_proba[:, noise_class]

    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, p_intention)

    # Remove last element (precision=1, recall=0, no threshold)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    result = {
        'all_thresholds': thresholds,
        'all_precisions': precisions,
        'all_recalls': recalls
    }

    if target_metric == 'f1':
        # Maximize F1
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        result['all_f1'] = f1_scores

    elif target_metric == 'precision':
        # Find threshold achieving target precision with max recall
        if target_value is None:
            target_value = 0.9
        valid_mask = precisions >= target_value
        if valid_mask.any():
            # Among valid, pick highest recall
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        else:
            # Fallback to highest precision
            best_idx = np.argmax(precisions)

    elif target_metric == 'recall':
        # Find threshold achieving target recall with max precision
        if target_value is None:
            target_value = 0.95
        valid_mask = recalls >= target_value
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        else:
            best_idx = np.argmax(recalls)

    elif target_metric == 'balanced':
        # Minimize |precision - recall|
        balance_score = np.abs(precisions - recalls)
        best_idx = np.argmin(balance_score)

    else:
        raise ValueError(f"Unknown target_metric: {target_metric}")

    result['threshold'] = float(thresholds[best_idx])
    result['precision'] = float(precisions[best_idx])
    result['recall'] = float(recalls[best_idx])
    result['f1'] = float(2 * precisions[best_idx] * recalls[best_idx] /
                         (precisions[best_idx] + recalls[best_idx] + 1e-10))
    result['target_metric'] = target_metric
    result['target_value'] = target_value

    return result


def apply_threshold(
    y_proba: np.ndarray,
    threshold: float,
    noise_class: int = 0
) -> np.ndarray:
    """
    Apply threshold to convert probabilities to predictions.

    If P(intention) >= threshold, predict the most likely intention class.
    Otherwise, predict noise.

    Args:
        y_proba: Predicted probabilities (N, num_classes)
        threshold: Detection threshold for P(intention)
        noise_class: Index of noise class

    Returns:
        Predicted class labels (N,)
    """
    n_samples, n_classes = y_proba.shape

    # P(intention) = 1 - P(noise)
    p_intention = 1.0 - y_proba[:, noise_class]

    # Start with noise predictions
    predictions = np.full(n_samples, noise_class, dtype=int)

    # For samples above threshold, pick most likely intention class
    above_threshold = p_intention >= threshold

    if above_threshold.any():
        # Mask out noise class and find argmax among intention classes
        intention_proba = y_proba[above_threshold].copy()
        intention_proba[:, noise_class] = -np.inf  # exclude noise
        predictions[above_threshold] = intention_proba.argmax(axis=1)

    return predictions


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: Optional[float] = None,
    noise_class: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot threshold analysis with precision-recall trade-off.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        optimal_threshold: Threshold to highlight (optional)
        noise_class: Index of noise class
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get threshold data
    result = find_optimal_threshold(
        y_true, y_proba, target_metric='f1', noise_class=noise_class
    )

    thresholds = result['all_thresholds']
    precisions = result['all_precisions']
    recalls = result['all_recalls']
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Precision/Recall/F1 vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax1.plot(thresholds, f1_scores, 'g-', label='F1', linewidth=2)

    if optimal_threshold is not None:
        ax1.axvline(x=optimal_threshold, color='k', linestyle='--',
                    label=f'Threshold={optimal_threshold:.3f}')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Detection Threshold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # Plot 2: Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(recalls, precisions, 'b-', linewidth=2)

    if optimal_threshold is not None:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax2.scatter([recalls[idx]], [precisions[idx]], color='red', s=100, zorder=5,
                    label=f'Operating point (τ={optimal_threshold:.3f})')

    # Mark key threshold points
    for tau in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds - tau))
        ax2.scatter([recalls[idx]], [precisions[idx]], color='gray', s=50, alpha=0.7)
        ax2.annotate(f'τ={tau}', (recalls[idx], precisions[idx]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve (Intention Detection)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    noise_class: int = 0,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute full classification metrics at a specific threshold.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        threshold: Detection threshold
        noise_class: Index of noise class
        class_names: Names of classes

    Returns:
        Dict with metrics at the specified threshold
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report
    )

    y_pred = apply_threshold(y_proba, threshold, noise_class)

    # Binary detection metrics
    y_true_binary = (y_true != noise_class).astype(int)
    y_pred_binary = (y_pred != noise_class).astype(int)

    metrics = {
        'threshold': threshold,
        # Multi-class metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        # Binary detection metrics
        'detection_precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'detection_recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'detection_f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_per_class.tolist()

    # Minority F1
    minority_indices = [i for i in range(len(f1_per_class)) if i != noise_class]
    if minority_indices:
        metrics['minority_f1'] = np.mean([f1_per_class[i] for i in minority_indices])

    return metrics