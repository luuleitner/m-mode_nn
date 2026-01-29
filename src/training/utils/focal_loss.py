"""
Focal Loss for imbalanced classification.

Focal Loss down-weights well-classified examples and focuses training
on hard, misclassified examples. Particularly effective for imbalanced
datasets where the majority class dominates the standard cross-entropy loss.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the correct class.

    Args:
        alpha: Class weights. Can be:
            - None: No class weighting
            - float: Weight for positive class (binary)
            - Tensor of shape (num_classes,): Per-class weights
            - "balanced": Compute weights from class frequencies
        gamma: Focusing parameter. Higher values focus more on hard examples.
            - gamma=0: Equivalent to cross-entropy
            - gamma=2: Common default, good for moderate imbalance
            - gamma=5: Aggressive focusing for severe imbalance
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Optional label smoothing factor (0.0 to 1.0)

    Example:
        >>> # With class weights for 3 classes (noise=majority, upward/downward=minority)
        >>> class_counts = torch.tensor([10000, 500, 300])
        >>> weights = 1.0 / class_counts.float()
        >>> weights = weights / weights.sum() * len(weights)  # normalize
        >>> criterion = FocalLoss(alpha=weights, gamma=2.0)
        >>>
        >>> logits = model(inputs)  # (batch_size, num_classes)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs (N, C) where C is num_classes
            targets: Ground truth class indices (N,) with values in [0, C-1]

        Returns:
            Focal loss value
        """
        num_classes = logits.size(-1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(logits)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            targets_smooth = None

        # Compute softmax probabilities
        p = F.softmax(logits, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = p[target_class]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    labels: np.ndarray,
    method: str = 'inverse_freq',
    smoothing: float = 0.0
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: Array of class labels
        method: Weighting method:
            - 'inverse_freq': Weight = 1 / class_frequency (normalized)
            - 'inverse_sqrt': Weight = 1 / sqrt(class_frequency)
            - 'effective_samples': Based on effective number of samples
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        Tensor of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)

    if method == 'inverse_freq':
        weights = n_samples / (n_classes * counts)
    elif method == 'inverse_sqrt':
        weights = np.sqrt(n_samples / (n_classes * counts))
    elif method == 'effective_samples':
        # Effective number of samples (Cui et al., 2019)
        beta = (n_samples - 1) / n_samples
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply smoothing: weight = (1 - smoothing) * weight + smoothing
    if smoothing > 0:
        weights = (1 - smoothing) * weights + smoothing

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


class FocalLossWithLabelSmoothing(FocalLoss):
    """
    Focal Loss with built-in label smoothing.

    Combines the benefits of focal loss (focus on hard examples)
    with label smoothing (regularization, prevents overconfidence).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            label_smoothing=smoothing
        )