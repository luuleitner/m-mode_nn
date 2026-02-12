"""
Unified loss functions for training pipeline.

Provides consistent loss computation for:
- Pure classification (CNN classifier)
- Joint reconstruction + classification (UNet with classification head)

Key principles:
- Reconstruction loss is NEVER class-weighted (uniform quality across classes)
- Classification loss CAN be class-weighted (to handle imbalance)
- Validation always uses unweighted loss for fair comparison
"""

import torch
import torch.nn.functional as F


def compute_classification_loss(logits, soft_labels, hard_labels, class_weights=None, weighted=True):
    """
    Unified loss computation for classification.

    Args:
        logits: Model outputs (B, num_classes)
        soft_labels: Soft label distributions (B, num_classes) or None
        hard_labels: Hard labels (B,) as class indices
        class_weights: Per-class weights tensor or None
        weighted: If True, apply class_weights (for training). If False, ignore weights (for val/test)

    Returns:
        loss: Scalar loss value

    Loss priority:
        1. If soft_labels provided: soft cross-entropy (with optional class weighting)
        2. If wewehwwighted=True and class_weights provided: weighted cross-entropy
        3. Otherwise: standard unweighted cross-entropy
    """
    if soft_labels is not None:
        # Soft cross-entropy: -sum(p * log(q))
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_labels * log_probs).sum(dim=-1)
        if weighted and class_weights is not None:
            # Weight each sample by its hard label's class weight (same as F.cross_entropy(weight=...))
            sample_weights = class_weights[hard_labels]
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
        else:
            loss = per_sample_loss.mean()
    elif weighted and class_weights is not None:
        # Weighted cross-entropy for training with imbalanced classes
        loss = F.cross_entropy(logits, hard_labels, weight=class_weights)
    else:
        # Standard unweighted cross-entropy for fair evaluation
        loss = F.cross_entropy(logits, hard_labels)

    return loss


def compute_loss(logits, soft_labels, hard_labels, class_weights=None, weighted=True):
    """Alias for compute_classification_loss (backward compatibility)."""
    return compute_classification_loss(logits, soft_labels, hard_labels, class_weights, weighted)


def compute_train_loss(logits, soft_labels, hard_labels, class_weights=None):
    """Training loss: applies class weights for imbalance handling."""
    return compute_classification_loss(logits, soft_labels, hard_labels, class_weights, weighted=True)


def compute_eval_loss(logits, soft_labels, hard_labels):
    """Evaluation loss (val/test): unweighted for fair performance measurement."""
    return compute_classification_loss(logits, soft_labels, hard_labels, class_weights=None, weighted=False)


# =============================================================================
# Joint Reconstruction + Classification Loss (for UNet with classification head)
# =============================================================================

def compute_reconstruction_loss(reconstruction, target, mse_weight=0.5, l1_weight=0.5):
    """
    Reconstruction loss (MSE + L1). NEVER class-weighted.

    Args:
        reconstruction: Reconstructed input (B, C, H, W)
        target: Original input (B, C, H, W)
        mse_weight: Weight for MSE loss component
        l1_weight: Weight for L1 loss component

    Returns:
        tuple: (total_loss, loss_dict with mse and l1 components)
    """
    mse_loss = F.mse_loss(reconstruction, target)
    l1_loss = F.l1_loss(reconstruction, target)
    total = mse_weight * mse_loss + l1_weight * l1_loss
    return total, {'mse': mse_loss.item(), 'l1': l1_loss.item()}


def compute_supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Supervised contrastive loss (Khosla et al. 2020) using in-batch pairs.

    For each anchor, all same-class samples in the batch are positives,
    all different-class samples are negatives. No special sampling needed.

    Args:
        embeddings: (B, D) embedding vectors
        labels: (B,) integer class labels
        temperature: Scaling factor for cosine similarities

    Returns:
        Scalar loss
    """
    # L2-normalize to unit sphere — cosine similarity via dot product
    embeddings = F.normalize(embeddings, dim=1)

    # Pairwise cosine similarity [B, B]
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    # Mask: same-class pairs (excluding self)
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()).float()
    positive_mask.fill_diagonal_(0)  # exclude self-pairs

    # Number of positives per anchor
    num_positives = positive_mask.sum(dim=1)

    # Skip anchors with no positives (e.g., singleton class in batch)
    valid = num_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Log-softmax over all non-self entries (numerically stable)
    # Mask out self-similarity by setting diagonal to -inf
    logits_mask = torch.ones_like(sim_matrix)
    logits_mask.fill_diagonal_(0)
    log_prob = sim_matrix - torch.logsumexp(
        sim_matrix * logits_mask + (1 - logits_mask) * (-1e9), dim=1, keepdim=True
    )

    # Mean log-prob over positive pairs, averaged over valid anchors
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
    loss = -mean_log_prob[valid].mean()

    return loss


def compute_joint_loss(reconstruction, target, logits, soft_labels, hard_labels,
                       class_weights=None, loss_weights=None, is_training=True):
    """
    Joint reconstruction + classification loss.

    Key design:
    - Reconstruction loss is NEVER class-weighted (uniform quality)
    - Classification loss is weighted during training (handles imbalance)

    Args:
        reconstruction: Reconstructed input (B, C, H, W)
        target: Original input (B, C, H, W)
        logits: Classification logits (B, num_classes) or None
        soft_labels: Soft label distributions (B, num_classes) or None
        hard_labels: Hard labels (B,)
        class_weights: Per-class weights tensor for classification or None
        loss_weights: Dict with mse_weight, l1_weight, cls_weight
        is_training: If True, apply class_weights to classification

    Returns:
        tuple: (total_loss, loss_dict)
    """
    if loss_weights is None:
        loss_weights = {}

    mse_weight = loss_weights.get('mse_weight', 0.3)
    l1_weight = loss_weights.get('l1_weight', 0.3)
    cls_weight = loss_weights.get('cls_weight', 0.4)

    # Reconstruction loss (never class-weighted)
    recon_loss, recon_dict = compute_reconstruction_loss(
        reconstruction, target, mse_weight, l1_weight
    )

    loss_dict = {
        'recon_loss': recon_loss.item(),
        'mse': recon_dict['mse'],
        'l1': recon_dict['l1']
    }

    total_loss = recon_loss

    # Classification loss (class-weighted for training only)
    if logits is not None and hard_labels is not None and cls_weight > 0:
        weights = class_weights if is_training else None
        cls_loss = compute_classification_loss(
            logits, soft_labels, hard_labels, weights, weighted=is_training
        )
        total_loss = total_loss + cls_weight * cls_loss
        loss_dict['cls_loss'] = cls_loss.item()

        # Track accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == hard_labels).float().mean().item()
            loss_dict['accuracy'] = accuracy

    loss_dict['total'] = total_loss.item()
    return total_loss, loss_dict


def compute_joint_train_loss(reconstruction, target, logits, soft_labels, hard_labels,
                             class_weights=None, loss_weights=None):
    """Joint training loss with class-weighted classification."""
    return compute_joint_loss(
        reconstruction, target, logits, soft_labels, hard_labels,
        class_weights, loss_weights, is_training=True
    )


def compute_joint_eval_loss(reconstruction, target, logits, soft_labels, hard_labels,
                            loss_weights=None):
    """Joint evaluation loss with unweighted classification."""
    return compute_joint_loss(
        reconstruction, target, logits, soft_labels, hard_labels,
        class_weights=None, loss_weights=loss_weights, is_training=False
    )
