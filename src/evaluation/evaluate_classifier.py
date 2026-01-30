"""
Classification Head Evaluation

Evaluates the CNN autoencoder's classification head on the test set.
Computes accuracy, precision, recall, F1, and confusion matrix.

Usage:
    python -m src.evaluation.evaluate_classifier --config config/config.yaml --checkpoint path/to/model.pth
    python -m src.evaluation.evaluate_classifier --config config/config.yaml --checkpoint path/to/model.pth --save
"""

import os
import sys
import argparse
import pickle
import yaml
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config
import utils.logging_config as logconf

logger = logconf.get_logger("EVAL_CLS")

# Load class names from centralized config
_label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
with open(_label_config_path) as _f:
    _label_config = yaml.safe_load(_f)
_classes_config = _label_config.get('classes', {})
CLASS_NAMES = [_classes_config['names'].get(i, f'class_{i}') for i in range(_classes_config.get('num_classes', 3))]


def create_model(config):
    """Create model based on config."""
    model_type = config.ml.model.type

    # Check if classification was enabled during training
    loss_weights = config.get_loss_weights()
    classification_weight = loss_weights.get('classification_weight', 0)

    # Load num_classes from label config if classification is enabled
    if classification_weight > 0:
        label_config_path = os.path.join(project_root, 'preprocessing/label_logic/label_config.yaml')
        with open(label_config_path) as f:
            label_config = yaml.safe_load(f)
        num_classes = label_config['classes']['num_classes']
    else:
        num_classes = 0

    # Input dimensions after transpose: [B, C, Pulses, Depth]
    input_pulses = config.preprocess.tokenization.window
    input_depth = 130

    if model_type == "UNetAutoencoder":
        from src.models.unet_ae import UNetAutoencoder
        model = UNetAutoencoder(
            in_channels=3,
            input_height=input_pulses,
            input_width=input_depth,
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=True,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info("Model weights loaded successfully")

    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")

    return model


def load_test_dataset(config):
    """Load test dataset from pickle."""
    pickle_path = config.get_train_data_root()
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    logger.info(f"Loading test dataset from: {test_path}")
    with open(test_path, 'rb') as f:
        test_ds = pickle.load(f)

    logger.info(f"Test samples: {len(test_ds)}")
    return test_ds


def compute_classification_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive classification metrics.

    Returns:
        dict with accuracy, per-class metrics, and confusion matrix
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # Macro and weighted averages
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Classification report string
    report = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            name: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm,
        'classification_report': report
    }

    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True):
    """Plot confusion matrix as heatmap."""
    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_plot = np.nan_to_num(cm_plot)  # Handle division by zero
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_plot = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm_plot.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f'{cm_plot[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if cm_plot[i, j] > thresh else 'black',
                    fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")

    return fig


def plot_roc_curves(y_true, y_probs, class_names=None, save_path=None):
    """Plot ROC curves for each class (one-vs-rest)."""
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#808080', '#2ecc71', '#e74c3c']  # gray, green, red

    # Compute ROC curve and AUC for each class
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    # Compute micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curves saved: {save_path}")

    return fig


def plot_precision_recall_curves(y_true, y_probs, class_names=None, save_path=None):
    """Plot Precision-Recall curves for each class."""
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#808080', '#2ecc71', '#e74c3c']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AP = {ap:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"PR curves saved: {save_path}")

    return fig


def plot_probability_distributions(y_true, y_probs, class_names=None, save_path=None):
    """Plot predicted probability distributions for each true class."""
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))
    colors = ['#808080', '#2ecc71', '#e74c3c']

    for true_cls in range(n_classes):
        ax = axes[true_cls]
        mask = y_true == true_cls

        if mask.sum() == 0:
            ax.set_title(f'True: {class_names[true_cls]} (n=0)')
            continue

        # Plot probability distribution for each predicted class
        for pred_cls in range(n_classes):
            probs = y_probs[mask, pred_cls]
            ax.hist(probs, bins=30, alpha=0.5, color=colors[pred_cls],
                    label=f'P({class_names[pred_cls]})', density=True)

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'True: {class_names[true_cls]} (n={mask.sum()})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([0, 1])

    plt.suptitle('Probability Distributions by True Class')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Probability distributions saved: {save_path}")

    return fig


def plot_class_distribution(y_true, y_pred, class_names=None, save_path=None):
    """Compare predicted vs true class distributions."""
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    true_counts = np.bincount(y_true, minlength=n_classes)
    pred_counts = np.bincount(y_pred, minlength=n_classes)

    x = np.arange(n_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', color='#3498db')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='#e74c3c')

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('True vs Predicted Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Class distribution saved: {save_path}")

    return fig


def plot_per_class_metrics(metrics, save_path=None):
    """Plot per-class precision, recall, F1 as bar chart."""
    class_names = list(metrics['per_class'].keys())
    x = np.arange(len(class_names))
    width = 0.25

    precision = [metrics['per_class'][c]['precision'] for c in class_names]
    recall = [metrics['per_class'][c]['recall'] for c in class_names]
    f1 = [metrics['per_class'][c]['f1'] for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved: {save_path}")

    return fig


def evaluate_classifier(model, test_loader, device):
    """
    Run classification evaluation on test set.

    Returns:
        tuple: (all_preds, all_labels, all_probs)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Handle batch format
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                labels = batch['labels']
            elif isinstance(batch, (list, tuple)):
                data = batch[0].to(device)
                labels = batch[1] if len(batch) > 1 else None
            else:
                data = batch.to(device)
                labels = None

            # Transpose H/W: [B, C, Depth, Pulses] → [B, C, Pulses, Depth]
            data = data.permute(0, 1, 3, 2)

            # Forward pass
            outputs = model(data)
            if len(outputs) == 3:
                _, _, logits = outputs
            else:
                raise ValueError("Model does not have classification head enabled")

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Convert labels
            if labels is not None:
                if labels.dim() > 1 and labels.shape[-1] > 1:
                    hard_labels = labels.argmax(dim=-1)
                elif labels.dim() > 1:
                    hard_labels = labels.squeeze(-1)
                else:
                    hard_labels = labels
                all_labels.append(hard_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels) if all_labels else None

    return all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description='Evaluate classification head')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', '-ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: same as checkpoint, creates evaluation subfolder)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, create_dirs=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model and load weights
    model = create_model(config)
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)

    if model.classifier is None:
        logger.error("Model does not have a classification head. Enable classification_weight > 0 in config.")
        return 1

    # Load test dataset
    test_ds = load_test_dataset(config)

    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=4)

    # Run evaluation
    logger.info("Running classification evaluation...")
    preds, labels, probs = evaluate_classifier(model, test_loader, device)

    if labels is None:
        logger.error("No labels found in test dataset")
        return 1

    # Compute metrics
    metrics = compute_classification_metrics(labels, preds, CLASS_NAMES)

    # Print results
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
    print("\n" + metrics['classification_report'])
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("=" * 60)

    # Save results - always save to evaluation folder in training path
    # Determine output directory: use training output path from config or checkpoint directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to checkpoint directory (typically the training output folder)
        output_dir = os.path.dirname(args.checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(output_dir, f'evaluation_{timestamp}')
    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Saving evaluation results to: {eval_dir}")

    # 1. Confusion matrix (normalized with counts)
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CLASS_NAMES,
        save_path=os.path.join(eval_dir, 'confusion_matrix.png')
    )

    # 2. Per-class metrics bar chart
    plot_per_class_metrics(
        metrics,
        save_path=os.path.join(eval_dir, 'per_class_metrics.png')
    )

    # 3. ROC curves (one-vs-rest)
    plot_roc_curves(
        labels, probs, CLASS_NAMES,
        save_path=os.path.join(eval_dir, 'roc_curves.png')
    )

    # 4. Precision-Recall curves
    plot_precision_recall_curves(
        labels, probs, CLASS_NAMES,
        save_path=os.path.join(eval_dir, 'precision_recall_curves.png')
    )

    # 5. Probability distributions by true class
    plot_probability_distributions(
        labels, probs, CLASS_NAMES,
        save_path=os.path.join(eval_dir, 'probability_distributions.png')
    )

    # 6. True vs Predicted class distribution
    plot_class_distribution(
        labels, preds, CLASS_NAMES,
        save_path=os.path.join(eval_dir, 'class_distribution.png')
    )

    # Save metrics as text report
    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write("CLASSIFICATION EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(labels)}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n")
        f.write(f"Macro F1-Score:   {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:      {metrics['weighted_f1']:.4f}\n\n")
        f.write(metrics['classification_report'])
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))

        # Add AUC scores
        n_classes = len(CLASS_NAMES)
        y_true_bin = label_binarize(labels, classes=range(n_classes))
        f.write("\n\nROC AUC Scores:\n")
        for i, name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            f.write(f"  {name}: {roc_auc:.4f}\n")

        f.write("\nAverage Precision Scores:\n")
        for i, name in enumerate(CLASS_NAMES):
            ap = average_precision_score(y_true_bin[:, i], probs[:, i])
            f.write(f"  {name}: {ap:.4f}\n")

    # Save raw predictions for further analysis
    np.savez(
        os.path.join(eval_dir, 'predictions.npz'),
        predictions=preds,
        labels=labels,
        probabilities=probs
    )

    logger.info(f"All evaluation results saved to: {eval_dir}")
    print(f"\nEvaluation complete. Results saved to: {eval_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
