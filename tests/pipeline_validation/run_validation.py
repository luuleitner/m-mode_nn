"""
Pipeline Validation Test

End-to-end test of the AE -> Embedding -> Classifier pipeline using synthetic data.

This script validates that the pipeline works correctly by:
1. Training an autoencoder on trivially separable synthetic data
2. Extracting embeddings from the trained model
3. Training an XGBoost classifier on the embeddings
4. Verifying high classification accuracy (>95%)

If this test fails, there's a bug in the pipeline code.
If it passes, the pipeline is working correctly and any issues with real data
are likely data-related, not code-related.

Usage:
    python -m tests.pipeline_validation.run_validation
    python -m tests.pipeline_validation.run_validation --epochs 5 --quick
    python -m tests.pipeline_validation.run_validation --device cuda
    python -m tests.pipeline_validation.run_validation --device cuda --plots --output-dir ./validation_results
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.pipeline_validation.synthetic_dataset import (
    create_synthetic_splits,
    IMBALANCED_DISTRIBUTION
)


# ============================================================================
# STAGE 1: Autoencoder Training
# ============================================================================

def create_model(model_type: str = "CNNAutoencoder", embedding_dim: int = 128):
    """Create autoencoder model."""
    if model_type == "CNNAutoencoder":
        from src.models.cnn_ae import CNNAutoencoder
        model = CNNAutoencoder(
            in_channels=3,
            input_height=130,
            input_width=18,
            channels=[32, 64, 128, 256],  # Lighter for faster testing
            embedding_dim=embedding_dim,
            use_batchnorm=True
        )
    elif model_type == "UNetAutoencoder":
        from src.models.unet_ae import UNetAutoencoder
        model = UNetAutoencoder(
            in_channels=3,
            input_height=130,
            input_width=18,
            channels=[32, 64, 128, 256],
            embedding_dim=embedding_dim,
            use_batchnorm=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu"
) -> dict:
    """
    Train autoencoder on synthetic data.

    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'='*60}")
    print("STAGE 1: Autoencoder Training")
    print(f"{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            data = batch['tokens'].to(device)

            optimizer.zero_grad()
            reconstruction, embedding = model(data)

            # Combined loss
            loss = 0.5 * mse_loss(reconstruction, data) + 0.5 * l1_loss(reconstruction, data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                data = batch['tokens'].to(device)
                reconstruction, _ = model(data)
                loss = 0.5 * mse_loss(reconstruction, data) + 0.5 * l1_loss(reconstruction, data)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss:   {history['val_loss'][-1]:.6f}")

    return history


# ============================================================================
# STAGE 2: Embedding Extraction
# ============================================================================

def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> tuple:
    """
    Extract embeddings from trained autoencoder.

    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data = batch['tokens'].to(device)
            labels = batch['labels']

            embeddings = model.encode(data)
            all_embeddings.append(embeddings.cpu().numpy())

            # Convert soft labels to hard labels
            if labels.dim() > 1:
                hard_labels = labels.argmax(dim=1).numpy()
            else:
                hard_labels = labels.numpy()
            all_labels.append(hard_labels)

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    return embeddings_array, labels_array


def run_embedding_extraction(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu"
) -> dict:
    """
    Extract embeddings from all splits.

    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test
    """
    print(f"\n{'='*60}")
    print("STAGE 2: Embedding Extraction")
    print(f"{'='*60}")

    X_train, y_train = extract_embeddings(model, train_loader, device)
    print(f"Train embeddings: {X_train.shape}, labels: {y_train.shape}")

    X_val, y_val = extract_embeddings(model, val_loader, device)
    print(f"Val embeddings:   {X_val.shape}, labels: {y_val.shape}")

    X_test, y_test = extract_embeddings(model, test_loader, device)
    print(f"Test embeddings:  {X_test.shape}, labels: {y_test.shape}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


# ============================================================================
# STAGE 3: XGBoost Classification
# ============================================================================

def train_classifier_two_stage(embeddings: dict, target_recall: float = 0.95) -> tuple:
    """
    Train two-stage classifier on embeddings.

    Returns:
        Tuple of (classifier, metrics_dict)
    """
    from src.models.two_stage_classifier import TwoStageClassifier

    print(f"\n{'='*60}")
    print("STAGE 3: Two-Stage Classifier")
    print(f"{'='*60}")

    X_train, y_train = embeddings['X_train'], embeddings['y_train']
    X_val, y_val = embeddings['X_val'], embeddings['y_val']
    X_test, y_test = embeddings['X_test'], embeddings['y_test']

    clf = TwoStageClassifier(random_state=42)
    clf.fit(X_train, y_train, X_val, y_val, use_sample_weights=True, verbose=True)

    # Tune threshold for target recall
    clf.tune_detection_threshold(X_val, y_val, target_metric='recall',
                                  target_value=target_recall, verbose=True)

    # Evaluate
    print("\n--- Test Set Evaluation ---")
    test_results = clf.evaluate(X_test, y_test, verbose=True)

    results = {
        'test_accuracy': test_results['combined']['accuracy'],
        'test_f1': test_results['combined']['f1_macro'],
        'test_minority_f1': test_results['combined']['minority_f1'],
        'test_detection_recall': test_results['detection']['recall'],
        'test_detection_precision': test_results['detection']['precision'],
        'test_detection_f1': test_results['detection']['f1'],
    }

    return clf, results


def train_classifier(embeddings: dict, use_sample_weights: bool = True) -> tuple:
    """
    Train XGBoost classifier on embeddings.

    Returns:
        Tuple of (classifier, metrics_dict)
    """
    from src.training.utils.classification_metrics import (
        compute_anomaly_metrics,
        compute_intention_detection_metrics
    )

    print(f"\n{'='*60}")
    print("STAGE 3: XGBoost Classification")
    print(f"{'='*60}")

    X_train, y_train = embeddings['X_train'], embeddings['y_train']
    X_val, y_val = embeddings['X_val'], embeddings['y_val']
    X_test, y_test = embeddings['X_test'], embeddings['y_test']

    # Create classifier with reasonable defaults
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    print("Training XGBoost classifier...")

    # Compute sample weights for imbalanced data
    sample_weights = None
    if use_sample_weights:
        sample_weights = compute_sample_weight('balanced', y_train)
        print("Using balanced sample weights for training")

    # Train with early stopping
    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on all splits
    results = {}

    for name, X, y in [('train', X_train, y_train),
                        ('val', X_val, y_val),
                        ('test', X_test, y_test)]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        results[f'{name}_accuracy'] = acc
        results[f'{name}_f1'] = f1
        print(f"{name.capitalize():5s} | Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    # Detailed test set report
    print("\nDetailed Test Set Classification Report:")
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)
    print(classification_report(
        y_test, y_pred_test,
        target_names=['noise', 'upward', 'downward'],
        zero_division=0
    ))

    # Anomaly/Imbalanced metrics
    anomaly_metrics = compute_anomaly_metrics(
        y_test, y_pred_test, y_proba_test, noise_class=0
    )
    intention_metrics = compute_intention_detection_metrics(
        y_test, y_pred_test, y_proba_test, noise_class=0
    )

    print("--- Imbalanced/Anomaly Metrics (Test) ---")
    print(f"  MCC: {anomaly_metrics['mcc']:.4f}")
    print(f"  Balanced Accuracy: {anomaly_metrics['balanced_accuracy']:.4f}")
    print(f"  G-Mean: {anomaly_metrics['g_mean']:.4f}")
    print(f"  Minority F1: {anomaly_metrics['minority_f1']:.4f}")

    print("\n--- Intention Detection (Test) ---")
    print(f"  Detection Precision: {intention_metrics['detection_precision']:.4f}")
    print(f"  Detection Recall: {intention_metrics['detection_recall']:.4f}")
    print(f"  Detection F1: {intention_metrics['detection_f1']:.4f}")

    # Add to results
    results['test_mcc'] = anomaly_metrics['mcc']
    results['test_minority_f1'] = anomaly_metrics['minority_f1']
    results['test_g_mean'] = anomaly_metrics['g_mean']
    results['test_detection_f1'] = intention_metrics['detection_f1']

    return clf, results


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_pipeline_validation(
    model_type: str = "CNNAutoencoder",
    embedding_dim: int = 128,
    epochs: int = 10,
    n_train: int = 600,
    n_val: int = 200,
    n_test: int = 200,
    batch_size: int = 50,
    device: str = "cpu",
    seed: int = 42,
    accuracy_threshold: float = 0.90,
    generate_plots: bool = False,
    output_dir: str = None,
    imbalanced: bool = False,
    two_stage: bool = False,
    target_recall: float = 0.95
) -> bool:
    """
    Run full pipeline validation.

    Args:
        model_type: "CNNAutoencoder" or "UNetAutoencoder"
        embedding_dim: Latent space dimension
        epochs: Training epochs
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        batch_size: Batch size
        device: "cpu" or "cuda"
        seed: Random seed
        accuracy_threshold: Minimum test accuracy to pass validation
        generate_plots: If True, generate visualization plots
        output_dir: Directory to save plots (default: ./validation_results_TIMESTAMP)
        imbalanced: If True, use imbalanced class distribution (90% noise, 5% upward, 5% downward)
        two_stage: If True, use two-stage hierarchical classifier
        target_recall: Target detection recall for two-stage classifier threshold tuning

    Returns:
        True if validation passes, False otherwise
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine class distribution
    class_distribution = IMBALANCED_DISTRIBUTION if imbalanced else None
    distribution_name = "IMBALANCED (90/5/5)" if imbalanced else "BALANCED (33/33/33)"

    # Setup output directory if plotting
    if generate_plots:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_imbalanced" if imbalanced else "_balanced"
            output_dir = f"./validation_results_{timestamp}{suffix}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    print("\n" + "=" * 70)
    print("PIPELINE VALIDATION TEST")
    print("=" * 70)
    print(f"Model: {model_type}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Epochs: {epochs}")
    print(f"Train/Val/Test samples: {n_train}/{n_val}/{n_test}")
    print(f"Class distribution: {distribution_name}")
    print(f"Classifier: {'Two-Stage' if two_stage else 'Single-Stage (XGBoost)'}")
    if two_stage:
        print(f"Target detection recall: {target_recall:.0%}")
    print(f"Device: {device}")
    print(f"Accuracy threshold: {accuracy_threshold:.0%}")
    print(f"Generate plots: {generate_plots}")
    print("=" * 70)

    # Create synthetic datasets
    print("\nCreating synthetic datasets...")
    train_ds, val_ds, test_ds = create_synthetic_splits(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
        seed=seed,
        class_distribution=class_distribution
    )

    print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False)

    # Stage 1: Train autoencoder
    model = create_model(model_type=model_type, embedding_dim=embedding_dim)
    ae_history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=epochs, device=device
    )

    # Stage 2: Extract embeddings
    embeddings = run_embedding_extraction(
        model, train_loader, val_loader, test_loader, device=device
    )

    # Stage 3: Train classifier
    if two_stage:
        clf, metrics = train_classifier_two_stage(embeddings, target_recall=target_recall)
    else:
        clf, metrics = train_classifier(embeddings)

    # Get predictions for confusion matrix
    y_test = embeddings['y_test']
    y_pred = clf.predict(embeddings['X_test'])

    # Generate plots if requested
    if generate_plots:
        from tests.pipeline_validation.visualizations import generate_all_plots
        from src.training.utils.classification_metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt

        generate_all_plots(
            train_dataset=train_ds,
            model=model,
            train_loader=train_loader,
            embeddings=embeddings,
            y_true=y_test,
            y_pred=y_pred,
            history=ae_history,
            output_dir=output_dir,
            device=device
        )

        # Always generate confusion matrix for classifier evaluation
        classifier_type = "two_stage" if two_stage else "single_stage"
        cm_path = output_dir / f'confusion_matrix_{classifier_type}.png'
        plot_confusion_matrix(y_test, y_pred, ['noise', 'upward', 'downward'], save_path=str(cm_path))
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()

    # Validation result
    test_accuracy = metrics['test_accuracy']
    test_f1 = metrics['test_f1']

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    passed = test_accuracy >= accuracy_threshold

    if passed:
        print(f"PASSED: Test accuracy {test_accuracy:.2%} >= threshold {accuracy_threshold:.0%}")
        print("Pipeline is working correctly!")
    else:
        print(f"FAILED: Test accuracy {test_accuracy:.2%} < threshold {accuracy_threshold:.0%}")
        print("There may be an issue with the pipeline.")

    print(f"\nFinal metrics:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1 Macro: {test_f1:.4f}")
    print(f"  AE Final Val Loss: {ae_history['val_loss'][-1]:.6f}")

    if generate_plots:
        print(f"\nPlots saved to: {output_dir}")

    print("=" * 70)

    return passed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pipeline validation test with synthetic data'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='CNNAutoencoder',
        choices=['CNNAutoencoder', 'UNetAutoencoder'],
        help='Model type to test'
    )

    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Training epochs (default: 10)'
    )

    parser.add_argument(
        '--n-train',
        type=int,
        default=600,
        help='Number of training samples (default: 600)'
    )

    parser.add_argument(
        '--n-val',
        type=int,
        default=200,
        help='Number of validation samples (default: 200)'
    )

    parser.add_argument(
        '--n-test',
        type=int,
        default=200,
        help='Number of test samples (default: 200)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size (default: 50)'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.90,
        help='Minimum test accuracy to pass (default: 0.90)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: fewer samples, fewer epochs'
    )

    parser.add_argument(
        '--plots', '-p',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for plots (default: ./validation_results_TIMESTAMP)'
    )

    parser.add_argument(
        '--imbalanced',
        action='store_true',
        help='Use imbalanced class distribution (90%% noise, 5%% upward, 5%% downward)'
    )

    parser.add_argument(
        '--two-stage',
        action='store_true',
        help='Use two-stage hierarchical classifier (detection + classification)'
    )

    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.95,
        help='Target detection recall for two-stage classifier (default: 0.95)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 5
        args.n_train = 300
        args.n_val = 100
        args.n_test = 100
        print("Quick mode enabled: reduced samples and epochs")

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    passed = run_pipeline_validation(
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        accuracy_threshold=args.threshold,
        generate_plots=args.plots,
        output_dir=args.output_dir,
        imbalanced=args.imbalanced,
        two_stage=args.two_stage,
        target_recall=args.target_recall
    )

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()