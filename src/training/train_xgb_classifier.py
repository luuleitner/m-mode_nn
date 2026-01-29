"""
XGBoost Classifier Training

Train an XGBoost classifier on extracted embeddings for movement classification.

Usage:
    python -m src.training.train_classifier --config config/config.yaml --embeddings path/to/embeddings.npz
    python -m src.training.train_classifier --config config/config.yaml --embeddings path/to/embeddings.npz --tune
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.training.utils.classification_metrics import (
    compute_classification_metrics,
    compute_anomaly_metrics,
    compute_intention_detection_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_embedding_tsne
)
from src.training.utils.threshold_optimizer import (
    find_optimal_threshold,
    apply_threshold,
    plot_threshold_analysis,
    compute_metrics_at_threshold
)

import utils.logging_config as logconf
logger = logconf.get_logger("CLASSIFIER")

# Default class names for the 3-class problem
CLASS_NAMES = ['noise', 'upward', 'downward']


def compute_balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute sample weights to handle class imbalance.

    Uses inverse class frequency so minority classes get higher weights.
    This helps XGBoost pay more attention to minority classes.

    Args:
        y: Label array

    Returns:
        Array of sample weights (same length as y)
    """
    sample_weights = compute_sample_weight('balanced', y)

    # Log class weights for transparency
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
    logger.info(f"Class weights (inverse frequency): {weight_dict}")

    return sample_weights


def load_embeddings(embeddings_path: str) -> dict:
    """Load embeddings from .npz file."""
    logger.info(f"Loading embeddings from: {embeddings_path}")

    data = np.load(embeddings_path, allow_pickle=True)

    embeddings = {
        'X_train': data['X_train'],
        'X_val': data['X_val'],
        'X_test': data['X_test'],
        'y_train': data['y_train'],
        'y_val': data['y_val'],
        'y_test': data['y_test'],
    }

    # Log info
    logger.info(f"Train: {embeddings['X_train'].shape}, Val: {embeddings['X_val'].shape}, Test: {embeddings['X_test'].shape}")

    if 'embedding_dim' in data:
        logger.info(f"Embedding dimension: {data['embedding_dim']}")
    if 'normalized' in data:
        logger.info(f"Normalized: {data['normalized']}")

    return embeddings


def get_classifier_config(config) -> dict:
    """Get XGBoost configuration from config or use defaults."""
    # Check if classifier config exists (nested under ml section)
    if hasattr(config.ml, 'classifier') and hasattr(config.ml.classifier, 'xgboost'):
        xgb_cfg = config.ml.classifier.xgboost
        logger.info("Loading XGBoost configuration from config file")
        return {
            'n_estimators': getattr(xgb_cfg, 'n_estimators', 300),
            'max_depth': getattr(xgb_cfg, 'max_depth', 6),
            'learning_rate': getattr(xgb_cfg, 'learning_rate', 0.05),
            'subsample': getattr(xgb_cfg, 'subsample', 0.8),
            'colsample_bytree': getattr(xgb_cfg, 'colsample_bytree', 0.8),
            'reg_alpha': getattr(xgb_cfg, 'reg_alpha', 0.1),
            'reg_lambda': getattr(xgb_cfg, 'reg_lambda', 1.0),
            'min_child_weight': getattr(xgb_cfg, 'min_child_weight', 3),
            'gamma': getattr(xgb_cfg, 'gamma', 0.1),
            'early_stopping_rounds': getattr(xgb_cfg, 'early_stopping_rounds', 30),
        }
    else:
        # Default configuration
        logger.warning("Classifier config not found in config.ml.classifier, using defaults")
        return {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'early_stopping_rounds': 30,
        }


def create_classifier(xgb_config: dict, seed: int = 42) -> XGBClassifier:
    """Create XGBoost classifier with given configuration."""
    clf = XGBClassifier(
        n_estimators=xgb_config['n_estimators'],
        max_depth=xgb_config['max_depth'],
        learning_rate=xgb_config['learning_rate'],
        subsample=xgb_config['subsample'],
        colsample_bytree=xgb_config['colsample_bytree'],
        reg_alpha=xgb_config['reg_alpha'],
        reg_lambda=xgb_config['reg_lambda'],
        min_child_weight=xgb_config.get('min_child_weight', 3),
        gamma=xgb_config.get('gamma', 0.1),
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
        verbosity=1
    )
    return clf


def train_classifier(
    clf: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray = None,
    early_stopping_rounds: int = 50
) -> XGBClassifier:
    """
    Train XGBoost classifier with early stopping and optional sample weighting.

    Args:
        clf: XGBClassifier instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sample_weight: Optional per-sample weights for handling class imbalance
        early_stopping_rounds: Stop if no improvement for N rounds
    """
    logger.info("Training XGBoost classifier...")
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    if sample_weight is not None:
        logger.info("Using sample weights for class balancing")

    clf.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # Get best iteration
    if hasattr(clf, 'best_iteration'):
        logger.info(f"Best iteration: {clf.best_iteration}")

    return clf


def evaluate_classifier(
    clf: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test",
    include_anomaly_metrics: bool = True
) -> dict:
    """Evaluate classifier and return metrics including anomaly-focused metrics."""
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    # Standard metrics
    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=CLASS_NAMES
    )

    logger.info(f"\n{split_name} Set Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    if 'roc_auc_macro' in metrics:
        logger.info(f"  ROC-AUC Macro: {metrics['roc_auc_macro']:.4f}")

    # Anomaly-focused metrics for imbalanced data
    if include_anomaly_metrics:
        anomaly_metrics = compute_anomaly_metrics(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            class_names=CLASS_NAMES,
            noise_class=0
        )
        metrics.update({f'anomaly_{k}': v for k, v in anomaly_metrics.items()})

        intention_metrics = compute_intention_detection_metrics(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            noise_class=0
        )
        metrics.update({f'intention_{k}': v for k, v in intention_metrics.items()})

        logger.info(f"  --- Anomaly/Imbalanced Metrics ---")
        logger.info(f"  MCC: {anomaly_metrics['mcc']:.4f}")
        logger.info(f"  Balanced Accuracy: {anomaly_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  G-Mean: {anomaly_metrics['g_mean']:.4f}")
        logger.info(f"  Minority F1: {anomaly_metrics['minority_f1']:.4f}")
        if 'minority_ap' in anomaly_metrics:
            logger.info(f"  Minority AP: {anomaly_metrics['minority_ap']:.4f}")
        logger.info(f"  --- Intention Detection ---")
        logger.info(f"  Detection Precision: {intention_metrics['detection_precision']:.4f}")
        logger.info(f"  Detection Recall: {intention_metrics['detection_recall']:.4f}")
        logger.info(f"  Detection F1: {intention_metrics['detection_f1']:.4f}")
        if intention_metrics['detection_ap'] is not None:
            logger.info(f"  Detection AP: {intention_metrics['detection_ap']:.4f}")

    return metrics


def save_results(
    clf: XGBClassifier,
    metrics: dict,
    xgb_config: dict,
    output_dir: str,
    embeddings_path: str
):
    """Save trained classifier and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = os.path.join(output_dir, f'xgboost_classifier_{timestamp}.json')
    clf.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Also save as latest
    latest_model_path = os.path.join(output_dir, 'xgboost_classifier_latest.json')
    clf.save_model(latest_model_path)

    # Save metrics and config
    results = {
        'timestamp': timestamp,
        'embeddings_path': embeddings_path,
        'xgboost_config': xgb_config,
        'test_metrics': metrics,
        'feature_importances': clf.feature_importances_.tolist()
    }

    results_path = os.path.join(output_dir, f'classification_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return model_path, results_path


def generate_plots(
    clf: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    X_all: np.ndarray = None,
    y_all: np.ndarray = None
):
    """Generate and save evaluation plots."""
    logger.info("Generating evaluation plots...")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, save_path=cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # ROC curves
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(y_test, y_proba, CLASS_NAMES, save_path=roc_path)
    logger.info(f"ROC curves saved to: {roc_path}")

    # Precision-Recall curves
    pr_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plot_precision_recall_curves(y_test, y_proba, CLASS_NAMES, save_path=pr_path)
    logger.info(f"PR curves saved to: {pr_path}")

    # Threshold analysis plot (for intention detection)
    try:
        # Find optimal threshold for F1
        threshold_result = find_optimal_threshold(
            y_test, y_proba, target_metric='f1', noise_class=0
        )
        optimal_threshold = threshold_result['threshold']

        threshold_path = os.path.join(output_dir, 'threshold_analysis.png')
        plot_threshold_analysis(
            y_test, y_proba,
            optimal_threshold=optimal_threshold,
            noise_class=0,
            save_path=threshold_path
        )
        logger.info(f"Threshold analysis saved to: {threshold_path}")
        logger.info(f"Optimal threshold (F1): {optimal_threshold:.3f} "
                    f"(P={threshold_result['precision']:.3f}, R={threshold_result['recall']:.3f})")

        # Also find high-recall threshold
        high_recall_result = find_optimal_threshold(
            y_test, y_proba, target_metric='recall', target_value=0.95, noise_class=0
        )
        logger.info(f"High-recall threshold (R>=95%): {high_recall_result['threshold']:.3f} "
                    f"(P={high_recall_result['precision']:.3f}, R={high_recall_result['recall']:.3f})")
    except Exception as e:
        logger.warning(f"Could not generate threshold analysis: {e}")

    # t-SNE visualization (on test set or combined)
    if X_all is not None and y_all is not None:
        # Sample if too large
        max_samples = 5000
        if len(X_all) > max_samples:
            indices = np.random.choice(len(X_all), max_samples, replace=False)
            X_tsne = X_all[indices]
            y_tsne = y_all[indices]
        else:
            X_tsne = X_all
            y_tsne = y_all

        tsne_path = os.path.join(output_dir, 'embedding_tsne.png')
        plot_embedding_tsne(X_tsne, y_tsne, CLASS_NAMES, save_path=tsne_path)
        logger.info(f"t-SNE plot saved to: {tsne_path}")


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    optimize_for: str = 'minority_f1'
) -> dict:
    """
    Perform hyperparameter search using Optuna.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of Optuna trials
        seed: Random seed
        optimize_for: Metric to optimize:
            - 'minority_f1': Average F1 of minority classes (upward, downward)
            - 'intention_ap': Average Precision for intention detection
            - 'f1_macro': Standard macro F1 (original behavior)
            - 'mcc': Matthews Correlation Coefficient

    Returns:
        Best hyperparameters dictionary
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        raise

    logger.info(f"Starting hyperparameter search with {n_trials} trials...")
    logger.info(f"Optimizing for: {optimize_for}")

    # Compute sample weights for balanced training
    sample_weights = compute_balanced_sample_weights(y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        clf = XGBClassifier(
            **params,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )

        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        if optimize_for == 'minority_f1':
            # Average F1 of minority classes (classes 1 and 2)
            from sklearn.metrics import f1_score
            f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
            return (f1_per_class[1] + f1_per_class[2]) / 2

        elif optimize_for == 'intention_ap':
            # Average Precision for intention detection (noise vs intention)
            from sklearn.metrics import average_precision_score
            y_val_binary = (y_val > 0).astype(int)
            p_intention = y_proba[:, 1] + y_proba[:, 2]
            return average_precision_score(y_val_binary, p_intention)

        elif optimize_for == 'mcc':
            from sklearn.metrics import matthews_corrcoef
            return matthews_corrcoef(y_val, y_pred)

        else:  # f1_macro (original)
            from sklearn.metrics import f1_score
            return f1_score(y_val, y_pred, average='macro')

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial {optimize_for} score: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

    # Add early_stopping_rounds to best params
    best_params = study.best_trial.params
    best_params['early_stopping_rounds'] = 50

    return best_params


def main(
    config_path: str,
    embeddings_path: str,
    output_dir: str = None,
    tune: bool = False,
    n_trials: int = 50
):
    """
    Main classifier training function.

    Args:
        config_path: Path to config YAML file
        embeddings_path: Path to embeddings .npz file
        output_dir: Output directory (default: alongside embeddings)
        tune: Whether to perform hyperparameter tuning
        n_trials: Number of Optuna trials if tuning
    """
    # Load config
    config = load_config(config_path, create_dirs=False)
    setup_environment(config)

    seed = config.get_seed() or 42

    logger.info("=" * 60)
    logger.info("XGBOOST CLASSIFIER TRAINING")
    logger.info("=" * 60)

    # Setup output directory
    # Classifier results go in the training run folder (sibling to embeddings folder)
    # e.g., .../training_20260129_002037/classifier/
    if output_dir is None:
        embeddings_dir = os.path.dirname(embeddings_path)  # .../embeddings/
        run_dir = os.path.dirname(embeddings_dir)          # .../training_YYYYMMDD_HHMMSS/
        output_dir = os.path.join(run_dir, 'classifier')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load embeddings
    embeddings = load_embeddings(embeddings_path)
    X_train, y_train = embeddings['X_train'], embeddings['y_train']
    X_val, y_val = embeddings['X_val'], embeddings['y_val']
    X_test, y_test = embeddings['X_test'], embeddings['y_test']

    # Get or tune hyperparameters
    if tune:
        xgb_config = hyperparameter_search(
            X_train, y_train, X_val, y_val,
            n_trials=n_trials, seed=seed
        )
    else:
        xgb_config = get_classifier_config(config)

    logger.info(f"XGBoost config: {xgb_config}")

    # Compute sample weights for class balancing
    # This helps XGBoost pay more attention to minority classes (upward/downward movements)
    sample_weights = compute_balanced_sample_weights(y_train)

    # Create and train classifier
    clf = create_classifier(xgb_config, seed=seed)
    clf = train_classifier(
        clf, X_train, y_train, X_val, y_val,
        sample_weight=sample_weights,
        early_stopping_rounds=xgb_config.get('early_stopping_rounds', 50)
    )

    # Evaluate on all splits
    logger.info("\n" + "=" * 40)
    train_metrics = evaluate_classifier(clf, X_train, y_train, "Train")
    val_metrics = evaluate_classifier(clf, X_val, y_val, "Validation")
    test_metrics = evaluate_classifier(clf, X_test, y_test, "Test")

    # Print detailed classification report
    logger.info("\nDetailed Classification Report (Test Set):")
    print_classification_report(y_test, clf.predict(X_test), CLASS_NAMES)

    # Save results
    model_path, results_path = save_results(
        clf, test_metrics, xgb_config, output_dir, embeddings_path
    )

    # Generate plots
    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    generate_plots(clf, X_test, y_test, output_dir, X_all, y_all)

    # Print summary
    print_summary(train_metrics, val_metrics, test_metrics, output_dir)

    return clf, test_metrics


def print_summary(train_metrics, val_metrics, test_metrics, output_dir):
    """Print training summary."""
    print("\n" + "=" * 60)
    print("CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)

    print("\nAccuracy:")
    print(f"  Train: {train_metrics['accuracy']:.4f}")
    print(f"  Val:   {val_metrics['accuracy']:.4f}")
    print(f"  Test:  {test_metrics['accuracy']:.4f}")

    print("\nF1 Score (Macro):")
    print(f"  Train: {train_metrics['f1_macro']:.4f}")
    print(f"  Val:   {val_metrics['f1_macro']:.4f}")
    print(f"  Test:  {test_metrics['f1_macro']:.4f}")

    print("\nPer-class F1 (Test):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {test_metrics['f1_per_class'][i]:.4f}")

    if 'roc_auc_macro' in test_metrics:
        print(f"\nROC-AUC (Macro): {test_metrics['roc_auc_macro']:.4f}")

    # Anomaly/Imbalanced metrics
    if 'anomaly_mcc' in test_metrics:
        print("\n--- Imbalanced/Anomaly Metrics (Test) ---")
        print(f"  MCC: {test_metrics['anomaly_mcc']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics['anomaly_balanced_accuracy']:.4f}")
        print(f"  G-Mean: {test_metrics['anomaly_g_mean']:.4f}")
        print(f"  Minority F1: {test_metrics['anomaly_minority_f1']:.4f}")
        if 'anomaly_minority_ap' in test_metrics:
            print(f"  Minority AP: {test_metrics['anomaly_minority_ap']:.4f}")

    # Intention detection metrics
    if 'intention_detection_f1' in test_metrics:
        print("\n--- Intention Detection (Test) ---")
        print(f"  Precision: {test_metrics['intention_detection_precision']:.4f}")
        print(f"  Recall: {test_metrics['intention_detection_recall']:.4f}")
        print(f"  F1: {test_metrics['intention_detection_f1']:.4f}")
        if test_metrics.get('intention_detection_ap') is not None:
            print(f"  AP: {test_metrics['intention_detection_ap']:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train XGBoost classifier on embeddings')

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--embeddings', '-e',
        type=str,
        required=True,
        help='Path to embeddings .npz file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: alongside embeddings)'
    )

    parser.add_argument(
        '--tune', '-t',
        action='store_true',
        help='Perform hyperparameter tuning with Optuna'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for tuning (default: 50)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        config_path=args.config,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        tune=args.tune,
        n_trials=args.n_trials
    )