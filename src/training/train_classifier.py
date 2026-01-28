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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.training.utils.classification_metrics import (
    compute_classification_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_embedding_tsne
)

import utils.logging_config as logconf
logger = logconf.get_logger("CLASSIFIER")

# Default class names for the 3-class problem
CLASS_NAMES = ['noise', 'upward', 'downward']


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
    # Check if classifier config exists
    if hasattr(config, 'classifier') and hasattr(config.classifier, 'xgboost'):
        xgb_cfg = config.classifier.xgboost
        return {
            'n_estimators': getattr(xgb_cfg, 'n_estimators', 500),
            'max_depth': getattr(xgb_cfg, 'max_depth', 6),
            'learning_rate': getattr(xgb_cfg, 'learning_rate', 0.1),
            'subsample': getattr(xgb_cfg, 'subsample', 0.8),
            'colsample_bytree': getattr(xgb_cfg, 'colsample_bytree', 0.8),
            'reg_alpha': getattr(xgb_cfg, 'reg_alpha', 0.0),
            'reg_lambda': getattr(xgb_cfg, 'reg_lambda', 1.0),
            'early_stopping_rounds': getattr(xgb_cfg, 'early_stopping_rounds', 50),
        }
    else:
        # Default configuration
        logger.info("Using default XGBoost configuration")
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 50,
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
    early_stopping_rounds: int = 50
) -> XGBClassifier:
    """Train XGBoost classifier with early stopping."""
    logger.info("Training XGBoost classifier...")
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    clf.fit(
        X_train, y_train,
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
    split_name: str = "Test"
) -> dict:
    """Evaluate classifier and return metrics."""
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

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
    seed: int = 42
) -> dict:
    """
    Perform hyperparameter search using Optuna.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of Optuna trials
        seed: Random seed

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

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
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
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = clf.predict(X_val)
        from sklearn.metrics import f1_score
        return f1_score(y_val, y_pred, average='macro')

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial F1 score: {study.best_trial.value:.4f}")
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
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(embeddings_path), 'classifier')
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

    # Create and train classifier
    clf = create_classifier(xgb_config, seed=seed)
    clf = train_classifier(
        clf, X_train, y_train, X_val, y_val,
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