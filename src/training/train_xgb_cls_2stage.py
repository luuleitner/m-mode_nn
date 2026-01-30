"""
Two-Stage XGBoost Classifier Training

Train a two-stage hierarchical XGBoost classifier on extracted embeddings:
  Stage 1: Intention detection (noise vs intention)
  Stage 2: Direction classification (upward vs downward)

Usage:
    python -m src.training.train_xgb_cls_2stage --config config/config.yaml --embeddings path/to/embeddings.npz
    python -m src.training.train_xgb_cls_2stage --config config/config.yaml --embeddings path/to/embeddings.npz --tune
    python -m src.training.train_xgb_cls_2stage --config config/config.yaml --embeddings path/to/embeddings.npz --target-recall 0.95
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment
from src.models.two_stage_classifier import TwoStageClassifier
from src.training.utils.classification_metrics import (
    compute_anomaly_metrics,
    compute_intention_detection_metrics
)
from src.training.utils.threshold_optimizer import plot_threshold_analysis

import utils.logging_config as logconf
logger = logconf.get_logger("TWO_STAGE")

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

    logger.info(f"Train: {embeddings['X_train'].shape}, Val: {embeddings['X_val'].shape}, Test: {embeddings['X_test'].shape}")

    # Log class distribution
    for split in ['train', 'val', 'test']:
        y = embeddings[f'y_{split}']
        counts = np.bincount(y, minlength=3)
        logger.info(f"  {split}: noise={counts[0]}, upward={counts[1]}, downward={counts[2]}")

    return embeddings


def hyperparameter_search_two_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    target_recall: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Hyperparameter search for two-stage classifier using Optuna.

    Optimizes for:
    - Stage 1: Detection recall >= target with max precision
    - Stage 2: Direction accuracy
    - Combined: Minority F1

    Returns:
        Best parameters for both stages
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        raise

    logger.info(f"Starting hyperparameter search with {n_trials} trials...")
    logger.info(f"Target detection recall: {target_recall}")

    def objective(trial):
        detector_params = {
            'n_estimators': trial.suggest_int('det_n_estimators', 100, 500),
            'max_depth': trial.suggest_int('det_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('det_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('det_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('det_colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('det_min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('det_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('det_reg_lambda', 1e-8, 10.0, log=True),
        }

        classifier_params = {
            'n_estimators': trial.suggest_int('cls_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('cls_max_depth', 3, 7),
            'learning_rate': trial.suggest_float('cls_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('cls_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cls_colsample_bytree', 0.6, 1.0),
        }

        clf = TwoStageClassifier(
            detector_params=detector_params,
            classifier_params=classifier_params,
            random_state=seed
        )

        clf.fit(X_train, y_train, X_val, y_val, verbose=False)
        clf.tune_detection_threshold(X_val, y_val, target_metric='recall',
                                     target_value=target_recall, verbose=False)

        results = clf.evaluate(X_val, y_val, verbose=False)

        # Multi-objective: want high recall AND good minority F1
        detection_recall = results['detection']['recall']
        minority_f1 = results['combined']['minority_f1']

        # Penalize if recall below target
        recall_penalty = max(0, target_recall - detection_recall) * 2

        return minority_f1 - recall_penalty

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial score: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

    # Extract best params for each stage
    best_params = study.best_trial.params

    detector_params = {k.replace('det_', ''): v for k, v in best_params.items() if k.startswith('det_')}
    classifier_params = {k.replace('cls_', ''): v for k, v in best_params.items() if k.startswith('cls_')}

    return {
        'detector_params': detector_params,
        'classifier_params': classifier_params
    }


def _to_python_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(
    clf: TwoStageClassifier,
    results: dict,
    output_dir: str,
    embeddings_path: str
):
    """Save trained classifier and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = os.path.join(output_dir, f'two_stage_classifier_{timestamp}')
    clf.save(model_path)
    logger.info(f"Model saved to: {model_path}_*.json")

    # Also save as latest
    latest_path = os.path.join(output_dir, 'two_stage_classifier_latest')
    clf.save(latest_path)

    # Save results (convert numpy types to Python types)
    results_serializable = _to_python_types({
        'timestamp': timestamp,
        'embeddings_path': embeddings_path,
        'detection_threshold': clf.detection_threshold,
        'detection': results['detection'],
        'direction': results['direction'],
        'combined': results['combined']
    })

    results_path = os.path.join(output_dir, f'two_stage_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return model_path, results_path


def generate_plots(
    clf: TwoStageClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
):
    """Generate evaluation plots for two-stage classifier."""
    from src.training.utils.classification_metrics import (
        plot_confusion_matrix,
        plot_precision_recall_curves,
        plot_roc_curves
    )
    import matplotlib.pyplot as plt

    logger.info("Generating evaluation plots...")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # 3-class confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix_3class.png')
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, save_path=cm_path)
    logger.info(f"3-class confusion matrix saved to: {cm_path}")
    plt.close()

    # Binary confusion matrix (Stage 1: detection)
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    cm_binary_path = os.path.join(output_dir, 'confusion_matrix_detection.png')
    plot_confusion_matrix(y_test_binary, y_pred_binary, ['noise', 'intention'], save_path=cm_binary_path)
    logger.info(f"Detection confusion matrix saved to: {cm_binary_path}")
    plt.close()

    # ROC curves
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(y_test, y_proba, CLASS_NAMES, save_path=roc_path)
    logger.info(f"ROC curves saved to: {roc_path}")
    plt.close()

    # PR curves
    pr_path = os.path.join(output_dir, 'pr_curves.png')
    plot_precision_recall_curves(y_test, y_proba, CLASS_NAMES, save_path=pr_path)
    logger.info(f"PR curves saved to: {pr_path}")
    plt.close()

    # Threshold analysis (Stage 1)
    try:
        threshold_path = os.path.join(output_dir, 'threshold_analysis.png')
        plot_threshold_analysis(
            y_test, y_proba,
            optimal_threshold=clf.detection_threshold,
            noise_class=0,
            save_path=threshold_path
        )
        logger.info(f"Threshold analysis saved to: {threshold_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate threshold analysis: {e}")

    logger.info(f"All plots saved to: {output_dir}")


def main(
    config_path: str,
    embeddings_path: str,
    output_dir: str = None,
    tune: bool = False,
    n_trials: int = 30,
    target_recall: float = 0.95
):
    """
    Main two-stage classifier training function.

    Args:
        config_path: Path to config YAML file
        embeddings_path: Path to embeddings .npz file
        output_dir: Output directory (default: alongside embeddings)
        tune: Whether to perform hyperparameter tuning
        n_trials: Number of Optuna trials if tuning
        target_recall: Target detection recall for threshold tuning
    """
    config = load_config(config_path, create_dirs=False)
    setup_environment(config)

    seed = config.get_seed() or 42

    logger.info("=" * 60)
    logger.info("TWO-STAGE CLASSIFIER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Target detection recall: {target_recall}")

    # Setup output directory
    if output_dir is None:
        embeddings_dir = os.path.dirname(embeddings_path)
        run_dir = os.path.dirname(embeddings_dir)
        output_dir = os.path.join(run_dir, 'classifier_two_stage')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load embeddings
    embeddings = load_embeddings(embeddings_path)
    X_train, y_train = embeddings['X_train'], embeddings['y_train']
    X_val, y_val = embeddings['X_val'], embeddings['y_val']
    X_test, y_test = embeddings['X_test'], embeddings['y_test']

    # Get or tune hyperparameters
    detector_params = None
    classifier_params = None

    if tune:
        params = hyperparameter_search_two_stage(
            X_train, y_train, X_val, y_val,
            n_trials=n_trials, target_recall=target_recall, seed=seed
        )
        detector_params = params['detector_params']
        classifier_params = params['classifier_params']

    # Create and train classifier
    clf = TwoStageClassifier(
        detector_params=detector_params,
        classifier_params=classifier_params,
        random_state=seed
    )

    clf.fit(X_train, y_train, X_val, y_val, use_sample_weights=True, verbose=True)

    # Tune threshold for target recall
    clf.tune_detection_threshold(
        X_val, y_val,
        target_metric='recall',
        target_value=target_recall,
        verbose=True
    )

    # Evaluate on all splits
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION")
    logger.info("=" * 40)

    print("\n--- Train Set ---")
    train_results = clf.evaluate(X_train, y_train, CLASS_NAMES, verbose=True)

    print("\n--- Validation Set ---")
    val_results = clf.evaluate(X_val, y_val, CLASS_NAMES, verbose=True)

    print("\n--- Test Set ---")
    test_results = clf.evaluate(X_test, y_test, CLASS_NAMES, verbose=True)

    # Save results
    model_path, results_path = save_results(
        clf, test_results, output_dir, embeddings_path
    )

    # Generate plots
    generate_plots(clf, X_test, y_test, output_dir)

    # Print summary
    print_summary(test_results, clf.detection_threshold, output_dir)

    return clf, test_results


def print_summary(results: dict, threshold: float, output_dir: str):
    """Print training summary."""
    print("\n" + "=" * 60)
    print("TWO-STAGE CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)

    print("\nStage 1 - Intention Detection:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Precision: {results['detection']['precision']:.4f}")
    print(f"  Recall: {results['detection']['recall']:.4f}")
    print(f"  F1: {results['detection']['f1']:.4f}")
    print(f"  AP (PR-AUC): {results['detection']['ap']:.4f}")

    if results['direction']:
        print(f"\nStage 2 - Direction Classification:")
        print(f"  Accuracy: {results['direction']['accuracy']:.4f}")

    print(f"\nCombined 3-Class:")
    print(f"  Accuracy: {results['combined']['accuracy']:.4f}")
    print(f"  F1 Macro: {results['combined']['f1_macro']:.4f}")
    print(f"  Minority F1: {results['combined']['minority_f1']:.4f}")

    print(f"\n  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {results['combined']['f1_per_class'][i]:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train two-stage classifier on embeddings')

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
        default=30,
        help='Number of Optuna trials for tuning (default: 30)'
    )

    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.95,
        help='Target detection recall for threshold tuning (default: 0.95)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        config_path=args.config,
        embeddings_path=args.embeddings,
        output_dir=args.output,
        tune=args.tune,
        n_trials=args.n_trials,
        target_recall=args.target_recall
    )
