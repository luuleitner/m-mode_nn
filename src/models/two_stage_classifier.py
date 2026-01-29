"""
Two-Stage Hierarchical Classifier for Imbalanced Anomaly Detection.

Stage 1: Intention Detection (binary: noise vs intention)
  - Optimized for high recall on intentions
  - Tunable detection threshold

Stage 2: Direction Classification (binary: upward vs downward)
  - Only runs on detected intentions
  - Trained exclusively on intention samples

This separation allows independent optimization of detection (don't miss intentions)
and classification (correctly identify direction).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score, classification_report
)


class TwoStageClassifier:
    """
    Two-stage hierarchical classifier for intention detection and classification.

    Stage 1: Binary detector (noise vs intention)
    Stage 2: Binary classifier (upward vs downward), conditional on detection

    Args:
        detector_params: XGBoost parameters for Stage 1 detector
        classifier_params: XGBoost parameters for Stage 2 classifier
        detection_threshold: Initial threshold for P(intention) (default: 0.5)
        noise_class: Index of noise class (default: 0)
        random_state: Random seed for reproducibility

    Example:
        >>> clf = TwoStageClassifier()
        >>> clf.fit(X_train, y_train)
        >>> clf.tune_detection_threshold(X_val, y_val, target_recall=0.95)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        detector_params: Optional[Dict] = None,
        classifier_params: Optional[Dict] = None,
        detection_threshold: float = 0.5,
        noise_class: int = 0,
        random_state: int = 42
    ):
        self.noise_class = noise_class
        self.detection_threshold = detection_threshold
        self.random_state = random_state

        # Default parameters optimized for each stage
        default_detector_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # PR-AUC for imbalanced detection
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

        default_classifier_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Merge with provided params
        self.detector_params = {**default_detector_params, **(detector_params or {})}
        self.classifier_params = {**default_classifier_params, **(classifier_params or {})}

        self.detector = None  # Stage 1: noise vs intention
        self.classifier = None  # Stage 2: upward vs downward

        # Store class mapping for Stage 2
        self.intention_classes = None  # e.g., [1, 2] for upward, downward

    def _convert_to_binary(self, y: np.ndarray) -> np.ndarray:
        """Convert multi-class labels to binary (noise=0, intention=1)."""
        return (y != self.noise_class).astype(int)

    def _convert_to_direction(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract intention samples and convert to direction labels.

        Returns:
            Tuple of (mask for intentions, direction labels 0/1)
        """
        intention_mask = y != self.noise_class

        if self.intention_classes is None:
            # Discover intention classes from data
            self.intention_classes = sorted([c for c in np.unique(y) if c != self.noise_class])

        # Map intention classes to 0, 1, ... for Stage 2
        y_direction = np.zeros(intention_mask.sum(), dtype=int)
        y_intentions = y[intention_mask]

        for i, cls in enumerate(self.intention_classes):
            y_direction[y_intentions == cls] = i

        return intention_mask, y_direction

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_sample_weights: bool = True,
        verbose: bool = True
    ) -> 'TwoStageClassifier':
        """
        Train both stages of the classifier.

        Args:
            X: Training features (N, D)
            y: Training labels (N,) with values in {0, 1, 2, ...}
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional)
            use_sample_weights: Whether to use balanced sample weights
            verbose: Print training progress

        Returns:
            self
        """
        if verbose:
            print("=" * 60)
            print("TWO-STAGE CLASSIFIER TRAINING")
            print("=" * 60)

        # ===== Stage 1: Train Detector =====
        if verbose:
            print("\n--- Stage 1: Intention Detector (noise vs intention) ---")

        y_binary = self._convert_to_binary(y)

        # Compute sample weights (heavy weight on intentions)
        sample_weights_1 = None
        if use_sample_weights:
            sample_weights_1 = compute_sample_weight('balanced', y_binary)

        self.detector = XGBClassifier(**self.detector_params)

        eval_set_1 = None
        if X_val is not None and y_val is not None:
            y_val_binary = self._convert_to_binary(y_val)
            eval_set_1 = [(X_val, y_val_binary)]

        if verbose:
            n_est = self.detector_params.get('n_estimators', 300)
            print(f"  Training detector ({n_est} estimators)...")

        self.detector.fit(
            X, y_binary,
            sample_weight=sample_weights_1,
            eval_set=eval_set_1,
            verbose=50 if verbose else False  # Print every 50 rounds
        )

        if verbose:
            y_pred_binary = self.detector.predict(X)
            det_recall = recall_score(y_binary, y_pred_binary)
            det_precision = precision_score(y_binary, y_pred_binary)
            print(f"  Train Detection - Precision: {det_precision:.4f}, Recall: {det_recall:.4f}")
            print(f"  Intentions in train: {y_binary.sum()} / {len(y_binary)} ({100*y_binary.mean():.1f}%)")

        # ===== Stage 2: Train Direction Classifier =====
        if verbose:
            print("\n--- Stage 2: Direction Classifier (upward vs downward) ---")

        intention_mask, y_direction = self._convert_to_direction(y)
        X_intentions = X[intention_mask]

        if verbose:
            print(f"  Training on {len(X_intentions)} intention samples")
            for i, cls in enumerate(self.intention_classes):
                count = (y_direction == i).sum()
                print(f"    Class {cls}: {count} samples")

        # Sample weights for Stage 2 (if classes are imbalanced)
        sample_weights_2 = None
        if use_sample_weights and len(np.unique(y_direction)) > 1:
            sample_weights_2 = compute_sample_weight('balanced', y_direction)

        self.classifier = XGBClassifier(**self.classifier_params)

        eval_set_2 = None
        if X_val is not None and y_val is not None:
            val_mask, y_val_direction = self._convert_to_direction(y_val)
            if val_mask.sum() > 0:
                eval_set_2 = [(X_val[val_mask], y_val_direction)]

        if verbose:
            n_est = self.classifier_params.get('n_estimators', 200)
            print(f"  Training classifier ({n_est} estimators)...")

        self.classifier.fit(
            X_intentions, y_direction,
            sample_weight=sample_weights_2,
            eval_set=eval_set_2,
            verbose=50 if verbose else False  # Print every 50 rounds
        )

        if verbose:
            y_pred_dir = self.classifier.predict(X_intentions)
            dir_acc = accuracy_score(y_direction, y_pred_dir)
            print(f"  Train Direction Accuracy: {dir_acc:.4f}")

        if verbose:
            print("\nTraining complete.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for all 3 classes.

        P(noise) = P(not intention)
        P(class_i) = P(intention) * P(class_i | intention)

        Args:
            X: Features (N, D)

        Returns:
            Probabilities (N, num_classes) where num_classes = 1 + len(intention_classes)
        """
        n_samples = len(X)
        num_classes = 1 + len(self.intention_classes)

        # Stage 1: P(intention)
        p_intention = self.detector.predict_proba(X)[:, 1]
        p_noise = 1 - p_intention

        # Stage 2: P(direction | intention)
        p_direction_given_intention = self.classifier.predict_proba(X)

        # Combine probabilities
        proba = np.zeros((n_samples, num_classes))
        proba[:, self.noise_class] = p_noise

        for i, cls in enumerate(self.intention_classes):
            proba[:, cls] = p_intention * p_direction_given_intention[:, i]

        return proba

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict class labels with optional custom threshold.

        Args:
            X: Features (N, D)
            threshold: Detection threshold (default: self.detection_threshold)

        Returns:
            Predicted labels (N,)
        """
        if threshold is None:
            threshold = self.detection_threshold

        n_samples = len(X)

        # Stage 1: Detect intentions
        p_intention = self.detector.predict_proba(X)[:, 1]
        is_intention = p_intention >= threshold

        # Default to noise
        predictions = np.full(n_samples, self.noise_class, dtype=int)

        # Stage 2: Classify detected intentions
        if is_intention.any():
            direction_preds = self.classifier.predict(X[is_intention])
            # Map back to original class labels
            for i, cls in enumerate(self.intention_classes):
                predictions[is_intention] = np.where(
                    direction_preds == i,
                    cls,
                    predictions[is_intention]
                )
            # Simpler: direct mapping
            predictions[is_intention] = np.array(self.intention_classes)[direction_preds]

        return predictions

    def tune_detection_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_metric: str = 'recall',
        target_value: float = 0.95,
        verbose: bool = True
    ) -> Dict:
        """
        Tune detection threshold on validation data.

        Args:
            X: Validation features
            y: Validation labels
            target_metric: 'recall', 'precision', 'f1', or 'balanced'
            target_value: Target value for recall/precision (e.g., 0.95)
            verbose: Print results

        Returns:
            Dict with threshold and metrics at that threshold
        """
        y_binary = self._convert_to_binary(y)
        p_intention = self.detector.predict_proba(X)[:, 1]

        # Get all unique thresholds
        thresholds = np.sort(np.unique(p_intention))

        best_threshold = 0.5
        best_score = -1
        results_at_best = {}

        for thresh in thresholds:
            pred_binary = (p_intention >= thresh).astype(int)

            precision = precision_score(y_binary, pred_binary, zero_division=0)
            recall = recall_score(y_binary, pred_binary, zero_division=0)
            f1 = f1_score(y_binary, pred_binary, zero_division=0)

            if target_metric == 'recall' and recall >= target_value:
                # Among thresholds achieving target recall, pick highest precision
                if precision > best_score:
                    best_score = precision
                    best_threshold = thresh
                    results_at_best = {'precision': precision, 'recall': recall, 'f1': f1}

            elif target_metric == 'precision' and precision >= target_value:
                # Among thresholds achieving target precision, pick highest recall
                if recall > best_score:
                    best_score = recall
                    best_threshold = thresh
                    results_at_best = {'precision': precision, 'recall': recall, 'f1': f1}

            elif target_metric == 'f1':
                if f1 > best_score:
                    best_score = f1
                    best_threshold = thresh
                    results_at_best = {'precision': precision, 'recall': recall, 'f1': f1}

            elif target_metric == 'balanced':
                balance = 1 - abs(precision - recall)
                if balance > best_score:
                    best_score = balance
                    best_threshold = thresh
                    results_at_best = {'precision': precision, 'recall': recall, 'f1': f1}

        # Fallback if target not achieved
        if not results_at_best:
            # Use threshold closest to target
            if target_metric == 'recall':
                recalls = [recall_score(y_binary, (p_intention >= t).astype(int), zero_division=0)
                           for t in thresholds]
                best_idx = np.argmin(np.abs(np.array(recalls) - target_value))
            else:
                best_idx = len(thresholds) // 2
            best_threshold = thresholds[best_idx]
            pred_binary = (p_intention >= best_threshold).astype(int)
            results_at_best = {
                'precision': precision_score(y_binary, pred_binary, zero_division=0),
                'recall': recall_score(y_binary, pred_binary, zero_division=0),
                'f1': f1_score(y_binary, pred_binary, zero_division=0)
            }

        self.detection_threshold = best_threshold

        if verbose:
            print(f"\nThreshold tuning ({target_metric} >= {target_value}):")
            print(f"  Optimal threshold: {best_threshold:.4f}")
            print(f"  Precision: {results_at_best['precision']:.4f}")
            print(f"  Recall: {results_at_best['recall']:.4f}")
            print(f"  F1: {results_at_best['f1']:.4f}")

        return {
            'threshold': best_threshold,
            **results_at_best,
            'target_metric': target_metric,
            'target_value': target_value
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation of both stages.

        Args:
            X: Test features
            y: Test labels
            class_names: Names for classes (default: ['noise', 'upward', 'downward'])
            verbose: Print detailed report

        Returns:
            Dict with metrics for both stages and combined
        """
        if class_names is None:
            class_names = ['noise', 'upward', 'downward']

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Binary detection metrics
        y_binary = self._convert_to_binary(y)
        y_pred_binary = self._convert_to_binary(y_pred)
        p_intention = self.detector.predict_proba(X)[:, 1]

        detection_metrics = {
            'precision': precision_score(y_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_binary, y_pred_binary, zero_division=0),
            'ap': average_precision_score(y_binary, p_intention),
            'threshold': self.detection_threshold
        }

        # Direction classification metrics (on true intentions only)
        intention_mask = y != self.noise_class
        direction_metrics = {}

        if intention_mask.sum() > 0:
            _, y_direction_true = self._convert_to_direction(y)
            direction_preds_on_intentions = self.classifier.predict(X[intention_mask])

            direction_metrics = {
                'accuracy': accuracy_score(y_direction_true, direction_preds_on_intentions),
                'n_samples': int(intention_mask.sum())
            }

        # Combined 3-class metrics
        combined_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'f1_per_class': f1_score(y, y_pred, average=None, zero_division=0).tolist()
        }

        # Minority F1
        minority_f1 = np.mean([combined_metrics['f1_per_class'][c] for c in self.intention_classes])
        combined_metrics['minority_f1'] = minority_f1

        results = {
            'detection': detection_metrics,
            'direction': direction_metrics,
            'combined': combined_metrics
        }

        if verbose:
            print("\n" + "=" * 60)
            print("TWO-STAGE CLASSIFIER EVALUATION")
            print("=" * 60)

            print("\n--- Stage 1: Intention Detection ---")
            print(f"  Threshold: {detection_metrics['threshold']:.4f}")
            print(f"  Precision: {detection_metrics['precision']:.4f}")
            print(f"  Recall: {detection_metrics['recall']:.4f}")
            print(f"  F1: {detection_metrics['f1']:.4f}")
            print(f"  AP (PR-AUC): {detection_metrics['ap']:.4f}")

            if direction_metrics:
                print("\n--- Stage 2: Direction Classification ---")
                print(f"  Accuracy (on {direction_metrics['n_samples']} true intentions): "
                      f"{direction_metrics['accuracy']:.4f}")

            print("\n--- Combined 3-Class Performance ---")
            print(f"  Accuracy: {combined_metrics['accuracy']:.4f}")
            print(f"  F1 Macro: {combined_metrics['f1_macro']:.4f}")
            print(f"  Minority F1: {combined_metrics['minority_f1']:.4f}")

            print("\n  Per-class F1:")
            for i, name in enumerate(class_names):
                print(f"    {name}: {combined_metrics['f1_per_class'][i]:.4f}")

            print("\n  Classification Report:")
            print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

        return results

    def save(self, path: str):
        """Save both models and configuration."""
        import json

        # Save detector
        self.detector.save_model(f"{path}_detector.json")

        # Save classifier
        self.classifier.save_model(f"{path}_classifier.json")

        # Helper to convert numpy types to Python types for JSON
        def to_python(obj):
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Save config
        config = {
            'detection_threshold': float(self.detection_threshold),
            'noise_class': int(self.noise_class),
            'intention_classes': [int(c) for c in self.intention_classes],
            'detector_params': to_python(self.detector_params),
            'classifier_params': to_python(self.classifier_params)
        }
        with open(f"{path}_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TwoStageClassifier':
        """Load saved two-stage classifier."""
        import json

        # Load config
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            detector_params=config['detector_params'],
            classifier_params=config['classifier_params'],
            detection_threshold=config['detection_threshold'],
            noise_class=config['noise_class']
        )
        instance.intention_classes = config['intention_classes']

        # Load models
        instance.detector = XGBClassifier()
        instance.detector.load_model(f"{path}_detector.json")

        instance.classifier = XGBClassifier()
        instance.classifier.load_model(f"{path}_classifier.json")

        return instance
