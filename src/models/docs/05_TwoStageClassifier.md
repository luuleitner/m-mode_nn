# TwoStageClassifier (Hierarchical XGBoost)

## Overview
Two-stage hierarchical classifier optimized for extreme class imbalance (90/5/5). Separates the problem into detection (noise vs intention) and classification (upward vs downward).

**Key Features:**
- Stage 1: High-recall intention detection
- Stage 2: Direction classification (only on detected intentions)
- Tunable detection threshold
- Optimized for minority class recall

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TwoStageClassifier Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Embeddings (N, 512) or raw features                                  │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                   STAGE 1: INTENTION DETECTION                               │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    Binary Classification: Noise (0) vs Intention (1)                    ││
│  │                                                                          ││
│  │    Labels mapping:                                                       ││
│  │      noise (0)    → 0 (negative)                                        ││
│  │      upward (1)   → 1 (positive)                                        ││
│  │      downward (2) → 1 (positive)                                        ││
│  │                                                                          ││
│  │    XGBoost Detector:                                                    ││
│  │      objective: binary:logistic                                         ││
│  │      eval_metric: aucpr (for imbalanced data)                           ││
│  │      sample_weight: balanced (heavy on intentions)                      ││
│  │                                                                          ││
│  │    Output: P(intention) ∈ [0, 1]                                        ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                   ┌──────────────────────┐                                   │
│                   │  Detection Threshold  │                                  │
│                   │  (tunable, e.g., 0.3) │                                  │
│                   └──────────┬───────────┘                                   │
│                              │                                               │
│              ┌───────────────┴───────────────┐                               │
│              │                               │                               │
│    P(intention) < threshold       P(intention) >= threshold                  │
│              │                               │                               │
│              ▼                               ▼                               │
│       ┌──────────┐                  ┌───────────────┐                        │
│       │  NOISE   │                  │   STAGE 2     │                        │
│       │  (done)  │                  │   (continue)  │                        │
│       └──────────┘                  └───────┬───────┘                        │
│                                             │                                │
│  ════════════════════════════════════════════════════════════════════════   │
│                   STAGE 2: DIRECTION CLASSIFICATION                          │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    Binary Classification: Upward (0) vs Downward (1)                    ││
│  │                                                                          ││
│  │    ONLY trained on intention samples (no noise)                         ││
│  │    ONLY run on detected intentions at inference                         ││
│  │                                                                          ││
│  │    Labels mapping:                                                       ││
│  │      upward (1)   → 0                                                   ││
│  │      downward (2) → 1                                                   ││
│  │                                                                          ││
│  │    XGBoost Classifier:                                                  ││
│  │      objective: binary:logistic                                         ││
│  │      eval_metric: logloss                                               ││
│  │      sample_weight: balanced (if needed)                                ││
│  │                                                                          ││
│  │    Output: P(downward | intention) ∈ [0, 1]                             ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                   ┌──────────────────────┐                                   │
│                   │   Direction Decision  │                                  │
│                   │  P(down) >= 0.5 ?     │                                  │
│                   └──────────┬───────────┘                                   │
│                              │                                               │
│              ┌───────────────┴───────────────┐                               │
│              │                               │                               │
│              ▼                               ▼                               │
│       ┌──────────┐                    ┌──────────┐                           │
│       │  UPWARD  │                    │ DOWNWARD │                           │
│       │   (1)    │                    │   (2)    │                           │
│       └──────────┘                    └──────────┘                           │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                      COMBINED PROBABILITIES                                  │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│    P(noise)    = 1 - P(intention)                                            │
│    P(upward)   = P(intention) × P(upward | intention)                        │
│    P(downward) = P(intention) × P(downward | intention)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Threshold Tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DETECTION THRESHOLD TUNING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  The detection threshold controls the trade-off between:                     │
│    - Precision: % of detected intentions that are true intentions            │
│    - Recall: % of true intentions that are detected                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Threshold    Recall    Precision    Effect                             ││
│  │  ─────────    ──────    ─────────    ──────                             ││
│  │    0.1        99%       10%          Almost everything detected         ││
│  │    0.3        95%       25%          High recall, moderate precision    ││
│  │    0.5        85%       50%          Balanced (default)                 ││
│  │    0.7        60%       75%          Fewer detections, more accurate    ││
│  │    0.9        20%       95%          Very conservative                  ││
│  │                                                                          ││
│  │  For 90/5/5 imbalance, recommended: threshold = 0.2-0.4                 ││
│  │  Prioritize recall (don't miss intentions)                              ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Tuning methods:                                                             │
│                                                                              │
│    1. Target recall (recommended):                                           │
│       clf.tune_detection_threshold(X_val, y_val, target_metric='recall',     │
│                                    target_value=0.95)                        │
│       → Find threshold that achieves ≥95% recall with best precision         │
│                                                                              │
│    2. Target precision:                                                      │
│       clf.tune_detection_threshold(X_val, y_val, target_metric='precision',  │
│                                    target_value=0.80)                        │
│       → Find threshold that achieves ≥80% precision with best recall         │
│                                                                              │
│    3. Maximize F1:                                                           │
│       clf.tune_detection_threshold(X_val, y_val, target_metric='f1')         │
│       → Find threshold that maximizes F1 score                               │
│                                                                              │
│    4. Balanced:                                                              │
│       clf.tune_detection_threshold(X_val, y_val, target_metric='balanced')   │
│       → Find threshold with smallest gap between precision and recall        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Two Stages?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MOTIVATION FOR TWO STAGES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROBLEM with single 3-class classifier:                                     │
│                                                                              │
│    Class distribution: 90% noise, 5% upward, 5% downward                     │
│                                                                              │
│    → Classifier tends to predict everything as noise                         │
│    → Even with class weights, minority classes underperform                  │
│    → Hard to tune for different detection/classification goals               │
│                                                                              │
│  SOLUTION with two stages:                                                   │
│                                                                              │
│    Stage 1: 90% vs 10%   (noise vs intention)                                │
│      - More balanced binary problem                                          │
│      - Can tune threshold for desired recall                                 │
│      - Optimized for "don't miss intentions"                                 │
│                                                                              │
│    Stage 2: 50% vs 50%   (upward vs downward, among intentions)              │
│      - Perfectly balanced                                                    │
│      - No class imbalance to worry about                                     │
│      - Focus purely on direction discrimination                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Single 3-class:     Two-stage:                                         ││
│  │                                                                          ││
│  │  ┌─────────┐         ┌─────────┐                                        ││
│  │  │ 90/5/5  │    vs   │  90/10  │ → ┌─────────┐                          ││
│  │  │ heavily │         │balanced │   │  50/50  │                          ││
│  │  │ skewed  │         │ binary  │   │ perfect │                          ││
│  │  └─────────┘         └─────────┘   └─────────┘                          ││
│  │                                                                          ││
│  │  Hard to optimize     Easy to optimize both stages independently         ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

```python
from src.models.two_stage_classifier import TwoStageClassifier

# Create classifier
clf = TwoStageClassifier(
    detection_threshold=0.5,
    noise_class=0
)

# Train both stages
clf.fit(X_train, y_train, X_val, y_val, verbose=True)

# Tune threshold for high recall
clf.tune_detection_threshold(
    X_val, y_val,
    target_metric='recall',
    target_value=0.95
)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate
results = clf.evaluate(X_test, y_test)

# Save/load
clf.save('two_stage_model')
clf2 = TwoStageClassifier.load('two_stage_model')
```

---

## Configuration

```yaml
# XGBoost parameters for detector
detector_params:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.1
  eval_metric: 'aucpr'   # PR-AUC for imbalanced detection

# XGBoost parameters for classifier
classifier_params:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.1
  eval_metric: 'logloss'
```

---

## Output Metrics

```
════════════════════════════════════════════════════════════
TWO-STAGE CLASSIFIER EVALUATION
════════════════════════════════════════════════════════════

--- Stage 1: Intention Detection ---
  Threshold: 0.3500
  Precision: 0.4521
  Recall: 0.9523
  F1: 0.6127
  AP (PR-AUC): 0.5834

--- Stage 2: Direction Classification ---
  Accuracy (on 200 true intentions): 0.8500

--- Combined 3-Class Performance ---
  Accuracy: 0.8912
  F1 Macro: 0.6234
  Minority F1: 0.5123    ← Average F1 of upward + downward

  Per-class F1:
    noise: 0.9456
    upward: 0.4892
    downward: 0.5354
```

---

## When to Use

**Use TwoStageClassifier when:**
- Extreme class imbalance (>10:1 ratio)
- Missing minority samples is costly
- You need tunable detection threshold
- You want to optimize detection and classification separately

**Don't use when:**
- Classes are relatively balanced
- Simple classifier performs well
- You need end-to-end neural network training
