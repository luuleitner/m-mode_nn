# Autoencoder + XGBoost Pipeline

## Overview
Two-stage pipeline where an autoencoder learns embeddings (without classification objective), then XGBoost classifies based on extracted embeddings.

**Key Features:**
- Decoupled training (AE for representation, XGBoost for classification)
- Interpretable: feature importance from XGBoost
- Flexible: easy to swap classifiers
- No gradient conflict between objectives

**Limitation:**
- AE embeddings may not be discriminative (optimized for reconstruction)
- Two separate training stages

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AE + XGBoost PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                       STAGE 1: TRAIN AUTOENCODER                             │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    Input ──→ Encoder ──→ Embedding (512) ──→ Decoder ──→ Reconstruction ││
│  │                                                                          ││
│  │    Loss = MSE(input, reconstruction) + L1(input, reconstruction)        ││
│  │                                                                          ││
│  │    NO CLASSIFICATION LOSS - pure reconstruction objective               ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                              │                                               │
│                              │ Save trained encoder                          │
│                              ▼                                               │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                       STAGE 2: EXTRACT EMBEDDINGS                            │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    For each sample in train/val/test:                                   ││
│  │                                                                          ││
│  │        Input ──→ Encoder (frozen) ──→ Embedding (512)                   ││
│  │                                                                          ││
│  │    Save to embeddings.npz:                                              ││
│  │        X_train: (N_train, 512)                                          ││
│  │        X_val:   (N_val, 512)                                            ││
│  │        X_test:  (N_test, 512)                                           ││
│  │        y_train, y_val, y_test: labels                                   ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                              │                                               │
│                              │ Load embeddings                               │
│                              ▼                                               │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                       STAGE 3: TRAIN XGBOOST                                 │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    ┌────────────────────────────────────────────────────────────────┐   ││
│  │    │                     XGBoost Classifier                          │   ││
│  │    │                                                                 │   ││
│  │    │    X_train (512 features)                                      │   ││
│  │    │         │                                                       │   ││
│  │    │         ▼                                                       │   ││
│  │    │    ┌─────────┐   ┌─────────┐   ┌─────────┐                     │   ││
│  │    │    │  Tree 1 │ + │  Tree 2 │ + │  ...    │ + ... (300 trees)   │   ││
│  │    │    └─────────┘   └─────────┘   └─────────┘                     │   ││
│  │    │         │                                                       │   ││
│  │    │         ▼                                                       │   ││
│  │    │    Predicted Class (0, 1, or 2)                                │   ││
│  │    │                                                                 │   ││
│  │    └────────────────────────────────────────────────────────────────┘   ││
│  │                                                                          ││
│  │    Loss: multi:softmax with sample weights                              ││
│  │    Early stopping on validation set                                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                       INFERENCE                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│    Input ──→ Encoder ──→ Embedding ──→ XGBoost ──→ Predicted Class          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## XGBoost Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      XGBOOST HYPERPARAMETERS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  From config/config.yaml:                                                    │
│                                                                              │
│  xgboost:                                                                    │
│    n_estimators: 300           # Number of boosting rounds (trees)           │
│    max_depth: 6                # Max tree depth                              │
│    learning_rate: 0.05         # Shrinkage factor                            │
│    subsample: 0.8              # Row sampling per tree                       │
│    colsample_bytree: 0.8       # Feature sampling per tree                   │
│    reg_alpha: 0.1              # L1 regularization                           │
│    reg_lambda: 1.0             # L2 regularization                           │
│    min_child_weight: 3         # Minimum samples in leaf                     │
│    gamma: 0.1                  # Min loss reduction for split                │
│    early_stopping_rounds: 30   # Stop if no improvement for 30 rounds        │
│                                                                              │
│  Class Imbalance Handling:                                                   │
│    - sample_weight: balanced (inverse frequency)                             │
│    - scale_pos_weight for binary stages                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding Preprocessing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PREPROCESSING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Before feeding to XGBoost, embeddings are optionally preprocessed:          │
│                                                                              │
│  preprocessing:                                                              │
│    normalize_embeddings: true                                                │
│    scaler: "standard"          # standard | minmax | none                    │
│                                                                              │
│  Standard Scaler (recommended):                                              │
│    x' = (x - mean) / std                                                     │
│    - Centers features at 0                                                   │
│    - Scales to unit variance                                                 │
│    - Helps XGBoost splits                                                    │
│                                                                              │
│  MinMax Scaler:                                                              │
│    x' = (x - min) / (max - min)                                              │
│    - Scales to [0, 1] range                                                  │
│    - Sensitive to outliers                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Importance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE IMPORTANCE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  XGBoost provides interpretable feature importance:                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Embedding Dimension    Importance Score                                 ││
│  │  ───────────────────    ────────────────                                 ││
│  │  dim_127                0.0842                                           ││
│  │  dim_256                0.0721                                           ││
│  │  dim_89                 0.0654                                           ││
│  │  dim_312                0.0598                                           ││
│  │  ...                    ...                                              ││
│  │                                                                          ││
│  │  This tells us which embedding dimensions are most discriminative        ││
│  │  for classification.                                                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Types of importance:                                                        │
│    - weight: # times feature used in splits                                  │
│    - gain: average gain from splits using feature                            │
│    - cover: average samples affected by splits                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

```bash
# STAGE 1: Train autoencoder (classification_weight: 0)
# Set in config: classification_weight: 0
python -m src.training.train_cnn_ae --config config/config.yaml

# STAGE 2: Extract embeddings
python -m src.training.extract_embeddings \
    --config config/config.yaml \
    --checkpoint path/to/ae_model.pth \
    --output path/to/embeddings.npz

# STAGE 3: Train XGBoost classifier
python -m src.training.train_classifier \
    --config config/config.yaml \
    --embeddings path/to/embeddings.npz

# Evaluate
python -m src.evaluation.evaluate_xgb \
    --embeddings path/to/embeddings.npz \
    --model path/to/xgb_model.json
```

---

## Problem: Non-Discriminative Embeddings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE EMBEDDING QUALITY PROBLEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  When AE is trained only for reconstruction:                                 │
│                                                                              │
│    - Embeddings capture reconstruction-friendly features                     │
│    - Smooth, average patterns preserved                                      │
│    - Class-specific discriminative features may be lost                      │
│                                                                              │
│  Typical result:                                                             │
│    Fisher Separability Score: 0.003 (very low)                               │
│    XGBoost accuracy: ~87% (predicts mostly noise)                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    t-SNE Visualization of Embeddings:                                   ││
│  │                                                                          ││
│  │    Without classification loss:    With classification loss:            ││
│  │                                                                          ││
│  │         • • • • • • •              •••           ○○○                     ││
│  │       •   ○ • ○ • ○   •              •••       ○○○○○                     ││
│  │      • ○ △ • △ • △ ○ •                •••     ○○○                       ││
│  │       •   ○ • ○ • ○   •                          △△△                    ││
│  │         • • • • • • •                           △△△△△                   ││
│  │                                                 △△△                      ││
│  │    (classes overlapped)            (classes separated)                  ││
│  │                                                                          ││
│  │    • noise   ○ upward   △ downward                                      ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  SOLUTION:                                                                   │
│    1. Use joint training with classification_weight > 0                      │
│    2. Or use DirectCNNClassifier instead                                     │
│    3. Or use TwoStageClassifier with tuned threshold                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use

**Use AE + XGBoost when:**
- You need interpretable feature importance
- You want to experiment with different classifiers on same embeddings
- You prefer decoupled training stages
- You have resources for hyperparameter tuning

**Don't use when:**
- Maximum classification accuracy is needed (use DirectCNN)
- Embeddings are not discriminative (Fisher score < 0.1)
- You want end-to-end training
