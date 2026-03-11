# M-Mode Neural Network: Architectures & Pipelines

## Overview

This document describes all implemented neural network architectures and training pipelines
for ultrasound M-mode signal classification.

---

## 1. Network Architectures

### 1.1 Direct CNN Classifier (USMModeCNN-style)

End-to-end classification with 3-block architecture and global average pooling.
Inspired by colleague's USMModeCNN, adapted for 130-depth decimated input.

**File:** `src/models/direct_cnn_classifier.py`

**Key Design:**
- Same-padding preserves edge information
- Progressive depth reduction with temporal preservation
- Global average pooling for position-invariant features
- ~25K parameters (vs ~2.5M in flatten-based approach)

```
Input: [B, 3, Pulses(10), Depth(130)]
       │
       │  3 ultrasound channels
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      CONV BLOCK 1                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Conv2d(3 → 16, kernel=(3, 13), padding=(1, 6))          │ │
│  │ BatchNorm2d(16)                                         │ │
│  │ ReLU                                                    │ │
│  │ MaxPool2d(1, 2)                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Output: [B, 16, 10, 65]  (depth halved)                     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      CONV BLOCK 2                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Conv2d(16 → 32, kernel=(3, 7), padding=(1, 3))          │ │
│  │ BatchNorm2d(32)                                         │ │
│  │ ReLU                                                    │ │
│  │ MaxPool2d(1, 2)                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Output: [B, 32, 10, 32]  (depth halved)                     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      CONV BLOCK 3                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Conv2d(32 → 64, kernel=(3, 5), padding=(1, 2))          │ │
│  │ BatchNorm2d(64)                                         │ │
│  │ ReLU                                                    │ │
│  │ MaxPool2d(2, 2)                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Output: [B, 64, 5, 16]  (both halved)                       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   GLOBAL AVERAGE POOLING                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ AdaptiveAvgPool2d((1, 1))                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Output: [B, 64, 1, 1] → [B, 64]  (position-invariant)       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION HEAD                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Linear(64 → 64)                                         │ │
│  │ ReLU                                                    │ │
│  │ Dropout(0.3)                                            │ │
│  │ Linear(64 → 3)                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
Output: [B, 3] logits (noise, upward, downward)
```

---

### 1.2 UNet Autoencoder (with optional Classification Head)

**File:** `src/models/unet_ae.py`

U-Net architecture with skip connections for better reconstruction quality.
Preserves fine details through concatenation-based skip connections.

```
Input: [B, 3, H, W]
       │
       ▼
┌══════════════════════════════════════════════════════════════════════════════┐
║                         ENCODER (with skip connections)                      ║
║                                                                              ║
║  Level 0: [B, 3, H, W]                                                       ║
║      │                                                                       ║
║      ▼                                                                       ║
║  ┌──────────────────┐                                                        ║
║  │ EncBlock(3→32)   │──────────────────────────────────────────┐ skip_0      ║
║  │ 2xConv + LeakyReLU│                                         │             ║
║  │ MaxPool(2)       │                                          │             ║
║  └──────────────────┘                                          │             ║
║      │ [B, 32, H/2, W/2]                                       │             ║
║      ▼                                                         │             ║
║  ┌──────────────────┐                                          │             ║
║  │ EncBlock(32→64)  │────────────────────────────────┐ skip_1  │             ║
║  └──────────────────┘                                │         │             ║
║      │ [B, 64, H/4, W/4]                             │         │             ║
║      ▼                                               │         │             ║
║  ┌──────────────────┐                                │         │             ║
║  │ EncBlock(64→128) │──────────────────────┐ skip_2  │         │             ║
║  └──────────────────┘                      │         │         │             ║
║      │ [B, 128, H/8, W/8]                  │         │         │             ║
║      ▼                                     │         │         │             ║
║  ┌──────────────────┐                      │         │         │             ║
║  │ EncBlock(128→256)│────────────┐ skip_3  │         │         │             ║
║  └──────────────────┘            │         │         │         │             ║
║      │ [B, 256, H/16, W/16]      │         │         │         │             ║
╚══════════════════════════════════│═════════│═════════│═════════│═════════════╝
       │                           │         │         │         │
       ▼                           │         │         │         │
┌──────────────────────────┐       │         │         │         │
│      BOTTLENECK          │       │         │         │         │
│  Flatten → FC(→512)      │       │         │         │         │
│  [B, 512] embedding      │───────┼─────────┼─────────┼─────────┼──▶ Classification
│  FC(512→) → Unflatten    │       │         │         │         │    Head (optional)
└──────────────────────────┘       │         │         │         │
       │                           │         │         │         │
       ▼                           │         │         │         │
┌══════════════════════════════════│═════════│═════════│═════════│═════════════┐
║                         DECODER (with skip connections)                      ║
║      │                           │         │         │         │             ║
║      ▼                           ▼         │         │         │             ║
║  ┌──────────────────────────────────┐      │         │         │             ║
║  │ DecBlock: ConvT(256→128)         │      │         │         │             ║
║  │ Concat with skip_3 → [B,256,H/8] │◄─────┘         │         │             ║
║  │ 2xConv(256→128)                  │                │         │             ║
║  └──────────────────────────────────┘                │         │             ║
║      │                                               │         │             ║
║      ▼                                               ▼         │             ║
║  ┌──────────────────────────────────┐                          │             ║
║  │ DecBlock: ConvT(128→64)          │                          │             ║
║  │ Concat with skip_2 → [B,128,H/4] │◄─────────────────────────┘             ║
║  │ 2xConv(128→64)                   │                                        ║
║  └──────────────────────────────────┘                                        ║
║      │                                                         │             ║
║      ▼                                                         ▼             ║
║  ┌──────────────────────────────────┐                                        ║
║  │ DecBlock: ConvT(64→32)           │                                        ║
║  │ Concat with skip_1 → [B,64,H/2]  │◄───────────────────────────────────────┘
║  │ 2xConv(64→32)                    │                                        ║
║  └──────────────────────────────────┘                          │             ║
║      │                                                         ▼             ║
║      ▼                                                                       ║
║  ┌──────────────────────────────────┐                                        ║
║  │ DecBlock: ConvT(32→16)           │                                        ║
║  │ Concat with skip_0 → [B,32,H]    │◄───────────────────────────────────────┘
║  │ 2xConv(32→16)                    │                                        ║
║  │ Final Conv(16→3)                 │                                        ║
║  └──────────────────────────────────┘                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
       │
       ▼
Output: (reconstruction [B,3,H,W], embedding [B,512], logits [B,3])
```

**Classification Head (optional):**
```
Embedding [B, 512]
       │
       ▼
┌──────────────────────────────────┐
│ Linear(512 → 256)                │
│ ReLU                             │
│ Dropout(0.3)                     │
│ Linear(256 → 3)                  │
└──────────────────────────────────┘
       │
       ▼
Logits [B, 3]
```

---

### 1.3 XGBoost Classifiers (on embeddings)

#### Standard XGBoost
**File:** `src/training/train_xgb_cls.py`

```
Embeddings: [N, embedding_dim]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    XGBClassifier                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ n_estimators: 300                                      │  │
│  │ max_depth: 6                                           │  │
│  │ learning_rate: 0.05                                    │  │
│  │ subsample: 0.8                                         │  │
│  │ colsample_bytree: 0.8                                  │  │
│  │ early_stopping_rounds: 30                              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
Output: [N, 3] class probabilities
```

#### TwoStageClassifier (Hierarchical)
**File:** `src/models/two_stage_classifier.py`

```
Embeddings: [N, embedding_dim]
       │
       ▼
┌══════════════════════════════════════════════════════════════┐
║                    STAGE 1: DETECTION                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Binary XGBoost: Noise vs Intention                     │  ║
║  │ Labels: noise=0, (upward|downward)=1                   │  ║
║  │ Optimized for: PR-AUC (imbalanced data)                │  ║
║  │ Tunable threshold for recall optimization              │  ║
║  └────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════╝
       │
       │ P(intention) >= threshold
       ▼
┌══════════════════════════════════════════════════════════════┐
║                  STAGE 2: CLASSIFICATION                     ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Binary XGBoost: Upward vs Downward                     │  ║
║  │ Trained ONLY on intention samples                      │  ║
║  │ Optimized for: Log-loss                                │  ║
║  └────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════╝
       │
       ▼
Output: [N] predictions (noise=0, upward=1, downward=2)
```

---

## 2. Training Pipelines

### Pipeline Option 1: Direct CNN (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE 1: DIRECT CNN CLASSIFIER                       │
│                                                                             │
│  Training Script: src/training/train_cnn_cls.py                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐              │
│  │   Raw H5    │───▶│ DataLoader  │───▶│ DirectCNNClassifier │              │
│  │   Files     │    │  [B,3,10,130]│    │                     │              │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘              │
│                                                   │                         │
│                                                   ▼                         │
│                                        ┌─────────────────────┐              │
│                                        │  CrossEntropy Loss  │              │
│                                        │  (class-weighted)   │              │
│                                        │  or Focal Loss      │              │
│                                        └──────────┬──────────┘              │
│                                                   │                         │
│                                                   ▼                         │
│                                        ┌─────────────────────┐              │
│                                        │   Backpropagation   │              │
│                                        │   AdamW Optimizer   │              │
│                                        └──────────┬──────────┘              │
│                                                   │                         │
│                                                   ▼                         │
│                                        ┌─────────────────────┐              │
│                                        │  Checkpoint + Plots │              │
│                                        │  WandB Logging      │              │
│                                        └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Pipeline Option 2: UNet AE + Classification Head (Joint Training)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               PIPELINE 2: AUTOENCODER + CLASSIFICATION HEAD                 │
│                                                                             │
│  Training Script: src/training/train_unet_ae.py                              │
│  Config: classification_weight > 0                                          │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐              │
│  │   Raw H5    │───▶│ DataLoader  │───▶│   UNetAutoencoder   │              │
│  │   Files     │    │             │    │   + CLS Head        │              │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘              │
│                                                   │                         │
│                                     ┌─────────────┼─────────────┐           │
│                                     │             │             │           │
│                                     ▼             ▼             ▼           │
│                              ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│                              │  Recon    │ │ Embedding │ │  Logits   │      │
│                              │ [B,3,H,W] │ │  [B,512]  │ │  [B,3]    │      │
│                              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      │
│                                    │             │             │            │
│                                    ▼             ▼             ▼            │
│                              ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│                              │ MSE + L1  │ │  L2 Reg   │ │   CE      │      │
│                              │   Loss    │ │   Loss    │ │   Loss    │      │
│                              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      │
│                                    │             │             │            │
│                                    └──────┬──────┴──────┬──────┘            │
│                                           │             │                   │
│                                           ▼             │                   │
│                              ┌────────────────────────┐ │                   │
│                              │      JOINT LOSS        │ │                   │
│                              │ 0.3*MSE + 0.3*L1 +     │◀┘                   │
│                              │ 0.0005*Reg + 0.4*CE    │                     │
│                              └───────────┬────────────┘                     │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌────────────────────────┐                     │
│                              │    Backpropagation     │                     │
│                              └────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Pipeline Option 3: UNet AE → XGBoost (Two-Stage)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE 3: AE → XGBoost                               │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 1: Train Autoencoder                                            ║  │
│  ║  Script: src/training/train_unet_ae.py                                 ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────┐    ┌──────────────────┐    ┌──────────────┐              ║  │
│  ║  │ Raw H5  │───▶│ UNetAutoencoder  │───▶│  Checkpoint  │              ║  │
│  ║  │ Files   │    │ (recon loss only)│    │  best.pt     │              ║  │
│  ║  └─────────┘    └──────────────────┘    └──────────────┘              ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 2: Extract Embeddings                                           ║  │
│  ║  Script: src/training/extract_embeddings.py                           ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐          ║  │
│  ║  │ Checkpoint  │───▶│ Forward pass     │───▶│ embeddings   │          ║  │
│  ║  │ + Raw H5    │    │ (encoder only)   │    │ .npz file    │          ║  │
│  ║  └─────────────┘    └──────────────────┘    └──────────────┘          ║  │
│  ║                                                                       ║  │
│  ║  embeddings.npz contains:                                             ║  │
│  ║    X_train: [N_train, 512]    y_train: [N_train]                      ║  │
│  ║    X_val:   [N_val, 512]      y_val:   [N_val]                        ║  │
│  ║    X_test:  [N_test, 512]     y_test:  [N_test]                       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 3: Train XGBoost Classifier                                     ║  │
│  ║  Script: src/training/train_xgb_cls.py                         ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐          ║  │
│  ║  │ embeddings  │───▶│ XGBClassifier    │───▶│ predictions  │          ║  │
│  ║  │ .npz file   │    │ + class weights  │    │ + metrics    │          ║  │
│  ║  └─────────────┘    └──────────────────┘    └──────────────┘          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Pipeline Option 4: UNet AE → TwoStageClassifier (Hierarchical)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE 4: AE → TWO-STAGE CLASSIFIER                     │
│                                                                             │
│  Steps 1 & 2: Same as Pipeline 3 (Train AE, Extract Embeddings)             │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  STEP 3: Train TwoStageClassifier                                     ║  │
│  ║  Script: src/training/train_xgb_cls_2stage.py                              ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────┐                                                      ║  │
│  ║  │ embeddings  │                                                      ║  │
│  ║  │ .npz file   │                                                      ║  │
│  ║  └──────┬──────┘                                                      ║  │
│  ║         │                                                             ║  │
│  ║         ▼                                                             ║  │
│  ║  ┌──────────────────────────────────────────────────────────────┐     ║  │
│  ║  │                 STAGE 1: INTENTION DETECTOR                  │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  Labels:  noise → 0                                          │     ║  │
│  ║  │           upward|downward → 1 (intention)                    │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  XGBoost trained on ALL samples                              │     ║  │
│  ║  │  Optimized for: PR-AUC (handles class imbalance)             │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  Output: P(intention) for each sample                        │     ║  │
│  ║  │  Threshold tuning: Find t where recall >= target             │     ║  │
│  ║  └────────────────────────────┬─────────────────────────────────┘     ║  │
│  ║                               │                                       ║  │
│  ║         P(intention) >= threshold                                     ║  │
│  ║                               │                                       ║  │
│  ║                               ▼                                       ║  │
│  ║  ┌──────────────────────────────────────────────────────────────┐     ║  │
│  ║  │               STAGE 2: DIRECTION CLASSIFIER                  │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  Labels:  upward → 0                                         │     ║  │
│  ║  │           downward → 1                                       │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  XGBoost trained ONLY on intention samples                   │     ║  │
│  ║  │  Optimized for: Log-loss                                     │     ║  │
│  ║  │                                                              │     ║  │
│  ║  │  Output: upward (1) or downward (2)                          │     ║  │
│  ║  └──────────────────────────────────────────────────────────────┘     ║  │
│  ║                               │                                       ║  │
│  ║                               ▼                                       ║  │
│  ║                    Final: noise(0), upward(1), downward(2)            ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### Input Data Format

```
Raw H5 Files
    │
    │  Structure per file:
    │    /data/sequence_XXX/
    │      ├── ultrasound: [T, 3, Depth]  (T pulses, 3 channels, 130 depth)
    │      ├── label: int (0=noise, 1=upward, 2=downward)
    │      └── metadata: {...}
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA LOADING PIPELINE                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        GLOBAL FILTERS (optional)                        │ │
│  │  Filter by: participant_id, session_id, experiment_id, label            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     TEST/VAL SELECTION STRATEGY                         │ │
│  │                                                                         │ │
│  │  Option A: "filter"                                                     │ │
│  │    Select specific participants/sessions/experiments for test/val       │ │
│  │                                                                         │ │
│  │  Option B: "random"                                                     │ │
│  │    Random selection of X% experiments OR N experiments                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         TEST/VAL SPLIT                                  │ │
│  │  test_val_split_ratio: 0.5 (50% test, 50% validation)                   │ │
│  │  split_level: "experiment" or "sequence"                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                         │
│                    │               │               │                         │
│                    ▼               ▼               ▼                         │
│              ┌──────────┐   ┌──────────┐   ┌──────────┐                      │
│              │  TRAIN   │   │   VAL    │   │   TEST   │                      │
│              │ (shuffle)│   │          │   │          │                      │
│              └────┬─────┘   └────┬─────┘   └────┬─────┘                      │
│                   │              │              │                            │
│                   ▼              ▼              ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    CLASS BALANCING (optional)                           │ │
│  │                                                                         │ │
│  │  Option A: Dataset-level oversampling                                   │ │
│  │    Methods: "duplicate" | "augment" | "mixed"                           │ │
│  │                                                                         │ │
│  │  Option B: Loss-level weighting (preferred)                             │ │
│  │    Compute inverse class frequencies as weights                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           DataLoader                                    │ │
│  │  batch_size: 50                                                         │ │
│  │  num_workers: 8                                                         │ │
│  │  pin_memory: true                                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        Output: [B, 3, Pulses, Depth]
                        Example: [50, 3, 10, 130]
```

### Dimension Transformations

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DIMENSION FLOW SUMMARY                                 │
│                                                                              │
│  Raw Data:        [T, 3, 130]         T=pulses, 3=channels, 130=depth        │
│                        │                                                     │
│                        ▼                                                     │
│  CNNAdapter:      [3, T, 130]         Transpose for conv (C, H, W)           │
│                        │                                                     │
│                        ▼                                                     │
│  Batched:         [B, 3, T, 130]      B=batch_size, T≈10 pulses              │
│                        │                                                     │
│           ┌────────────┴────────────┐                                        │
│           │                         │                                        │
│           ▼                         ▼                                        │
│  ┌─────────────────────┐   ┌─────────────────────┐                           │
│  │   Direct CNN        │   │   UNet Autoencoder  │                           │
│  │                     │   │                     │                           │
│  │ [B,3,10,130]        │   │ [B,3,10,130]        │                           │
│  │       │             │   │       │             │                           │
│  │       ▼             │   │       ▼             │                           │
│  │ Conv1: [B,32,10,43] │   │ Enc: [B,256,1,2]    │                           │
│  │       │             │   │       │             │                           │
│  │       ▼             │   │       ▼             │                           │
│  │ Conv2: [B,64,10,14] │   │ Bottleneck: [B,512] │                           │
│  │       │             │   │       │             │                           │
│  │       ▼             │   │       ├──────────────────▶ Embedding            │
│  │ Flat: [B,8960]      │   │       │             │                           │
│  │       │             │   │       ▼             │                           │
│  │       ▼             │   │ Dec: [B,3,10,130]   │◀──── Reconstruction       │
│  │ FC: [B,3] logits    │   │       │             │                           │
│  └─────────────────────┘   │       ▼             │                           │
│                            │ CLS: [B,3] logits   │◀──── Classification       │
│                            └─────────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Summary Table

| # | Pipeline | Model(s) | Training Script | Classification |
|---|----------|----------|-----------------|----------------|
| 1 | Direct CNN | DirectCNNClassifier | `train_cnn_cls.py` | End-to-end |
| 2 | AE + CLS Head | UNetAutoencoder + Head | `train_unet_ae.py` | Joint loss |
| 3 | AE → XGBoost | UNetAutoencoder → XGBClassifier | `train_xgb_cls.py` | On embeddings |
| 4 | AE → TwoStage | UNetAutoencoder → TwoStageClassifier | `train_xgb_cls_2stage.py` | Hierarchical |

### Model Inventory

| Model | File | Input Shape | Output |
|-------|------|-------------|--------|
| DirectCNNClassifier | `src/models/direct_cnn_classifier.py` | [B,3,T,D] | logits |
| UNetAutoencoder | `src/models/unet_ae.py` | [B,3,H,W] | (recon, emb, [logits]) |
| TwoStageClassifier | `src/models/two_stage_classifier.py` | embeddings | predictions |

---

## 5. Configuration Reference

Key config sections in `config/config.yaml`:

```yaml
ml:
  model:
    type: "UNetAutoencoder"
    channels_per_layer: [32, 64, 128, 256]
    embedding_dim: 512

  classifier:
    type: "XGBoost"  # or TwoStage
    enable_classification_head: true  # For AE + CLS head
    classification_weight: 0.4        # Weight in joint loss

  training:
    epochs: 1000
    learning_rate: 0.0003
    batch_size: 50
    early_stopping:
      patience: 20
      min_delta: 0.00001

  loss_weights:
    mse_weight: 0.3
    l1_weight: 0.3
    embedding_reg: 0.0005
    classification_weight: 0.4
```