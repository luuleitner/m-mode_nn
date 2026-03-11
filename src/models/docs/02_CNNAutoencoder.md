# CNNAutoencoder + Classification Head

## Overview
Vanilla CNN Autoencoder with optional classification head for joint reconstruction + classification training. Compresses input to a low-dimensional embedding, then reconstructs.

**Key Features:**
- Dual objective: reconstruction (MSE/L1) + classification (CE)
- Embedding bottleneck for dimensionality reduction
- Embeddings usable for visualization, clustering, XGBoost

**Limitation:**
- Bottleneck (512 dims) may lose discriminative information

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CNNAutoencoder Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, 3, 10, 130)     [Batch, Channels, Pulses, Depth]                 │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                              ENCODER                                         │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ENCODER BLOCK 1                                                         ││
│  │  Conv2d(3→32, k=3, s=2, p=1) + BN + ReLU                                ││
│  │  Output: (B, 32, 5, 65)        [10/2=5, 130/2=65]                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ENCODER BLOCK 2                                                         ││
│  │  Conv2d(32→64, k=3, s=2, p=1) + BN + ReLU                               ││
│  │  Output: (B, 64, 3, 33)        [5/2≈3, 65/2≈33]                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ENCODER BLOCK 3                                                         ││
│  │  Conv2d(64→128, k=3, s=2, p=1) + BN + ReLU                              ││
│  │  Output: (B, 128, 2, 17)       [3/2≈2, 33/2≈17]                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ FLATTEN + PROJECT                                                       ││
│  │  Flatten: 128 × 2 × 17 = 4,352                                          ││
│  │  Linear(4352 → 512)                                                     ││
│  │  Output: (B, 512)              ← EMBEDDING                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│              ┌───────────────┴───────────────┐                               │
│              │                               │                               │
│              ▼                               ▼                               │
│  ════════════════════════         ════════════════════════                   │
│         DECODER                      CLASSIFIER (optional)                   │
│  ════════════════════════         ════════════════════════                   │
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐                   │
│  │ Linear(512 → 4352)   │         │ Linear(512 → 256)    │                   │
│  │ ReLU                 │         │ ReLU + Dropout(0.3)  │                   │
│  │ Reshape to           │         │ Linear(256 → 3)      │                   │
│  │ (B, 128, 2, 17)      │         │                      │                   │
│  └──────────────────────┘         └──────────────────────┘                   │
│           │                                │                                 │
│           ▼                                ▼                                 │
│  ┌──────────────────────┐         ┌──────────────────────┐                   │
│  │ DECODER BLOCK 3      │         │ LOGITS: (B, 3)       │                   │
│  │ ConvT(128→64, s=2)   │         │ [noise, up, down]    │                   │
│  │ + BN + ReLU          │         └──────────────────────┘                   │
│  │ → (B, 64, 4, 34)     │                                                    │
│  └──────────────────────┘                                                    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────┐                                                    │
│  │ DECODER BLOCK 2      │                                                    │
│  │ ConvT(64→32, s=2)    │                                                    │
│  │ + BN + ReLU          │                                                    │
│  │ → (B, 32, 8, 68)     │                                                    │
│  └──────────────────────┘                                                    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────┐                                                    │
│  │ DECODER FINAL        │                                                    │
│  │ ConvT(32→3, s=2)     │                                                    │
│  │ → (B, 3, 16, 136)    │                                                    │
│  │ Crop to (B, 3, 10, 130)│                                                  │
│  └──────────────────────┘                                                    │
│           │                                                                  │
│           ▼                                                                  │
│  RECONSTRUCTION: (B, 3, 10, 130)                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

OUTPUT: (reconstruction, embedding, logits)  if classifier enabled
        (reconstruction, embedding)          if classifier disabled
```

---

## Data Flow

```
                    INPUT
                      │
                      │ (B, 3, 10, 130)
                      ▼
              ┌───────────────┐
              │    ENCODER    │
              │  (3 blocks)   │
              └───────┬───────┘
                      │
                      │ (B, 128, 2, 17) = 4,352 features
                      ▼
              ┌───────────────┐
              │   FLATTEN +   │
              │   PROJECT     │
              └───────┬───────┘
                      │
                      │ (B, 512) EMBEDDING
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
  ┌───────────────┐      ┌───────────────┐
  │    DECODER    │      │  CLASSIFIER   │
  │  (3 blocks)   │      │    (MLP)      │
  └───────┬───────┘      └───────┬───────┘
          │                       │
          │ (B, 3, 10, 130)       │ (B, 3)
          ▼                       ▼
    RECONSTRUCTION              LOGITS
```

---

## Loss Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOINT LOSS                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L_total = w_mse × L_mse + w_l1 × L_l1 + w_emb × L_emb + w_cls × L_cls       │
│                                                                              │
│  Where (from config):                                                        │
│    w_mse = 0.3   (reconstruction MSE)                                        │
│    w_l1  = 0.3   (reconstruction L1 - sharper edges)                         │
│    w_emb = 0.0005 (embedding regularization)                                 │
│    w_cls = 0.4   (classification cross-entropy)                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Reconstruction Loss:                                                    ││
│  │    L_mse = ||input - reconstruction||²                                   ││
│  │    L_l1  = ||input - reconstruction||₁                                   ││
│  │                                                                          ││
│  │  Embedding Regularization:                                               ││
│  │    L_emb = ||embedding||²  (prevents explosion)                          ││
│  │                                                                          ││
│  │  Classification Loss (with class weights):                               ││
│  │    L_cls = CrossEntropy(logits, labels, weight=[0.37, 6.67, 6.67])       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  IMPORTANT:                                                                  │
│  The reconstruction and classification objectives can CONFLICT:              │
│    - Reconstruction wants smooth, average features                           │
│    - Classification wants sharp, discriminative features                     │
│  This is why DirectCNNClassifier may outperform for pure classification.     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```yaml
ml:
  model:
    type: "CNNAutoencoder"
    channels_per_layer: [32, 64, 128]   # 3 encoder blocks
    embedding_dim: 512                   # Bottleneck size

  training:
    loss_weights:
      mse_weight: 0.3
      l1_weight: 0.3
      embedding_reg: 0.0005
      classification_weight: 0.4        # Set > 0 to enable classifier

    class_balancing:
      enabled: true
      method: "weighted_loss"
```

---

## Embedding Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING QUALITY                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Fisher Separability Score:                                                  │
│    score = trace(S_between) / trace(S_within)                                │
│                                                                              │
│  Interpretation:                                                             │
│    score < 0.1  : VERY LOW - embeddings not discriminative                   │
│    score < 0.5  : LOW - some overlap, classification challenging             │
│    score < 1.0  : MODERATE - partial separation                              │
│    score >= 1.0 : GOOD - classes reasonably separated                        │
│                                                                              │
│  Typical Results:                                                            │
│    Pure AE (no classification): score ≈ 0.003 (very poor)                    │
│    Joint AE+CLS:                score ≈ 0.1-0.5 (improved)                   │
│                                                                              │
│  Use visualization/embedding_visualizer.py to diagnose:                      │
│    python visualization/embedding_visualizer.py -e embeddings.npz            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

```bash
# Train with joint classification
python -m src.training.train_cnn_ae --config config/config.yaml

# Extract embeddings for analysis
python -m src.training.extract_embeddings \
    --config config/config.yaml \
    --checkpoint path/to/model.pth

# Visualize embedding quality
python visualization/embedding_visualizer.py \
    --embeddings path/to/embeddings.npz \
    --method tsne
```

---

## When to Use

**Use CNNAutoencoder when:**
- You need embeddings for clustering/visualization
- You want both reconstruction and classification
- Memory is limited (smaller than UNet)

**Don't use when:**
- Classification accuracy is the only goal (use DirectCNN)
- You need best reconstruction quality (use UNet)
- Data is extremely imbalanced (use TwoStage)
