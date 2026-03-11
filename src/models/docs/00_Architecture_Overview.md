# M-Mode Neural Network Architectures Overview

## Available Architectures

| # | Architecture | Type | Training | Classification | Best For |
|---|--------------|------|----------|----------------|----------|
| 1 | DirectCNNClassifier | End-to-end CNN | Supervised | Direct | Simple, proven baseline |
| 2 | CNNAutoencoder + CLS Head | Multi-task AE | Joint (MSE+CE) | From embedding | Embedding + classification |
| 3 | UNetAutoencoder + CLS Head | Multi-task AE | Joint (MSE+CE) | From embedding | Better reconstruction |
| 4 | AE + XGBoost | Two-stage | Separate | XGBoost on embedding | Interpretable features |
| 5 | TransformerAutoencoder | Sequence AE | Reconstruction | None (extract emb) | Temporal modeling |
| 6 | TwoStageClassifier | Hierarchical XGB | Separate stages | Detection + Direction | Imbalanced data |

---

## Quick Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DirectCNNClassifier (Recommended for classification)                     │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ Input → Conv1 → Pool → Conv2 → Pool → Flatten(3840) → FC → Classes │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│     Features: 3840 dims | Single objective | Proven ~70% accuracy            │
│                                                                              │
│  2. CNNAutoencoder + CLS Head                                                │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ Input → Encoder → Embedding(512) → Decoder → Reconstruction         │ │
│     │                         ↓                                            │ │
│     │                   Classifier → Classes                               │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│     Features: 512 dims | Dual objective (MSE+CE) | Embeddings available      │
│                                                                              │
│  3. UNetAutoencoder + CLS Head                                               │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ Input → Enc1 ──────────────────────────────────────────→ Dec1 → Out │ │
│     │          ↓                                                  ↑        │ │
│     │        Enc2 ────────────────────────────────────────→ Dec2 ─┘        │ │
│     │          ↓                                              ↑            │ │
│     │        Enc3 ──────────────────────────────────→ Dec3 ───┘            │ │
│     │          ↓           Embedding(512)              ↑                   │ │
│     │        Enc4 → Flatten → FC → Unflatten → Dec4 ───┘                   │ │
│     │                         ↓                                            │ │
│     │                   Classifier → Classes                               │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│     Features: Skip connections | Better reconstruction | 512 dim embedding   │
│                                                                              │
│  4. AE + XGBoost Pipeline                                                    │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ Stage 1: Train AE (reconstruction only)                             │ │
│     │          Input → Encoder → Embedding → Decoder → Reconstruction     │ │
│     │                                                                      │ │
│     │ Stage 2: Extract embeddings, train XGBoost                          │ │
│     │          Embeddings → XGBoost → Classes                             │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│     Features: Decoupled training | Interpretable | Feature importance        │
│                                                                              │
│  5. TwoStageClassifier (XGBoost)                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ Stage 1: Intention Detection                                        │ │
│     │          Embedding → XGBoost → {Noise, Intention}                   │ │
│     │                                     ↓                               │ │
│     │ Stage 2: Direction (if Intention detected)                          │ │
│     │          Embedding → XGBoost → {Upward, Downward}                   │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│     Features: Tunable threshold | Optimized for imbalanced data              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw RF Data                                                                 │
│  (N, 3, 2048, M)           3 US channels, 2048 depth samples, M pulses       │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ PREPROCESSING                                                           ││
│  │  1. Clip         : Remove start/end samples → 1300 depth               ││
│  │  2. TGC          : Time Gain Compensation                              ││
│  │  3. Bandpass     : 8-12 MHz filter                                     ││
│  │  4. Envelope     : Hilbert transform                                   ││
│  │  5. LogCompress  : 45 dB dynamic range                                 ││
│  │  6. Normalize    : Peak-Z normalization                                ││
│  │  7. Decimate     : Factor 10 → 130 depth samples                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│       │                                                                      │
│       ▼                                                                      │
│  Preprocessed Data                                                           │
│  (N, 3, 130, M)            3 channels, 130 depth, M pulses                   │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TOKENIZATION                                                            ││
│  │  Window: 10 pulses                                                      ││
│  │  Stride: 5 pulses                                                       ││
│  │  → Sliding window creates overlapping tokens                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│       │                                                                      │
│       ▼                                                                      │
│  Tokens (CNN Input)                                                          │
│  (B, 3, 130, 10)           Batch of 3-channel, 130 depth, 10 pulse tokens    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TRANSPOSE (for temporal preservation)                                   ││
│  │  (B, 3, 130, 10) → (B, 3, 10, 130)                                     ││
│  │  Now: 10 temporal steps × 130 spatial features                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│       │                                                                      │
│       ▼                                                                      │
│  Model Input                                                                 │
│  (B, 3, 10, 130)           Ready for CNN processing                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding Dimension Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING/FEATURE DIMENSIONS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Architecture               Flatten Dim    Embedding    Classification       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  DirectCNNClassifier        3,840          N/A          3,840 → FC → 3      │
│  DirectCNNClassifierLarge   ~2,000         N/A          ~2,000 → FC → 3     │
│  CNNAutoencoder             ~2,048         512          512 → MLP → 3       │
│  UNetAutoencoder            ~2,048         512          512 → MLP → 3       │
│  Colleague's CNN            6,336          N/A          6,336 → FC → 9      │
│                                                                              │
│  KEY INSIGHT:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ DirectCNN:   3,840 features → Classifier (no bottleneck)               ││
│  │ AE + CLS:      512 features → Classifier (bottleneck loses info)       ││
│  │                                                                         ││
│  │ The AE bottleneck (512) compresses too much for good classification!   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Scripts

| Architecture | Training Script | Command |
|--------------|-----------------|---------|
| DirectCNNClassifier | `train_direct_classifier.py` | `python -m src.training.train_direct_classifier -c config/config.yaml -d /path/to/data` |
| CNNAutoencoder | `train_cnn_ae.py` | `python -m src.training.train_cnn_ae -c config/config.yaml` |
| UNetAutoencoder | `train_cnn_ae.py` | Change `model.type: UNetAutoencoder` in config |
| AE + XGBoost | `extract_embeddings.py` + `train_classifier.py` | Two-stage training |
| TwoStageClassifier | `train_two_stage.py` | `python -m src.training.train_two_stage -c config/config.yaml` |

---

## Recommendations

### For Best Classification Accuracy
Use **DirectCNNClassifier** with focal loss:
- Proven architecture (colleague's ~70%)
- No bottleneck compression
- Single objective training
- Focal loss handles class imbalance

### For Embeddings + Classification
Use **UNetAutoencoder + CLS Head**:
- Skip connections preserve details
- Joint training with classification_weight > 0
- Embedding useful for visualization/clustering

### For Extreme Class Imbalance (90/5/5)
Use **TwoStageClassifier**:
- Separate detection (noise vs intention)
- Tunable detection threshold
- Optimized recall on minority classes

---

## File Index

- `01_DirectCNNClassifier.md` - Direct CNN architecture details
- `02_CNNAutoencoder.md` - CNN Autoencoder + Classification head
- `03_UNetAutoencoder.md` - U-Net Autoencoder + Classification head
- `04_AE_XGBoost_Pipeline.md` - AE + XGBoost two-stage pipeline
- `05_TwoStageClassifier.md` - Hierarchical XGBoost classifier
- `06_TransformerAutoencoder.md` - Transformer-based architectures
