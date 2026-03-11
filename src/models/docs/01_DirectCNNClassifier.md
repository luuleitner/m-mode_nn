# DirectCNNClassifier

## Overview
Simple 2-layer CNN classifier without autoencoder bottleneck. Based on colleague's proven architecture adapted for decimated input (130 depth vs 1000).

**Key Features:**
- No reconstruction objective (focused gradients)
- Large flattened feature space (3840 dims)
- Focal loss for class imbalance
- ~70% accuracy proven on similar data

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DirectCNNClassifier Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, 3, 10, 130)                                                      │
│         ├─ B: Batch size                                                     │
│         ├─ 3: US channels (I/Q/Envelope or similar)                          │
│         ├─ 10: Temporal (pulses)                                             │
│         └─ 130: Spatial (depth samples after decimation)                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CONV BLOCK 1: Depth Pattern Extraction                                  ││
│  │                                                                          ││
│  │  Conv2d(3→32, kernel=(1,13), stride=1, padding=0)                       ││
│  │  ├─ Kernel (1,13): operates only on depth dimension                     ││
│  │  ├─ Output: (B, 32, 10, 118)  [130-13+1=118]                            ││
│  │  │                                                                       ││
│  │  BatchNorm2d(32)                                                        ││
│  │  ReLU                                                                    ││
│  │  │                                                                       ││
│  │  MaxPool2d(kernel=(1,3), stride=(1,3))                                  ││
│  │  └─ Output: (B, 32, 10, 39)   [118//3=39]                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CONV BLOCK 2: Temporal + Depth Pattern Extraction                       ││
│  │                                                                          ││
│  │  Conv2d(32→64, kernel=(5,9), stride=1, padding=0)                       ││
│  │  ├─ Kernel (5,9): captures both temporal and depth patterns             ││
│  │  ├─ Output: (B, 64, 6, 31)   [10-5+1=6, 39-9+1=31]                      ││
│  │  │                                                                       ││
│  │  BatchNorm2d(64)                                                        ││
│  │  ReLU                                                                    ││
│  │  │                                                                       ││
│  │  MaxPool2d(kernel=(1,3), stride=(1,3))                                  ││
│  │  └─ Output: (B, 64, 6, 10)   [31//3=10]                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ FLATTEN                                                                 ││
│  │  64 × 6 × 10 = 3,840 features                                           ││
│  │  Output: (B, 3840)                                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CLASSIFICATION HEAD                                                     ││
│  │                                                                          ││
│  │  Dropout(0.3)                                                           ││
│  │  Linear(3840 → 256) + ReLU                                              ││
│  │  Dropout(0.3)                                                           ││
│  │  Linear(256 → 3)                                                        ││
│  │  │                                                                       ││
│  │  └─ Output: (B, 3) logits for [noise, upward, downward]                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  OUTPUT: (B, 3) class logits                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Colleague's Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLLEAGUE'S CNN (Reference)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (B, 3, 10, 1000)     ← 1000 depth samples (no decimation)            │
│                                                                              │
│  Conv1(1,51) → (B, 32, 10, 950)    ← Large 51-sample depth kernel            │
│  Pool1(1,5)  → (B, 32, 10, 190)                                              │
│                                                                              │
│  Conv2(5,23) → (B, 32, 6, 168)     ← 5 temporal, 23 depth                    │
│  Pool2(1,5)  → (B, 32, 6, 33)                                                │
│                                                                              │
│  Flatten    → (B, 6336)            ← 32 × 6 × 33 = 6,336 features            │
│  FC         → (B, 9)               ← 9 classes                               │
│                                                                              │
│  Result: ~70% accuracy                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    OUR ADAPTED CNN                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (B, 3, 10, 130)      ← 130 depth samples (10x decimation)            │
│                                                                              │
│  Conv1(1,13) → (B, 32, 10, 118)    ← Scaled kernel (13 ≈ 130×10%)            │
│  Pool1(1,3)  → (B, 32, 10, 39)     ← Smaller pool (less depth to work with)  │
│                                                                              │
│  Conv2(5,9)  → (B, 64, 6, 31)      ← 5 temporal, 9 depth                     │
│  Pool2(1,3)  → (B, 64, 6, 10)                                                │
│                                                                              │
│  Flatten    → (B, 3840)            ← 64 × 6 × 10 = 3,840 features            │
│  FC         → (B, 3)               ← 3 classes                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Input (B, 3, 10, 130)
    │
    │ Conv1 (1,13): Extract depth patterns
    ▼
(B, 32, 10, 118)
    │
    │ Pool (1,3): Reduce depth
    ▼
(B, 32, 10, 39)
    │
    │ Conv2 (5,9): Extract temporal+depth patterns
    ▼
(B, 64, 6, 31)
    │
    │ Pool (1,3): Final reduction
    ▼
(B, 64, 6, 10)
    │
    │ Flatten
    ▼
(B, 3840)          ← EMBEDDING (can be extracted for analysis)
    │
    │ FC layers
    ▼
(B, 3)             ← Class logits
```

---

## Loss Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FOCAL LOSS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Standard Cross-Entropy:                                                     │
│    CE(p) = -log(p_correct)                                                   │
│                                                                              │
│  Focal Loss (for class imbalance):                                           │
│    FL(p) = -(1 - p_correct)^γ × log(p_correct)                               │
│                                                                              │
│  Where:                                                                      │
│    p_correct = predicted probability for the true class                      │
│    γ (gamma) = focusing parameter (default: 2.0)                             │
│                                                                              │
│  Effect:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  When p_correct is HIGH (easy example, e.g., noise correctly predicted):││
│  │    (1 - 0.95)^2 = 0.0025  ← Loss nearly zero, not learned from          ││
│  │                                                                          ││
│  │  When p_correct is LOW (hard example, e.g., minority misclassified):    ││
│  │    (1 - 0.1)^2 = 0.81    ← Loss still high, forces learning             ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Combined with class weights:                                                │
│    weights = [0.37, 6.67, 6.67] for 90/5/5 imbalance                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

```bash
# Train with default settings (focal loss enabled)
python -m src.training.train_direct_classifier \
    --config config/config.yaml \
    --data-dir /path/to/pickles

# Train larger variant
python -m src.training.train_direct_classifier \
    --config config/config.yaml \
    --data-dir /path/to/pickles \
    --variant large

# Adjust focal loss gamma
python -m src.training.train_direct_classifier \
    --config config/config.yaml \
    --data-dir /path/to/pickles \
    --focal-gamma 3.0
```

---

## Model Parameters

| Component | Parameters |
|-----------|------------|
| Conv1 | 3 × 32 × 1 × 13 + 32 = 1,280 |
| BatchNorm1 | 64 |
| Conv2 | 32 × 64 × 5 × 9 + 64 = 92,224 |
| BatchNorm2 | 128 |
| FC1 | 3840 × 256 + 256 = 983,296 |
| FC2 | 256 × 3 + 3 = 771 |
| **Total** | **~1.1M parameters** |
