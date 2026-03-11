# TransformerAutoencoder

## Overview
Transformer-based autoencoders for temporal sequence modeling. Uses self-attention to capture long-range temporal dependencies in M-mode data.

**Variants:**
1. `TransformerAutoencoder`: Pure transformer (linear projection)
2. `CNNTransformerAutoencoder`: CNN for spatial + Transformer for temporal

**Note:** These are experimental architectures, primarily for sequence-to-sequence reconstruction.

---

## TransformerAutoencoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  TransformerAutoencoder Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, T, C, H, W)    [Batch, Time, Channels, Height, Width]            │
│         (B, 10, 3, 130, 5)  10 frames, 3 channels, 130×5 spatial             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ INPUT PROJECTION                                                        ││
│  │                                                                          ││
│  │  For each time step t:                                                  ││
│  │    Flatten: (C, H, W) → (C×H×W) = (3×130×5) = 1,950                     ││
│  │    Linear: 1950 → 512 (2 layers with LayerNorm)                         ││
│  │                                                                          ││
│  │  Output: (B, T, 256) = (B, 10, 256)                                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ POSITIONAL ENCODING                                                     ││
│  │                                                                          ││
│  │  Add sinusoidal position encoding to capture temporal order:            ││
│  │                                                                          ││
│  │  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                          ││
│  │  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                          ││
│  │                                                                          ││
│  │  Frame 0: [sin, cos, sin, cos, ...]                                     ││
│  │  Frame 1: [sin, cos, sin, cos, ...]                                     ││
│  │  ...                                                                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TRANSFORMER ENCODER (4 layers)                                          ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  TransformerEncoderLayer:                                        │   ││
│  │  │    - Multi-Head Self-Attention (8 heads)                        │   ││
│  │  │    - Feed-Forward Network (256 → 1024 → 256)                    │   ││
│  │  │    - LayerNorm (pre-norm)                                       │   ││
│  │  │    - Dropout (0.1)                                              │   ││
│  │  │    - Residual connections                                       │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                          ││
│  │  Self-Attention allows each frame to attend to all other frames:        ││
│  │                                                                          ││
│  │    Frame 0 ←───────────────→ Frame 5                                    ││
│  │    Frame 1 ←───────────────→ Frame 6                                    ││
│  │    ...                                                                   ││
│  │                                                                          ││
│  │  Output: (B, T, 256) = memory                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ BOTTLENECK                                                              ││
│  │                                                                          ││
│  │  Linear(256 → 128) + ReLU + Linear(128 → 256)                           ││
│  │                                                                          ││
│  │  Output: (B, T, 256) = latent                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TRANSFORMER DECODER (4 layers)                                          ││
│  │                                                                          ││
│  │  TransformerDecoderLayer:                                               ││
│  │    - Self-Attention on latent                                           ││
│  │    - Cross-Attention with encoder memory                                ││
│  │    - Feed-Forward Network                                               ││
│  │                                                                          ││
│  │  Output: (B, T, 256)                                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ OUTPUT PROJECTION                                                       ││
│  │                                                                          ││
│  │  Linear: 256 → 1950 (2 layers)                                          ││
│  │  Reshape: 1950 → (3, 130, 5)                                            ││
│  │                                                                          ││
│  │  Output: (B, T, C, H, W) = reconstruction                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  OUTPUT: (reconstruction, latent)                                            │
│          (B, 10, 3, 130, 5), (B, 10, 256)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CNNTransformerAutoencoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               CNNTransformerAutoencoder Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Combines CNN (for spatial features) with Transformer (for temporal).        │
│                                                                              │
│  INPUT: (B, T, C, H, W) = (B, 10, 3, 130, 5)                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CNN ENCODER (per frame)                                                 ││
│  │                                                                          ││
│  │  For each frame independently:                                          ││
│  │                                                                          ││
│  │    (3, 130, 5) ──→ Width Reducer ──→ (32, 130, 1)                       ││
│  │                           │                                              ││
│  │                           ▼                                              ││
│  │    CNN Encoder: Conv2d layers with strides                              ││
│  │      (32, 130, 1) → (64, 43, 1) → (128, 22, 1) →                        ││
│  │      (256, 11, 1) → (512, 5, 1) → AdaptivePool → (512, 1, 1)            ││
│  │                           │                                              ││
│  │                           ▼                                              ││
│  │    Flatten + Linear → (256)                                             ││
│  │                                                                          ││
│  │  Stack all frames: (B, T, 256)                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ POSITIONAL ENCODING                                                     ││
│  │  Add temporal position information                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TRANSFORMER ENCODER                                                     ││
│  │  Model temporal dependencies across frames                              ││
│  │  Output: (B, T, 256)                                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ BOTTLENECK                                                              ││
│  │  Compression: 256 → 128 → 256                                           ││
│  │  Output: (B, T, 256) = latent                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ TRANSFORMER DECODER                                                     ││
│  │  Reconstruct temporal sequence                                          ││
│  │  Output: (B, T, 256)                                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CNN DECODER (per frame)                                                 ││
│  │                                                                          ││
│  │  Linear + Reshape → (512, 5, 1)                                         ││
│  │  ConvTranspose layers → (256, 11, 1) → (128, 22, 1) →                   ││
│  │    (64, 43, 1) → (32, 130, 1)                                           ││
│  │  Width Restorer → (3, 130, 5)                                           ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  OUTPUT: (B, T, 3, 130, 5), (B, T, 256)                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Self-Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SELF-ATTENTION                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Multi-Head Self-Attention allows each time step to attend to all others:    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Query (Q), Key (K), Value (V) computed from input X:                   ││
│  │                                                                          ││
│  │    Q = X × W_Q                                                          ││
│  │    K = X × W_K                                                          ││
│  │    V = X × W_V                                                          ││
│  │                                                                          ││
│  │  Attention weights:                                                      ││
│  │                                                                          ││
│  │    Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V                      ││
│  │                                                                          ││
│  │  For M-mode sequence:                                                   ││
│  │                                                                          ││
│  │    Frame 0: "Which other frames are relevant to reconstruct me?"        ││
│  │    → Attends to frames with similar or causally related patterns        ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────┐                ││
│  │  │  Attention Matrix (10×10 for 10 frames):            │                ││
│  │  │                                                      │                ││
│  │  │       F0   F1   F2   F3   F4   F5   F6   F7   F8   F9│                ││
│  │  │  F0 [0.3  0.2  0.1  0.05 0.05 0.05 0.05 0.05 0.05 0.1]│               ││
│  │  │  F1 [0.2  0.3  0.2  0.1  0.05 ...]                   │                ││
│  │  │  F2 [0.1  0.2  0.3  0.2  ...]                        │                ││
│  │  │  ...                                                 │                ││
│  │  │                                                      │                ││
│  │  │  Higher values = more attention (more important)     │                ││
│  │  └─────────────────────────────────────────────────────┘                ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  BENEFIT for M-mode:                                                         │
│    - Captures long-range temporal dependencies                               │
│    - Movement patterns span multiple frames                                  │
│    - Can model periodic or gradual changes                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: CNN vs Transformer for M-mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CNN vs TRANSFORMER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Aspect               CNN Autoencoder      Transformer Autoencoder          │
│  ──────               ───────────────      ────────────────────             │
│  Receptive Field      Local (kernel size)  Global (all frames)              │
│  Temporal Modeling    Implicit (strides)   Explicit (attention)             │
│  Parameters           ~1-2M                ~5-10M                            │
│  Training Speed       Fast                 Slower                            │
│  Memory Usage         Lower                Higher (O(T²) attention)          │
│  Sequence Length      Any (conv)           Limited (memory)                  │
│                                                                              │
│  M-mode characteristics:                                                     │
│    - Sequence length: 10-15 frames (manageable for transformer)              │
│    - Movement patterns: span multiple frames (benefits attention)            │
│    - Spatial structure: 130×5 (benefits CNN)                                 │
│                                                                              │
│  RECOMMENDATION:                                                             │
│    - For classification: DirectCNN or CNNAutoencoder                         │
│    - For sequence modeling: CNNTransformerAutoencoder                        │
│    - Transformer adds complexity with marginal benefit for this data         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```yaml
ml:
  model:
    type: "TransformerAutoencoder"
    embedding_dim: 256
    num_heads: 8           # Multi-head attention heads
    num_layers: 4          # Encoder/decoder layers
```

---

## When to Use

**Use Transformer AE when:**
- Long temporal dependencies are important
- Sequence-to-sequence reconstruction needed
- You have sufficient GPU memory
- Experimenting with attention mechanisms

**Don't use when:**
- Classification is the primary goal
- Memory/compute is limited
- Sequence is very short (<5 frames)
- Spatial features are more important than temporal
