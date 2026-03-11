# UNetAutoencoder + Classification Head

## Overview
U-Net architecture with skip connections for superior reconstruction quality. Skip connections preserve fine details (edges, texture) that would be lost in a standard autoencoder bottleneck.

**Key Features:**
- Skip connections at each encoder level
- Better reconstruction of high-frequency details
- Same embedding/classification capability as CNNAutoencoder
- Slightly more parameters due to skip convolutions

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNetAutoencoder Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: (B, 3, 10, 130)                                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    ENCODER                              DECODER                          ││
│  │    ═══════                              ═══════                          ││
│  │                                                                          ││
│  │    ┌─────────┐                          ┌─────────┐                      ││
│  │    │  Input  │ ─────────────────────────│  Final  │───→ Reconstruction  ││
│  │    │(3,10,130)│        Skip 1           │(3,10,130)│                     ││
│  │    └────┬────┘                          └────▲────┘                      ││
│  │         │                                    │                           ││
│  │         ▼                                    │                           ││
│  │    ┌─────────┐                          ┌────┴────┐                      ││
│  │    │  Enc1   │ ─────────────────────────│  Dec1   │                      ││
│  │    │(32,5,65)│        Skip 2            │(32,5,65)│                      ││
│  │    └────┬────┘         (concat)         └────▲────┘                      ││
│  │         │                                    │                           ││
│  │         ▼                                    │                           ││
│  │    ┌─────────┐                          ┌────┴────┐                      ││
│  │    │  Enc2   │ ─────────────────────────│  Dec2   │                      ││
│  │    │(64,3,33)│        Skip 3            │(64,3,33)│                      ││
│  │    └────┬────┘         (concat)         └────▲────┘                      ││
│  │         │                                    │                           ││
│  │         ▼                                    │                           ││
│  │    ┌─────────┐                          ┌────┴────┐                      ││
│  │    │  Enc3   │ ─────────────────────────│  Dec3   │                      ││
│  │    │(128,2,17)│       Skip 4            │(128,2,17)│                     ││
│  │    └────┬────┘        (concat)          └────▲────┘                      ││
│  │         │                                    │                           ││
│  │         ▼                                    │                           ││
│  │    ┌─────────────────────────────────────────┴─────┐                     ││
│  │    │              BOTTLENECK                        │                    ││
│  │    │  Flatten: 128 × 2 × 17 = 4,352                │                    ││
│  │    │  Linear(4352 → 512) → EMBEDDING               │                    ││
│  │    │  Linear(512 → 4352) + Reshape                 │                    ││
│  │    └───────────────────────┬───────────────────────┘                    ││
│  │                            │                                             ││
│  │                            │                                             ││
│  │                    ┌───────┴───────┐                                     ││
│  │                    │  CLASSIFIER   │                                     ││
│  │                    │  (optional)   │                                     ││
│  │                    │ 512→256→3     │                                     ││
│  │                    └───────┬───────┘                                     ││
│  │                            │                                             ││
│  │                            ▼                                             ││
│  │                    LOGITS: (B, 3)                                        ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Skip Connection Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SKIP CONNECTION MECHANISM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Encoder output at level i:   e_i  (shape: C_i × H_i × W_i)                  │
│  Decoder input at level i:    d_i  (shape: C_i × H_i × W_i)                  │
│                                                                              │
│  Skip connection operation:                                                  │
│    1. Upsample d_i to match e_i spatial dimensions                           │
│    2. Concatenate along channel dimension: [d_i, e_i]                        │
│    3. Convolve to reduce channels back: 2×C_i → C_i                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │    Encoder                            Decoder                            ││
│  │    Output                             Input                              ││
│  │      │                                  │                                ││
│  │      │ (C, H, W)                        │ (C, H', W')                    ││
│  │      │                                  │                                ││
│  │      │                                  ▼                                ││
│  │      │                           ┌──────────────┐                        ││
│  │      │                           │  ConvTransp  │                        ││
│  │      │                           │  (Upsample)  │                        ││
│  │      │                           └──────┬───────┘                        ││
│  │      │                                  │                                ││
│  │      │ (C, H, W)                        │ (C, H, W)                      ││
│  │      │                                  │                                ││
│  │      └─────────────┐    ┌───────────────┘                                ││
│  │                    │    │                                                ││
│  │                    ▼    ▼                                                ││
│  │              ┌──────────────┐                                            ││
│  │              │  CONCAT      │                                            ││
│  │              │  (2C, H, W)  │                                            ││
│  │              └──────┬───────┘                                            ││
│  │                     │                                                    ││
│  │                     ▼                                                    ││
│  │              ┌──────────────┐                                            ││
│  │              │  Conv2d      │                                            ││
│  │              │  2C → C      │                                            ││
│  │              └──────┬───────┘                                            ││
│  │                     │                                                    ││
│  │                     │ (C, H, W)                                          ││
│  │                     ▼                                                    ││
│  │              Output to next                                              ││
│  │              decoder level                                               ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  BENEFIT:                                                                    │
│    - High-frequency details (edges, textures) bypass the bottleneck          │
│    - Decoder receives both compressed (embedding) and original features      │
│    - Results in sharper, more accurate reconstructions                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoder/Decoder Block Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENCODER BLOCK                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (C_in, H, W)                                                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Conv2d(C_in → C_out, k=3, s=2, p=1)    ← Downsamples by 2              ││
│  │  BatchNorm2d(C_out)                                                     ││
│  │  LeakyReLU(0.2)                                                         ││
│  │                                                                          ││
│  │  Conv2d(C_out → C_out, k=3, s=1, p=1)   ← Refines features              ││
│  │  BatchNorm2d(C_out)                                                     ││
│  │  LeakyReLU(0.2)                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Output: (C_out, H/2, W/2)                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      DECODER BLOCK                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (C_in, H, W) + Skip: (C_out, H×2, W×2)                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ConvTranspose2d(C_in → C_out, k=3, s=2, p=1, op=1)  ← Upsample ×2      ││
│  │  BatchNorm2d(C_out)                                                     ││
│  │  LeakyReLU(0.2)                                                         ││
│  │                                                                          ││
│  │  [Handle size mismatch if needed - crop/pad]                            ││
│  │                                                                          ││
│  │  Concatenate with skip: (2×C_out, H×2, W×2)                             ││
│  │                                                                          ││
│  │  Conv2d(2×C_out → C_out, k=3, s=1, p=1)  ← Fuse features                ││
│  │  BatchNorm2d(C_out)                                                     ││
│  │  LeakyReLU(0.2)                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Output: (C_out, H×2, W×2)                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: CNNAutoencoder vs UNetAutoencoder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE COMPARISON                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    CNNAutoencoder          UNetAutoencoder                   │
│                    ══════════════          ═══════════════                   │
│                                                                              │
│  Skip Connections     No                      Yes                            │
│  Parameters           ~1.5M                   ~2.5M                          │
│  Reconstruction       Good                    Excellent                      │
│  Edge Preservation    Moderate                High                           │
│  Training Speed       Faster                  Slower                         │
│  Memory Usage         Lower                   Higher                         │
│                                                                              │
│  Information Flow:                                                           │
│                                                                              │
│  CNNAutoencoder:                                                             │
│    Input ──→ Encoder ──→ [Bottleneck] ──→ Decoder ──→ Output                │
│                              512 dims                                        │
│    ALL information must pass through 512-dim bottleneck                      │
│                                                                              │
│  UNetAutoencoder:                                                            │
│    Input ──→ Encoder ──→ [Bottleneck] ──→ Decoder ──→ Output                │
│         └──────────────────────────────────────┘                             │
│           Skip connections bypass bottleneck                                 │
│    High-frequency details preserved through skips                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```yaml
ml:
  model:
    type: "UNetAutoencoder"            # Change from CNNAutoencoder
    channels_per_layer: [32, 64, 128]  # 3 encoder levels
    embedding_dim: 512

  training:
    loss_weights:
      mse_weight: 0.3
      l1_weight: 0.3                   # L1 helps with sharpness
      embedding_reg: 0.0005
      classification_weight: 0.4       # Enable joint classification
```

---

## Usage

```bash
# Train UNet with classification
python -m src.training.train_cnn_ae --config config/config.yaml
# (Set model.type: UNetAutoencoder in config)

# The same evaluation scripts work for both CNNAutoencoder and UNetAutoencoder
python -m src.evaluation.evaluate_classifier \
    --config config/config.yaml \
    --checkpoint path/to/model.pth
```

---

## When to Use

**Use UNetAutoencoder when:**
- Reconstruction quality is important
- You need to preserve fine details (edges, textures)
- You have sufficient GPU memory
- Joint reconstruction + classification

**Use CNNAutoencoder instead when:**
- Memory is constrained
- Reconstruction quality is less critical
- Faster training is needed
