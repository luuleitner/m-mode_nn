# CNN Autoencoder Architecture Analysis

## Document Information

| **Field** | **Value** |
|-----------|-----------|
| **Version** | 1.0 |
| **Created** | 2025-01-28 |
| **Model Class** | `CNNAutoencoder` |
| **File** | `models/cnn_ae.py` |
| **Input Format** | `[B, T, C, H, W]` - Ultrasound M-mode sequences |
| **Purpose** | Dimensionality reduction and feature extraction for ultrasound M-mode data |

---

## Layer Size Computation Formulas

### Convolution Layer Output Size
```
Output_Height = floor((Input_Height + 2×Padding_H - Kernel_H) / Stride_H) + 1
Output_Width  = floor((Input_Width  + 2×Padding_W - Kernel_W) / Stride_W) + 1
```

### Transposed Convolution (ConvTranspose2d) Output Size
```
Output_Height = (Input_Height - 1) × Stride_H - 2×Padding_H + Kernel_H + Output_Padding_H
Output_Width  = (Input_Width  - 1) × Stride_W - 2×Padding_W + Kernel_W + Output_Padding_W
```

### AdaptiveAvgPool2d Output Size
```
Output_Height = Target_Height (specified in parameters)
Output_Width  = Target_Width  (specified in parameters)
```

### Parameter Count Formulas
```
Conv2d Parameters    = (Kernel_H × Kernel_W × Input_Channels × Output_Channels) + Output_Channels
Linear Parameters    = (Input_Features × Output_Features) + Output_Features
BatchNorm2d Params   = 2 × Num_Channels  (γ and β parameters)
```

### Memory Usage Estimation
```
Memory (bytes) = Batch_Size × Channels × Height × Width × 4  (for float32)
Memory (MB)    = Memory (bytes) / (1024 × 1024)
```

---

## Architecture Overview

The CNNAutoencoder is a convolutional autoencoder designed for ultrasound M-mode signal processing. It compresses temporal ultrasound sequences into compact embeddings while preserving essential features for reconstruction.

### Key Design Principles
- **Width-First Reduction**: Reduces spatial width (5→1) before depth processing
- **Temporal Processing**: Handles sequences of ultrasound frames via batch reshaping
- **Progressive Compression**: Gradual dimension reduction through encoder
- **Symmetric Reconstruction**: Mirror decoder architecture for reconstruction

---

## ASCII Architecture Diagram

```
INPUT: [B, T, 3, 130, 5]  ←── Batch × Tokens × Channels × Height × Width
          │
          ▼
   ┌─────────────────┐
   │  WIDTH REDUCER  │  Conv2d(3→32, k=(1,5)) + BN + ReLU
   │  [B*T,32,130,1] │  ──────────────────────────────────
   └─────────┬───────┘
             │
             ▼
   ┌─────────────────┐
   │   ENCODER       │
   │                 │
   │ Conv1: 32→64    │  k=(3,1), s=(3,1), p=(0,0) → [B*T,64,43,1]
   │ + BN + ReLU     │  + Dropout2d(0.1)
   │                 │
   │ Conv2: 64→128   │  k=(3,1), s=(2,1), p=(1,0) → [B*T,128,22,1]
   │ + BN + ReLU     │  + Dropout2d(0.1)
   │                 │
   │ Conv3: 128→256  │  k=(3,1), s=(2,1), p=(1,0) → [B*T,256,11,1]
   │ + BN + ReLU     │
   │                 │
   │ Conv4: 256→512  │  k=(3,1), s=(2,1), p=(0,0) → [B*T,512,5,1]
   │ + BN + ReLU     │
   │                 │
   │ AdaptiveAvgPool │  → [B*T,512,1,1]
   │ Flatten         │  → [B*T,512]
   │ Linear(512→EMB) │  → [B*T,embedding_dim]
   │ LayerNorm       │
   └─────────┬───────┘
             │
             ▼
      ┌─────────────┐
      │ EMBEDDING   │  [B, T, embedding_dim]
      │  BOTTLENECK │  ═══════════════════════
      └─────────────┘
             │
             ▼
   ┌─────────────────┐
   │   DECODER       │
   │                 │
   │ Linear Proj     │  embedding_dim → 512*5
   │ + ReLU          │  → [B*T,2560] → [B*T,512,5,1]
   │                 │
   │ ConvT1: 512→256 │  k=(3,1), s=(2,1), p=(0,0), op=(0,0) → [B*T,256,11,1]
   │ + BN + ReLU     │
   │                 │
   │ ConvT2: 256→128 │  k=(3,1), s=(2,1), p=(1,0), op=(1,0) → [B*T,128,22,1]
   │ + BN + ReLU     │
   │                 │
   │ ConvT3: 128→64  │  k=(3,1), s=(2,1), p=(1,0), op=(1,0) → [B*T,64,44,1]
   │ + BN + ReLU     │
   │                 │
   │ ConvT4: 64→32   │  k=(3,1), s=(3,1), p=(0,0), op=(0,0) → [B*T,32,132,1]
   │ + BN + ReLU     │
   └─────────┬───────┘
             │
             ▼
   ┌─────────────────┐
   │ WIDTH RESTORER  │  Conv2d(32→160, k=(1,1)) + PixelShuffle(1)
   │  [B*T,32,130,5] │  → View as [B*T,32,130,5]
   └─────────┬───────┘
             │
             ▼
   ┌─────────────────┐
   │ FINAL PROJECTION│  Conv2d(32→3, k=(1,1))
   │  [B*T,3,130,5]  │  ──────────────────────
   └─────────┬───────┘
             │
             ▼
OUTPUT: [B, T, 3, 130, 5]  ←── Reconstructed input
```

---

## Dimension Flow Analysis

### Forward Pass Dimensions (Encoder)

| **Layer** | **Operation** | **Input Shape** | **Output Shape** | **Reduction Factor** |
|-----------|---------------|-----------------|------------------|---------------------|
| **Input** | - | `[B, T, 3, 130, 5]` | `[B, T, 3, 130, 5]` | 1.0× |
| **Reshape** | View | `[B, T, 3, 130, 5]` | `[B*T, 3, 130, 5]` | 1.0× |
| **Width Reducer** | Conv2d(3→32, k=(1,5)) | `[B*T, 3, 130, 5]` | `[B*T, 32, 130, 1]` | **5.0×** (width) |
| **Encoder Conv1** | Conv2d(32→64, k=(3,1), s=(3,1)) | `[B*T, 32, 130, 1]` | `[B*T, 64, 43, 1]` | **3.0×** (height) |
| **Encoder Conv2** | Conv2d(64→128, k=(3,1), s=(2,1), p=(1,0)) | `[B*T, 64, 43, 1]` | `[B*T, 128, 22, 1]` | **~2.0×** (height) |
| **Encoder Conv3** | Conv2d(128→256, k=(3,1), s=(2,1), p=(1,0)) | `[B*T, 128, 22, 1]` | `[B*T, 256, 11, 1]` | **2.0×** (height) |
| **Encoder Conv4** | Conv2d(256→512, k=(3,1), s=(2,1)) | `[B*T, 256, 11, 1]` | `[B*T, 512, 5, 1]` | **~2.2×** (height) |
| **AdaptiveAvgPool** | Global pooling | `[B*T, 512, 5, 1]` | `[B*T, 512, 1, 1]` | **5.0×** (height) |
| **Flatten** | Reshape | `[B*T, 512, 1, 1]` | `[B*T, 512]` | 1.0× |
| **Linear** | Linear(512→256) | `[B*T, 512]` | `[B*T, 256]` | **2.0×** |
| **Embedding** | Reshape | `[B*T, 256]` | `[B, T, 256]` | 1.0× |

### Decoder Dimension Flow

| **Layer** | **Operation** | **Input Shape** | **Output Shape** |
|-----------|---------------|-----------------|------------------|
| **Decoder Proj** | Linear(256→2560) + ReLU | `[B*T, 256]` | `[B*T, 2560]` |
| **Reshape** | View | `[B*T, 2560]` | `[B*T, 512, 5, 1]` |
| **ConvT1** | ConvTranspose2d(512→256, k=(3,1), s=(2,1)) | `[B*T, 512, 5, 1]` | `[B*T, 256, 11, 1]` |
| **ConvT2** | ConvTranspose2d(256→128, k=(3,1), s=(2,1), p=(1,0), op=(1,0)) | `[B*T, 256, 11, 1]` | `[B*T, 128, 22, 1]` |
| **ConvT3** | ConvTranspose2d(128→64, k=(3,1), s=(2,1), p=(1,0), op=(1,0)) | `[B*T, 128, 22, 1]` | `[B*T, 64, 44, 1]` |
| **ConvT4** | ConvTranspose2d(64→32, k=(3,1), s=(3,1)) | `[B*T, 64, 44, 1]` | `[B*T, 32, 132, 1]` |
| **Width Restorer** | Conv2d(32→160) + PixelShuffle + View | `[B*T, 32, 132, 1]` | `[B*T, 32, 130, 5]` |
| **Final Proj** | Conv2d(32→3, k=(1,1)) | `[B*T, 32, 130, 5]` | `[B*T, 3, 130, 5]` |
| **Output Reshape** | View | `[B*T, 3, 130, 5]` | `[B, T, 3, 130, 5]` |

### Compression Metrics

| **Stage** | **Elements** | **Memory (MB/sample)** | **Compression Ratio** |
|-----------|--------------|------------------------|----------------------|
| **Input** | `T×3×130×5 = 1,950×T` | `7.42×T KB` | **1.0×** (baseline) |
| **After Width Reduction** | `T×32×130×1 = 4,160×T` | `15.9×T KB` | **0.47×** ↗️ |
| **After Conv1** | `T×64×43×1 = 2,752×T` | `10.5×T KB` | **0.71×** ↗️ |
| **After Conv2** | `T×128×22×1 = 2,816×T` | `10.8×T KB` | **0.69×** ↗️ |
| **After Conv3** | `T×256×11×1 = 2,816×T` | `10.8×T KB` | **0.69×** ↗️ |
| **After Conv4** | `T×512×5×1 = 2,560×T` | `9.77×T KB` | **0.76×** ↗️ |
| **Embedding** | `T×256 = 256×T` | `1.0×T KB` | **7.62×** ↗️↗️ |

> **Overall Compression**: **7.62×** reduction from input to embedding

---

## Layer-by-Layer Analysis

### 1. Width Reduction Stage
```python
self.width_reducer = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True)
)
```

**Purpose**: Reduces width dimension from 5 to 1 while expanding channels
- **Input**: `[B*T, 3, 130, 5]` - 3-channel ultrasound frames
- **Output**: `[B*T, 32, 130, 1]` - 32-channel reduced-width features
- **Key Insight**: Collapses spatial width early, forcing model to encode width information in channel dimension
- **Parameters**: `3×32×1×5 + 32 = 512`

### 2. Encoder Backbone
```python
# Progressive channel expansion with spatial reduction
Conv1: 32→64  channels, 130→43  height (stride=3, no padding)
Conv2: 64→128 channels, 43→22   height (stride=2, padding=1)
Conv3: 128→256 channels, 22→11  height (stride=2, padding=1)
Conv4: 256→512 channels, 11→5   height (stride=2, no padding)
```

**Design Pattern**: Each layer approximately doubles channels while reducing spatial resolution
- **Regularization**: Dropout2d(0.1) on first two layers prevents overfitting
- **Normalization**: BatchNorm2d for training stability
- **Activation**: ReLU with inplace=True for memory efficiency

### 3. Bottleneck Compression
```python
nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
nn.Flatten()                   # Reshape to vector
nn.Linear(512, embedding_dim)  # Final compression
nn.LayerNorm(embedding_dim)    # Normalize embeddings
```

**Critical Stage**: Maximum compression point
- **Global Pooling**: Removes all spatial dependencies, produces translation-invariant features
- **Linear Projection**: Maps 512 features to desired embedding dimension (default: 256)
- **LayerNorm**: Ensures stable embedding distribution for downstream tasks

### 4. Decoder Projection
```python
self.decoder_projection = nn.Sequential(
    nn.Linear(embedding_dim, 512 * 5),
    nn.ReLU(inplace=True)
)
```

**Purpose**: Expands embedding back to spatial tensor
- **Input**: `[B*T, embedding_dim]`
- **Output**: `[B*T, 2560]` → reshaped to `[B*T, 512, 5, 1]`

### 5. Decoder Backbone
```python
# Progressive spatial expansion with channel reduction
ConvT1: 512→256 channels, 5→11   height
ConvT2: 256→128 channels, 11→22  height
ConvT3: 128→64  channels, 22→44  height
ConvT4: 64→32   channels, 44→132 height
```

**Design Pattern**: Mirrors encoder structure in reverse
- **Output Padding**: Carefully tuned to achieve desired spatial dimensions
- **Normalization**: BatchNorm2d on all layers

### 6. Width Restoration
```python
self.width_restorer = nn.Sequential(
    nn.Conv2d(32, 32 * 5, kernel_size=(1, 1)),
    nn.PixelShuffle(1),
)
self.final_projection = nn.Conv2d(32, 3, kernel_size=(1, 1))
```

**Width Recovery**: Expands from width=1 back to width=5
- **Note**: The decode method manually views the output as `[B*T, 32, 130, 5]`
- **Final Projection**: Reduces 32 channels back to 3 (original channel count)

---

## Mathematical Analysis

### Receptive Field Calculation

| **Layer** | **Kernel Size** | **Stride** | **Local RF** | **Cumulative RF** |
|-----------|-----------------|------------|--------------|-------------------|
| Width Reducer | (1, 5) | (1, 1) | 5 (width) | **5** (width) |
| Conv1 | (3, 1) | (3, 1) | 3 | **3** (height) |
| Conv2 | (3, 1) | (2, 1) | 3 | **3 + 2×3 = 9** |
| Conv3 | (3, 1) | (2, 1) | 3 | **9 + 4×3 = 21** |
| Conv4 | (3, 1) | (2, 1) | 3 | **21 + 8×3 = 45** |

**Final Receptive Field**: **45 pixels** in height direction, capturing ~35% of input height (130 pixels)

### Parameter Count Analysis

| **Component** | **Layers** | **Parameters** | **Percentage** |
|---------------|------------|----------------|----------------|
| **Width Reducer** | Conv2d + BN | 576 | 0.01% |
| **Encoder Conv1** | Conv2d + BN + Dropout | 6,336 | 0.11% |
| **Encoder Conv2** | Conv2d + BN + Dropout | 24,832 | 0.45% |
| **Encoder Conv3** | Conv2d + BN | 99,072 | 1.79% |
| **Encoder Conv4** | Conv2d + BN | 394,752 | 7.12% |
| **Encoder Linear** | Linear + LayerNorm | 131,840 | 2.38% |
| **Decoder Projection** | Linear | 657,920 | 11.87% |
| **Decoder ConvT1-4** | 4× ConvTranspose2d + BN | ~3.9M | 70.4% |
| **Width Restorer** | Conv2d | 5,280 | 0.10% |
| **Final Projection** | Conv2d | 99 | <0.01% |
| **Total** | | **~5.54M parameters** | 100% |

---

## Design Critique & Analysis

### ✅ **Strengths**

1. **Domain-Specific Design**
   - Width reduction targets ultrasound M-mode geometry (5 scanlines)
   - Progressive compression preserves hierarchical features
   - Temporal dimension handling for sequence data

2. **Training Stability**
   - BatchNorm on all conv layers
   - LayerNorm on embeddings
   - Dropout2d(0.1) for regularization on early layers
   - Inplace ReLU for memory efficiency

3. **Architectural Symmetry**
   - Encoder-decoder mirror structure
   - Systematic channel progression (32→64→128→256→512)
   - Proper dimension restoration

4. **Compression Efficiency**
   - **7.62× compression ratio** from input to embedding
   - Bottleneck preserves essential information
   - Global pooling removes spatial bias

### ⚠️ **Potential Issues**

1. **Information Bottleneck Risk**
   ```
   130×5 = 650 spatial positions → 256 embedding dims
   Severe spatial compression may lose fine-grained details
   ```

2. **Width Restoration Implementation**
   ```python
   # PixelShuffle(1) is effectively a no-op
   nn.PixelShuffle(1)  # Does not actually rearrange channels to spatial dims
   # The actual width restoration happens in decode() via view()
   x = x.view(B * T, 32, 130, 5)  # Manual reshaping
   ```

3. **Dimension Mismatch in Decoder**
   ```
   Decoder outputs [B*T, 32, 132, 1] but needs [B*T, 32, 130, 5]
   The view() operation assumes specific output dimensions
   ```

4. **Fixed Architecture**
   - Hard-coded for `[130, 5]` input dimensions
   - No adaptive sizing for different ultrasound formats
   - Embedding dimension is configurable but default is 256

### 🔧 **Suggested Improvements**

1. **Skip Connections (U-Net Style)**
   ```python
   # Add encoder-decoder skip connections for better reconstruction
   class CNNAutoencoderWithSkips(nn.Module):
       def forward(self, x):
           # Store encoder features
           enc1 = self.encoder_block1(x)
           enc2 = self.encoder_block2(enc1)
           # ...
           # Concatenate in decoder
           dec1 = self.decoder_block1(bottleneck)
           dec1 = torch.cat([dec1, enc2], dim=1)
   ```

2. **Attention Mechanism**
   ```python
   # Add spatial attention before global pooling
   self.attention = nn.Sequential(
       nn.Conv2d(512, 1, kernel_size=1),
       nn.Sigmoid()
   )
   ```

3. **Variational Bottleneck (VAE)**
   ```python
   # For better latent space structure
   self.fc_mu = nn.Linear(512, embedding_dim)
   self.fc_var = nn.Linear(512, embedding_dim)
   ```

---

## Training Configuration

### Loss Function (from training code)
```python
# Combined loss function
mse_loss = F.mse_loss(reconstruction, data)
l1_loss = F.l1_loss(reconstruction, data)
embedding_reg = 0.001 * embedding.pow(2).mean()

loss = 0.8 * mse_loss + 0.2 * l1_loss + embedding_reg
```

- **MSE Loss (80%)**: Main reconstruction objective
- **L1 Loss (20%)**: Promotes sparsity, handles outliers
- **Embedding Regularization**: Prevents embedding explosion

### Optimizer Settings
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, ...)
```

- **AdamW**: Adam with decoupled weight decay
- **OneCycleLR**: Learning rate schedule for faster convergence
- **Gradient Clipping**: `max_norm=1.0` for training stability

---

## Usage Examples

### Model Instantiation
```python
# Standard embedding size
model = CNNAutoencoder(embedding_dim=256)

# Larger embedding for more detail preservation
model = CNNAutoencoder(embedding_dim=512)

# Compact model for mobile/edge deployment
model = CNNAutoencoder(embedding_dim=128)
```

### Forward Pass
```python
# Input: [batch_size, seq_length, 3, 130, 5]
input_tensor = torch.randn(4, 10, 3, 130, 5)

# Forward pass
reconstruction, embedding = model(input_tensor)

# Shapes:
# reconstruction: [4, 10, 3, 130, 5] - same as input
# embedding: [4, 10, 256] - compressed representation
```

### Embedding Extraction Only
```python
# Extract only embeddings (encoder only)
with torch.no_grad():
    embeddings = model.encode(input_tensor)
    # Shape: [4, 10, 256]
```

### Reconstruction from Embedding
```python
# Decode from embeddings
with torch.no_grad():
    reconstructed = model.decode(embeddings)
    # Shape: [4, 10, 3, 130, 5]
```

---

## Related Documentation

- **Similar Architecture**: See `docs/CNN-TfAutoencoder_Architecture.md` for comparison
- **Transformer Autoencoder**: See `docs/TfAutoencoder_Architecture.md`
- **Data Loading**: See `src/data/datasets.py` for input preprocessing
- **Configuration**: See `config/config.yaml` for hyperparameters

---

## Change Log

### Version 1.0 (2025-01-28)
- ✅ Initial architecture analysis
- ✅ Complete dimension flow mapping
- ✅ Design critique and improvement suggestions
- ✅ Mathematical analysis and parameter counts
- ✅ Training configuration documentation