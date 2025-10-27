#!/usr/bin/env python3
"""
CNN-Transformer Hybrid Model for Ultrasound M-Mode Classification
Integrates the CNN autoencoder encoder with transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


# ============================================================================
# CNN ENCODER EXTRACTION
# ============================================================================

class CNNEncoder(nn.Module):
    """Extracted and adapted CNN encoder from the autoencoder"""

    def __init__(self, embedding_dim=256, freeze_weights=False):
        super().__init__()

        # Width reduction layer (from original autoencoder)
        self.width_reducer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Encoder backbone (from original autoencoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Option to freeze CNN weights
        if freeze_weights:
            self.freeze_cnn_weights()

    def freeze_cnn_weights(self):
        """Freeze CNN parameters to prevent updates during transformer training"""
        for param in self.parameters():
            param.requires_grad = False
        print("CNN encoder weights frozen")

    def unfreeze_cnn_weights(self):
        """Unfreeze CNN parameters for joint training"""
        for param in self.parameters():
            param.requires_grad = True
        print("CNN encoder weights unfrozen")

    def forward(self, x):
        """
        Forward pass for CNN encoder
        Input: x of shape (B, T, C, H, W) where:
            B = batch size
            T = sequence length (time steps)
            C = channels (3)
            H = height (130)
            W = width (5)
        Output: embeddings of shape (B, T, embedding_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape to process all frames at once
        x = x.view(B * T, C, H, W)

        # Apply CNN layers
        x = self.width_reducer(x)
        embedding = self.encoder(x)

        # Reshape back to sequence format
        return embedding.view(B, T, -1)


# ============================================================================
# ENHANCED TRANSFORMER CLASSIFIER
# ============================================================================

class CNNTransformerClassifier(nn.Module):
    """
    Hybrid CNN-Transformer model for ultrasound M-mode classification
    """

    def __init__(self,
                 seq_length=10,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=4,
                 num_classes=4,
                 dropout=0.1,
                 cnn_freeze=False,
                 use_cls_token=True):
        super().__init__()

        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # CNN Feature Extractor
        self.cnn_encoder = CNNEncoder(
            embedding_dim=embed_dim,
            freeze_weights=cnn_freeze
        )

        # Positional Encoding
        self.register_buffer("pos_encoding", self._generate_positional_encoding())

        # CLS token (optional)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            effective_seq_len = seq_length + 1
        else:
            effective_seq_len = seq_length

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Increased feedforward dim
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _generate_positional_encoding(self):
        """Generate sinusoidal positional encodings"""
        pe = torch.zeros(self.seq_length, self.embed_dim)
        position = torch.arange(0, self.seq_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                             -(math.log(10000.0) / self.embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Add batch dimension

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_attention=False):
        """
        Forward pass
        Input: x of shape (B, T, C, H, W)
        Output: class logits of shape (B, num_classes)
        """
        batch_size = x.shape[0]

        # Extract CNN features
        cnn_features = self.cnn_encoder(x)  # (B, T, embed_dim)

        # Add positional encoding
        x = cnn_features + self.pos_encoding.expand(batch_size, -1, -1)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, embed_dim)

        # Apply transformer
        if return_attention:
            # For attention visualization (requires modification of transformer)
            transformer_out = self.transformer(x)
            attention_weights = None  # Would need custom transformer for this
        else:
            transformer_out = self.transformer(x)

        # Classification
        if self.use_cls_token:
            # Use CLS token for classification
            cls_output = transformer_out[:, 0, :]  # (B, embed_dim)
        else:
            # Use mean pooling over sequence
            cls_output = transformer_out.mean(dim=1)  # (B, embed_dim)

        logits = self.classifier(cls_output)  # (B, num_classes)

        if return_attention:
            return logits, attention_weights
        return logits

    def load_pretrained_cnn(self, autoencoder_path):
        """Load pretrained CNN weights from autoencoder"""
        checkpoint = torch.load(autoencoder_path, map_location='cpu')
        autoencoder_state = checkpoint['model_state_dict']

        # Extract CNN encoder weights
        cnn_state = {}
        for key, value in autoencoder_state.items():
            if key.startswith('width_reducer') or key.startswith('encoder'):
                cnn_state[key] = value

        # Load weights
        missing_keys, unexpected_keys = self.cnn_encoder.load_state_dict(cnn_state, strict=False)

        print(f"Loaded pretrained CNN weights:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")

        return self


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class TrainingConfig:
    """Configuration class for training parameters"""

    def __init__(self):
        # Model parameters
        self.seq_length = 10
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        self.num_classes = 4
        self.dropout = 0.1

        # Training parameters
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.epochs = 50
        self.warmup_epochs = 5

        # Training strategy
        self.cnn_freeze_epochs = 10  # Freeze CNN for first N epochs
        self.gradient_clip = 1.0

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_optimizer_scheduler(model, config):
    """Create optimizer and learning rate scheduler"""

    # Separate CNN and transformer parameters for different learning rates
    cnn_params = []
    transformer_params = []

    for name, param in model.named_parameters():
        if 'cnn_encoder' in name:
            cnn_params.append(param)
        else:
            transformer_params.append(param)

    # Different learning rates for different components
    optimizer = optim.AdamW([
        {'params': transformer_params, 'lr': config.learning_rate},
        {'params': cnn_params, 'lr': config.learning_rate * 0.1}  # Lower LR for CNN
    ], weight_decay=config.weight_decay)

    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.epochs // 3, T_mult=1, eta_min=1e-6
    )

    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Unfreeze CNN after specified epochs
    if epoch == config.cnn_freeze_epochs:
        model.cnn_encoder.unfreeze_cnn_weights()

    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch + 1}')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(config.device), targets.to(config.device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
        scheduler.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate_epoch(model, dataloader, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(config.device), targets.to(config.device)

            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def create_dummy_data(num_samples=1000, seq_length=10, num_classes=4):
    """Create dummy ultrasound data for testing"""

    # Generate synthetic ultrasound sequences
    data = torch.randn(num_samples, seq_length, 3, 130, 5)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return data, labels


def main_example():
    """Example usage of the CNN-Transformer model"""

    print("=" * 60)
    print("CNN-TRANSFORMER HYBRID MODEL EXAMPLE")
    print("=" * 60)

    # Configuration
    config = TrainingConfig()
    print(f"Training on device: {config.device}")

    # Create model
    model = CNNTransformerClassifier(
        seq_length=config.seq_length,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        cnn_freeze=True,  # Start with frozen CNN
        use_cls_token=True
    ).to(config.device)

    # Load pretrained CNN weights (uncomment when available)
    # model.load_pretrained_cnn('trained_autoencoder.pth')

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy data
    print(f"\nCreating dummy data...")
    train_data, train_labels = create_dummy_data(800)
    val_data, val_labels = create_dummy_data(200)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Setup training
    optimizer, scheduler = create_optimizer_scheduler(model, config)

    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, config)

        # Record history
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch + 1}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print()

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, config.epochs + 1)

    ax1.plot(epochs_range, train_history['loss'], 'b-', label='Train')
    ax1.plot(epochs_range, val_history['loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_history['acc'], 'b-', label='Train')
    ax2.plot(epochs_range, val_history['acc'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cnn_transformer_training.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training completed!")
    print(f"Final validation accuracy: {val_history['acc'][-1]:.2f}%")

    return model, train_history, val_history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run example
    model, train_hist, val_hist = main_example()