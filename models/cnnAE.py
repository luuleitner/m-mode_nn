#!/usr/bin/env python3
"""
Standalone training script for CNN Autoencoder with width reduction.
Includes model definitions and training/visualization code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import os

from utils.utils import load_config
from data.loader import create_filtered_split_datasets

import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class CNNAutoencoder(nn.Module):
    """Optimized CNN autoencoder with width reduction to 1"""

    def __init__(self, embedding_dim=256):
        super().__init__()

        # Width reduction layer
        self.width_reducer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Encoder backbone
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

        # Decoder
        self.decoder_projection = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 5),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 1), stride=(2, 1),
                               padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(2, 1),
                               padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1),
                               padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(3, 1),
                               padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.width_restorer = nn.Sequential(
            nn.Conv2d(32, 32 * 5, kernel_size=(1, 1)),
            nn.PixelShuffle(1),
        )

        self.final_projection = nn.Conv2d(32, 3, kernel_size=(1, 1))

    def encode(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.width_reducer(x)
        embedding = self.encoder(x)
        return embedding.view(B, T, -1)

    def decode(self, embedding):
        B, T, _ = embedding.shape
        x = self.decoder_projection(embedding.view(B * T, -1))
        x = x.view(B * T, 512, 5, 1)
        x = self.decoder(x)
        x = self.width_restorer(x)
        x = x.view(B * T, 32, 130, 5)
        x = self.final_projection(x)
        return x.view(B, T, 3, 130, 5)

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding


# ============================================================================
# DATA GENERATION
# ============================================================================

def create_synthetic_ultrasound_data(n_samples=100, add_artifacts=True):
    """Create realistic synthetic ultrasound M-mode data"""

    data = []
    seq_length, n_channels, height, width = 10, 3, 130, 5

    for _ in range(n_samples):
        sample = np.zeros((seq_length, n_channels, height, width))

        # Base tissue properties
        n_layers = np.random.randint(3, 7)
        layer_properties = []

        for _ in range(n_layers):
            layer_properties.append({
                'depth': np.random.randint(10, height - 10),
                'thickness': np.random.randint(5, 20),
                'intensity': np.random.uniform(0.2, 0.9),
                'attenuation': np.random.uniform(0.98, 0.995)
            })

        # Sort layers by depth
        layer_properties.sort(key=lambda x: x['depth'])

        for t in range(seq_length):
            # Simulate cardiac cycle
            cardiac_phase = np.sin(2 * np.pi * t / seq_length)

            for c in range(n_channels):
                # Create depth-dependent pattern
                signal = np.ones((height, width)) * 0.1

                for layer in layer_properties:
                    depth = layer['depth'] + int(cardiac_phase * 3)  # Motion
                    thickness = layer['thickness']
                    intensity = layer['intensity']

                    # Apply depth-dependent attenuation
                    for d in range(max(0, depth - thickness // 2),
                                   min(height, depth + thickness // 2)):
                        atten_factor = layer['attenuation'] ** (d / height)
                        signal[d, :] = intensity * atten_factor

                        # Add slight variation across scanlines
                        for w in range(width):
                            signal[d, w] += np.random.normal(0, 0.05)

                # Add speckle noise (characteristic of ultrasound)
                speckle = np.random.gamma(2, 0.2, (height, width))
                signal = signal * speckle

                # Add artifacts if requested
                if add_artifacts and np.random.random() < 0.3:
                    # Add shadow artifact
                    shadow_start = np.random.randint(30, height - 30)
                    signal[shadow_start:, :] *= 0.3

                # Normalize and clip
                signal = np.clip(signal, 0, 1)

                # Apply smoothing for realism
                from scipy.ndimage import gaussian_filter1d
                signal = gaussian_filter1d(signal, sigma=1.5, axis=0)

                sample[t, c] = signal

        data.append(sample)

    return torch.FloatTensor(np.array(data))


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def load_or_create_datasets(load_data_pickle_flag, train_path, val_path, test_path, dataset_parameters):
    if load_data_pickle_flag:
        logger.info("Loading datasets from pickle files...")
        with open(train_path, 'rb') as f:
            train_ds = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_ds = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_ds = pickle.load(f)
    else:
        logger.info("Initializing train, validation and test dataset from metadata file...")
        train_ds, test_ds, val_ds = create_filtered_split_datasets(**dataset_parameters)
        logger.info("Saving datasets in pickle files...")
        with open(train_path, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_ds, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_ds, f)

    return train_ds, val_ds, test_ds

def train_with_visualization(model, epochs=10, batch_size=16, lr=1e-3, device='cuda'):
    """Complete training pipeline with visualization"""

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # # Generate data
    # print("\nGenerating synthetic ultrasound data...")
    # train_data = create_synthetic_ultrasound_data(n_samples=200)
    # val_data = create_synthetic_ultrasound_data(n_samples=50)
    # test_data = create_synthetic_ultrasound_data(n_samples=20)

    # # ---------------------------------------------
    # # Load Dataset
    config = load_config(config_file='/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml')
    # Set Datahandling
    load_data_pickle_flag = config.ml.loading.load_data_pickle
    pickle_path = config.ml.dataset.data_root
    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')
    load_data_pickle_flag=True
    dataset_parameters = config.ml.dataset
    train_ds, val_ds, test_ds = load_or_create_datasets(load_data_pickle_flag, train_path, val_path, test_path, dataset_parameters)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # Each item is already a batch of 200
        shuffle=True,  # Shuffle batches across epochs
        num_workers=2,  # Parallel loading
        pin_memory=True,  # Faster GPU transfer
        drop_last=False  # Keep last incomplete batch
    )

    sequence_length = test_ds[0].shape[1]
    channels_nbr = test_ds[0].shape[2]
    height = test_ds[0].shape[3]
    width = test_ds[0].shape[4]


    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,  # No shuffling for validation
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,  # No shuffling for test
        num_workers=2,
        pin_memory=True
    )
    logger.info("DONE.")



    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_mse': []}

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in pbar:
            data = batch[0].to(device)

            # Forward pass
            reconstruction, embedding = model(data)

            # Calculate losses
            mse_loss = F.mse_loss(reconstruction, data)
            l1_loss = F.l1_loss(reconstruction, data)
            embedding_reg = 0.001 * embedding.pow(2).mean()

            # Combined loss
            loss = 0.8 * mse_loss + 0.2 * l1_loss + embedding_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        val_loss = 0
        val_mse = 0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                reconstruction, embedding = model(data)

                mse = F.mse_loss(reconstruction, data)
                loss = mse + 0.001 * embedding.pow(2).mean()

                val_loss += loss.item()
                val_mse += mse.item()

        # Record metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_mse = val_mse / len(val_loader)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_mse'].append(avg_mse)

        print(f"Epoch {epoch + 1}: Train={avg_train:.4f}, Val={avg_val:.4f}, MSE={avg_mse:.4f}")

    # Final evaluation and visualization
    print("\nTraining completed! Generating visualizations...")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, epochs + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history['val_mse'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Validation MSE')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Visualize reconstructions
    model.eval()
    with torch.no_grad():
        test_batch = test_data[:3].to(device)
        reconstructions, embeddings = model(test_batch)

        test_batch = test_batch.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()

    # Create detailed visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i in range(3):
        t = 5  # Middle time step
        c = 0  # First channel

        # Original M-mode image
        ax = axes[i, 0]
        im = ax.imshow(test_batch[i, t, c].T, aspect='auto', cmap='gray',
                       extent=[0, 130, 0, 5], origin='lower')
        ax.set_title(f'Original Sample {i + 1}')
        ax.set_xlabel('Depth (pixels)')
        ax.set_ylabel('Scanline')
        plt.colorbar(im, ax=ax)

        # Reconstruction
        ax = axes[i, 1]
        im = ax.imshow(reconstructions[i, t, c].T, aspect='auto', cmap='gray',
                       extent=[0, 130, 0, 5], origin='lower')
        ax.set_title(f'Reconstruction')
        ax.set_xlabel('Depth (pixels)')
        ax.set_ylabel('Scanline')
        plt.colorbar(im, ax=ax)

        # Difference map
        ax = axes[i, 2]
        diff = np.abs(test_batch[i, t, c] - reconstructions[i, t, c])
        im = ax.imshow(diff.T, aspect='auto', cmap='hot',
                       extent=[0, 130, 0, 5], origin='lower')
        ax.set_title(f'Absolute Difference (MSE: {np.mean(diff ** 2):.4f})')
        ax.set_xlabel('Depth (pixels)')
        ax.set_ylabel('Scanline')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot individual scanlines
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(2):  # Two samples
        for w in range(3):  # Three scanlines
            ax = axes[i, w]

            original = test_batch[i, 5, 0, :, w]
            reconstructed = reconstructions[i, 5, 0, :, w]

            ax.set_title(f'Sample {i + 1}, Scanline {w + 1}')
            ax.set_xlabel('Depth (pixels)')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add MSE annotation
            mse = np.mean((original - reconstructed) ** 2)
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('scanline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Test on final test set
    print("\nFinal test set evaluation:")
    test_mse_total = 0
    test_mae_total = 0

    model.eval()
    with torch.no_grad():
        for i in range(len(test_data)):
            data = test_data[i:i + 1].to(device)
            reconstruction, _ = model(data)

            mse = F.mse_loss(reconstruction, data).item()
            mae = F.l1_loss(reconstruction, data).item()

            test_mse_total += mse
            test_mae_total += mae

    avg_test_mse = test_mse_total / len(test_data)
    avg_test_mae = test_mae_total / len(test_data)

    print(f"Average Test MSE: {avg_test_mse:.6f}")
    print(f"Average Test MAE: {avg_test_mae:.6f}")

    # Embedding visualization
    print("\nVisualizing embeddings...")

    # Get embeddings for visualization
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(min(50, len(test_data))):
            data = test_data[i:i + 1].to(device)
            _, embedding = model(data)
            all_embeddings.append(embedding.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, T, 256]

    # Plot embedding statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Embedding magnitude over time
    embedding_norms = np.linalg.norm(all_embeddings, axis=2)  # [N, T]
    mean_norms = np.mean(embedding_norms, axis=0)
    std_norms = np.std(embedding_norms, axis=0)

    time_steps = range(1, 11)
    ax1.plot(time_steps, mean_norms, 'b-', linewidth=2)
    ax1.fill_between(time_steps, mean_norms - std_norms, mean_norms + std_norms,
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Embedding L2 Norm')
    ax1.set_title('Embedding Magnitude Over Time')
    ax1.grid(True, alpha=0.3)

    # Embedding distribution
    ax2.hist(all_embeddings.flatten(), bins=50, density=True, alpha=0.7, color='green')
    ax2.set_xlabel('Embedding Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Embedding Value Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete training pipeline"""

    print("=" * 60)
    print("CNN AUTOENCODER TRAINING - ULTRASOUND M-MODE DATA")
    print("=" * 60)

    # Check for scipy (needed for data generation)
    try:
        import scipy
    except ImportError:
        print("\nWarning: scipy not found. Installing basic version...")
        print("Run: pip install scipy")
        # Use simpler data generation without scipy

    # Initialize model
    print("\nInitializing OptimizedWidthReducedAutoencoder...")
    model = CNNAutoencoder(embedding_dim=256)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024 ** 2):.2f} MB (float32)")

    # Run training
    print("\nStarting training pipeline...")
    trained_model, history = train_with_visualization(
        model=model,
        epochs=10,
        batch_size=16,
        lr=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Save model
    print("\nSaving trained model...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'model_config': {
            'embedding_dim': 256,
            'input_shape': [10, 3, 130, 5]
        }
    }, 'trained_autoencoder.pth')

    print("\nTraining complete! Model saved as 'trained_autoencoder.pth'")
    print("\nGenerated visualizations:")
    print("- training_curves.png: Loss curves over epochs")
    print("- reconstruction_comparison.png: Input vs output comparison")
    print("- scanline_comparison.png: Individual scanline reconstructions")
    print("- embedding_analysis.png: Embedding statistics")

    return trained_model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the training
    model, history = main()

    # Keep matplotlib windows open
    print("\nPress Ctrl+C to exit...")
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nExiting...")