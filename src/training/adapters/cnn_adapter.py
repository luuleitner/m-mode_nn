"""
CNN Adapter - Handles CNN autoencoder with input [B, C, H, W] = [Batch, 3, Pulses, Depth]

Note: Data is transposed from storage format [B, C, Depth, Pulses] to model format
[B, C, Pulses, Depth] to preserve temporal resolution through encoder pooling.
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_adapter import BaseAdapter


class CNNAdapter(BaseAdapter):
    """Adapter for CNN autoencoder with 4D input [B, C, H, W]."""

    def __init__(self, log_compression_db=65.0, transpose_hw=True):
        self.log_compression_db = log_compression_db
        self.transpose_hw = transpose_hw  # Transpose H/W for better temporal preservation

    @property
    def input_format(self):
        if self.transpose_hw:
            return "[B, C, H, W] = [Batch, US_Channels, Pulses, Depth] (transposed)"
        return "[B, C, H, W] = [Batch, US_Channels, Depth, Pulses]"

    def prepare_batch(self, batch, device):
        """
        Prepare batch for CNN model.
        Handles dict format {'tokens': tensor, 'labels': tensor}, tuples, or plain tensors.

        If transpose_hw=True, transposes from storage [B,C,Depth,Pulses] to [B,C,Pulses,Depth]
        to preserve temporal resolution through encoder pooling layers.

        Returns:
            tuple: (data, labels) where labels may be None
        """
        if isinstance(batch, dict):
            data = batch['tokens'].to(device)
            labels = batch['labels'].to(device) if batch['labels'] is not None else None
        elif isinstance(batch, (list, tuple)):
            data = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
        else:
            data = batch.to(device)
            labels = None

        # Transpose H/W: [B, C, Depth, Pulses] → [B, C, Pulses, Depth]
        if self.transpose_hw:
            data = data.permute(0, 1, 3, 2)

        return data, labels

    def get_input_info(self, data):
        """Extract shape information for logging."""
        if len(data.shape) != 4:
            return {'shape': data.shape, 'warning': f'Expected 4D [B,C,H,W], got {len(data.shape)}D'}

        B, C, H, W = data.shape
        if self.transpose_hw:
            # After transpose: H=Pulses, W=Depth
            return {
                'shape': list(data.shape),
                'batch_size': B,
                'channels': C,
                'pulses': H,
                'depth_samples': W,
                'transposed': True,
                'memory_mb': (data.numel() * data.element_size()) / (1024 * 1024)
            }
        return {
            'shape': list(data.shape),
            'batch_size': B,
            'channels': C,
            'depth_samples': H,
            'pulses': W,
            'transposed': False,
            'memory_mb': (data.numel() * data.element_size()) / (1024 * 1024)
        }

    def _log_compress(self, data):
        """Apply log compression for visualization."""
        eps = 1e-10
        data_pos = np.abs(data) + eps
        data_db = 20 * np.log10(data_pos / data_pos.max())
        data_db = np.clip(data_db, -self.log_compression_db, 0)
        return (data_db + self.log_compression_db) / self.log_compression_db

    def visualize_reconstruction(self, original, reconstruction, save_path, n_samples=3):
        """Visualize M-mode reconstructions as heatmaps."""
        original = original.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()

        # Transpose back for visualization if needed: [B,C,Pulses,Depth] → [B,C,Depth,Pulses]
        if self.transpose_hw:
            original = np.transpose(original, (0, 1, 3, 2))
            reconstruction = np.transpose(reconstruction, (0, 1, 3, 2))

        n_samples = min(n_samples, original.shape[0])

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            orig = self._log_compress(original[i, 0])
            recon = self._log_compress(reconstruction[i, 0])

            # Original
            im1 = axes[i, 0].imshow(orig, aspect='auto', cmap='gray', origin='lower')
            axes[i, 0].set_title(f'Original (Sample {i+1})')
            axes[i, 0].set_xlabel('Pulse')
            axes[i, 0].set_ylabel('Depth')
            plt.colorbar(im1, ax=axes[i, 0])

            # Reconstruction
            im2 = axes[i, 1].imshow(recon, aspect='auto', cmap='gray', origin='lower')
            axes[i, 1].set_title('Reconstruction')
            axes[i, 1].set_xlabel('Pulse')
            axes[i, 1].set_ylabel('Depth')
            plt.colorbar(im2, ax=axes[i, 1])

            # Difference
            diff = np.abs(orig - recon)
            mse = np.mean((original[i, 0] - reconstruction[i, 0]) ** 2)
            im3 = axes[i, 2].imshow(diff, aspect='auto', cmap='hot', origin='lower')
            axes[i, 2].set_title(f'|Difference| (MSE: {mse:.6f})')
            axes[i, 2].set_xlabel('Pulse')
            axes[i, 2].set_ylabel('Depth')
            plt.colorbar(im3, ax=axes[i, 2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_samples(self, original, reconstruction, save_path, n_samples=2):
        """Visualize A-line comparisons (depth profiles at specific pulses)."""
        original = original.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()

        # Transpose back for visualization if needed: [B,C,Pulses,Depth] → [B,C,Depth,Pulses]
        if self.transpose_hw:
            original = np.transpose(original, (0, 1, 3, 2))
            reconstruction = np.transpose(reconstruction, (0, 1, 3, 2))

        n_samples = min(n_samples, original.shape[0])
        n_pulses = 3

        fig, axes = plt.subplots(n_samples, n_pulses, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        pulse_width = original.shape[3]
        pulse_indices = [0, pulse_width // 2, pulse_width - 1]

        for i in range(n_samples):
            for j, pulse_idx in enumerate(pulse_indices):
                ax = axes[i, j]
                orig_line = original[i, 0, :, pulse_idx]
                recon_line = reconstruction[i, 0, :, pulse_idx]

                ax.plot(orig_line, 'b-', label='Original', linewidth=2)
                ax.plot(recon_line, 'r--', label='Reconstruction', linewidth=2)
                ax.set_title(f'Sample {i+1}, Pulse {pulse_idx}')
                ax.set_xlabel('Depth')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)

                mse = np.mean((orig_line - recon_line) ** 2)
                ax.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_all_channels(self, original, reconstruction, save_path, sample_idx=0):
        """Visualize all 3 US channels for a single sample."""
        original = original.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()

        # Transpose back for visualization if needed: [B,C,Pulses,Depth] → [B,C,Depth,Pulses]
        if self.transpose_hw:
            original = np.transpose(original, (0, 1, 3, 2))
            reconstruction = np.transpose(reconstruction, (0, 1, 3, 2))

        n_channels = min(3, original.shape[1])

        fig, axes = plt.subplots(n_channels, 3, figsize=(12, 4 * n_channels))

        for ch in range(n_channels):
            orig = self._log_compress(original[sample_idx, ch])
            recon = self._log_compress(reconstruction[sample_idx, ch])

            im1 = axes[ch, 0].imshow(orig, aspect='auto', cmap='gray', origin='lower')
            axes[ch, 0].set_title(f'Original (Ch {ch+1})')
            plt.colorbar(im1, ax=axes[ch, 0])

            im2 = axes[ch, 1].imshow(recon, aspect='auto', cmap='gray', origin='lower')
            axes[ch, 1].set_title(f'Reconstruction (Ch {ch+1})')
            plt.colorbar(im2, ax=axes[ch, 1])

            diff = np.abs(orig - recon)
            mse = np.mean((original[sample_idx, ch] - reconstruction[sample_idx, ch]) ** 2)
            im3 = axes[ch, 2].imshow(diff, aspect='auto', cmap='hot', origin='lower')
            axes[ch, 2].set_title(f'|Diff| Ch {ch+1} (MSE: {mse:.6f})')
            plt.colorbar(im3, ax=axes[ch, 2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
