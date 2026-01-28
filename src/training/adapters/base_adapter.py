"""
Base Adapter Interface - Adapters handle model-specific data preparation and visualization.
"""

from abc import ABC, abstractmethod
import torch.nn.functional as F


class BaseAdapter(ABC):
    """Abstract base class for model-specific adapters."""

    @property
    @abstractmethod
    def input_format(self):
        """Return string describing expected input format, e.g., '[B,C,H,W]'"""
        pass

    @abstractmethod
    def prepare_batch(self, batch, device):
        """Prepare batch from dataloader for model. Returns tensor ready for forward pass."""
        pass

    @abstractmethod
    def get_input_info(self, data):
        """Extract and return dict with shape information for logging."""
        pass

    @abstractmethod
    def visualize_reconstruction(self, original, reconstruction, save_path, n_samples=3):
        """Visualize original vs reconstructed samples."""
        pass

    @abstractmethod
    def visualize_samples(self, original, reconstruction, save_path, n_samples=2):
        """Visualize detailed sample comparisons (e.g., line plots)."""
        pass

    def compute_metrics(self, original, reconstruction):
        """Compute reconstruction metrics. Override for custom metrics."""
        mse = F.mse_loss(reconstruction, original).item()
        mae = F.l1_loss(reconstruction, original).item()
        return {'mse': mse, 'mae': mae}
