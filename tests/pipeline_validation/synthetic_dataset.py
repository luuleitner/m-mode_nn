"""
Synthetic Dataset for Pipeline Validation

Generates trivially separable data to verify the AE -> Embedding -> Classifier pipeline works.

Classes:
    0 (noise):    Random noise with low amplitude
    1 (upward):   Low-frequency sinusoid with positive slope trend
    2 (downward): High-frequency sinusoid with negative slope trend

The patterns are designed to be trivially separable so that:
- The autoencoder can easily reconstruct them
- The latent space will show clear clusters
- The classifier should achieve ~100% accuracy

If the pipeline fails with this data, the bug is in the code, not the data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticModeDataset(Dataset):
    """
    Generates synthetic M-mode-like signals with distinct patterns per class.

    Shape: [batch_size, 3, 130, 18] matching real data format
    - 3 channels (mimicking ultrasound channels)
    - 130 depth samples
    - 18 temporal samples (window size)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_classes: int = 3,
        input_shape: tuple = (3, 130, 18),
        seed: int = 42,
        noise_level: float = 0.1,
        return_soft_labels: bool = True,
        class_distribution: dict = None
    ):
        """
        Args:
            n_samples: Total number of samples to generate
            n_classes: Number of classes (default 3: noise, upward, downward)
            input_shape: Shape of each sample (C, H, W)
            seed: Random seed for reproducibility
            noise_level: Amount of noise to add (0-1)
            return_soft_labels: If True, return one-hot labels; else hard labels
            class_distribution: Dict mapping class_idx to proportion (e.g., {0: 0.9, 1: 0.05, 2: 0.05})
                               If None, uses balanced distribution (1/n_classes each)
        """
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.noise_level = noise_level
        self.return_soft_labels = return_soft_labels
        self.class_distribution = class_distribution

        np.random.seed(seed)

        # Generate all data upfront
        self.data, self.labels = self._generate_all_data()

    def _generate_all_data(self) -> tuple:
        """Generate all synthetic samples."""
        C, H, W = self.input_shape

        all_data = []
        all_labels = []

        # Determine samples per class based on distribution
        if self.class_distribution is not None:
            # Use specified distribution
            samples_per_class = {}
            total_assigned = 0
            for class_idx in range(self.n_classes):
                proportion = self.class_distribution.get(class_idx, 0.0)
                count = int(self.n_samples * proportion)
                samples_per_class[class_idx] = count
                total_assigned += count

            # Assign remainder to largest class
            remainder = self.n_samples - total_assigned
            if remainder > 0:
                largest_class = max(samples_per_class, key=samples_per_class.get)
                samples_per_class[largest_class] += remainder
        else:
            # Balanced distribution
            base_count = self.n_samples // self.n_classes
            samples_per_class = {i: base_count for i in range(self.n_classes)}

            # Handle remainder
            remainder = self.n_samples - (base_count * self.n_classes)
            for i in range(remainder):
                samples_per_class[i % self.n_classes] += 1

        # Generate samples for each class
        for class_idx in range(self.n_classes):
            for _ in range(samples_per_class[class_idx]):
                sample = self._generate_sample(class_idx)
                all_data.append(sample)
                all_labels.append(class_idx)

        # Shuffle
        indices = np.random.permutation(len(all_data))
        all_data = [all_data[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

        return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    def _generate_sample(self, class_idx: int) -> np.ndarray:
        """
        Generate a single sample for a given class.

        Class patterns:
            0 (noise): Low-amplitude random noise
            1 (upward): Low-freq sine + positive linear trend
            2 (downward): High-freq sine + negative linear trend
        """
        C, H, W = self.input_shape

        # Create coordinate grids
        y = np.linspace(0, 1, H)  # depth axis
        x = np.linspace(0, 1, W)  # temporal axis
        Y, X = np.meshgrid(y, x, indexing='ij')  # [H, W]

        if class_idx == 0:  # Noise class
            # Low-amplitude random pattern
            base = np.random.randn(H, W) * 0.3
            # Add slight structure (very low frequency)
            base += 0.2 * np.sin(2 * np.pi * Y * 0.5)

        elif class_idx == 1:  # Upward movement
            # Low frequency sinusoid
            freq = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2 * np.pi)
            base = np.sin(2 * np.pi * freq * Y + phase)
            # Add positive slope along temporal axis
            slope = np.random.uniform(0.5, 1.5)
            base += slope * X
            # Add characteristic "upward" diagonal pattern
            base += 0.5 * np.sin(2 * np.pi * (Y - X) * 2)

        elif class_idx == 2:  # Downward movement
            # High frequency sinusoid
            freq = np.random.uniform(5, 8)
            phase = np.random.uniform(0, 2 * np.pi)
            base = np.sin(2 * np.pi * freq * Y + phase)
            # Add negative slope along temporal axis
            slope = np.random.uniform(-1.5, -0.5)
            base += slope * X
            # Add characteristic "downward" diagonal pattern
            base += 0.5 * np.sin(2 * np.pi * (Y + X) * 2)

        # Normalize to [0, 1] range
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)

        # Add noise
        noise = np.random.randn(H, W) * self.noise_level
        base = np.clip(base + noise, 0, 1)

        # Replicate across channels with slight variations
        sample = np.zeros((C, H, W), dtype=np.float32)
        for c in range(C):
            channel_noise = np.random.randn(H, W) * (self.noise_level * 0.5)
            sample[c] = np.clip(base + channel_noise, 0, 1)

        return sample

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Return sample in the same format as FilteredSplitH5Dataset."""
        data = torch.from_numpy(self.data[idx])

        if self.return_soft_labels:
            # One-hot encoding
            label = torch.zeros(self.n_classes, dtype=torch.float32)
            label[self.labels[idx]] = 1.0
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'tokens': data,
            'labels': label
        }


class SyntheticBatchedDataset(Dataset):
    """
    Wrapper that pre-batches the synthetic dataset to match production format.

    The production dataset returns pre-batched data, so this mimics that behavior.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        batch_size: int = 50,
        class_distribution: dict = None,
        **kwargs
    ):
        """
        Args:
            n_samples: Total samples
            batch_size: Samples per batch
            class_distribution: Dict mapping class_idx to proportion
            **kwargs: Passed to SyntheticModeDataset
        """
        self.base_dataset = SyntheticModeDataset(
            n_samples=n_samples,
            class_distribution=class_distribution,
            **kwargs
        )
        self.batch_size = batch_size
        self.n_batches = (n_samples + batch_size - 1) // batch_size
        self.class_distribution = class_distribution

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> dict:
        """Return a full batch."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.base_dataset))

        batch_data = []
        batch_labels = []

        for idx in range(start_idx, end_idx):
            sample = self.base_dataset[idx]
            batch_data.append(sample['tokens'])
            batch_labels.append(sample['labels'])

        return {
            'tokens': torch.stack(batch_data),
            'labels': torch.stack(batch_labels)
        }


def create_synthetic_splits(
    n_train: int = 600,
    n_val: int = 200,
    n_test: int = 200,
    batch_size: int = 50,
    seed: int = 42,
    class_distribution: dict = None,
    **kwargs
) -> tuple:
    """
    Create train/val/test splits of synthetic data.

    Args:
        n_train, n_val, n_test: Samples per split
        batch_size: Batch size
        seed: Base random seed (each split gets seed + offset)
        class_distribution: Dict mapping class_idx to proportion (e.g., {0: 0.9, 1: 0.05, 2: 0.05})
        **kwargs: Passed to SyntheticBatchedDataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_ds = SyntheticBatchedDataset(
        n_samples=n_train,
        batch_size=batch_size,
        seed=seed,
        class_distribution=class_distribution,
        **kwargs
    )

    val_ds = SyntheticBatchedDataset(
        n_samples=n_val,
        batch_size=batch_size,
        seed=seed + 1000,
        class_distribution=class_distribution,
        **kwargs
    )

    test_ds = SyntheticBatchedDataset(
        n_samples=n_test,
        batch_size=batch_size,
        seed=seed + 2000,
        class_distribution=class_distribution,
        **kwargs
    )

    return train_ds, val_ds, test_ds


# Preset imbalanced distribution matching real data (90% noise, 5% upward, 5% downward)
IMBALANCED_DISTRIBUTION = {0: 0.90, 1: 0.05, 2: 0.05}


def create_imbalanced_splits(
    n_train: int = 600,
    n_val: int = 200,
    n_test: int = 200,
    batch_size: int = 50,
    seed: int = 42,
    **kwargs
) -> tuple:
    """
    Create train/val/test splits with imbalanced class distribution.

    Uses 90% noise, 5% upward, 5% downward to match real data distribution.

    Args:
        n_train, n_val, n_test: Samples per split
        batch_size: Batch size
        seed: Base random seed
        **kwargs: Passed to SyntheticBatchedDataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    return create_synthetic_splits(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
        seed=seed,
        class_distribution=IMBALANCED_DISTRIBUTION,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test
    print("Testing SyntheticModeDataset...")

    ds = SyntheticModeDataset(n_samples=100, seed=42)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"Sample tokens shape: {sample['tokens'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")

    # Check class distribution
    from collections import Counter
    labels = [ds.labels[i] for i in range(len(ds))]
    print(f"Class distribution: {Counter(labels)}")

    print("\nTesting SyntheticBatchedDataset...")
    batched_ds = SyntheticBatchedDataset(n_samples=100, batch_size=25)
    print(f"Batched dataset size: {len(batched_ds)} batches")

    batch = batched_ds[0]
    print(f"Batch tokens shape: {batch['tokens'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")

    print("\nTesting create_synthetic_splits...")
    train_ds, val_ds, test_ds = create_synthetic_splits(
        n_train=300, n_val=100, n_test=100, batch_size=50
    )
    print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("\nAll tests passed!")