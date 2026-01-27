import numpy as np


class SoftLabelGenerator:
    """
    Converts per-sample hard labels into per-token soft label probability distributions.

    Hard labels: [pulses] array with values {0=noise, 1=up, 2=down}
    Soft labels: [num_tokens, num_classes] float32 probability distributions
    """

    def __init__(self, num_classes: int, weighting: str = "gaussian", gaussian_sigma_ratio: float = 0.25):
        """
        Args:
            num_classes: Number of label classes (e.g., 3 for noise/up/down)
            weighting: Weighting scheme for aggregation - "uniform" or "gaussian"
            gaussian_sigma_ratio: For gaussian weighting, sigma = window_size * ratio
        """
        self.num_classes = num_classes
        self.weighting = weighting
        self.sigma_ratio = gaussian_sigma_ratio

        if weighting not in ("uniform", "gaussian"):
            raise ValueError(f"weighting must be 'uniform' or 'gaussian', got '{weighting}'")

    def create_soft_labels(self, hard_labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
        """
        Convert per-sample hard labels to per-token soft label distributions.

        Args:
            hard_labels: [pulses] array with integer class labels (0, 1, 2, ...)
            window_size: Token window size (number of samples per token)
            stride: Token stride (step between consecutive tokens)

        Returns:
            soft_labels: [num_tokens, num_classes] float32 array where each row sums to 1.0
        """
        num_samples = len(hard_labels)
        num_tokens = (num_samples - window_size) // stride + 1

        if num_tokens <= 0:
            raise ValueError(f"Not enough samples ({num_samples}) for window_size={window_size}")

        weights = self._get_weights(window_size)
        soft_labels = np.zeros((num_tokens, self.num_classes), dtype=np.float32)

        for i in range(num_tokens):
            start = i * stride
            end = start + window_size
            window_labels = hard_labels[start:end]

            for c in range(self.num_classes):
                soft_labels[i, c] = np.sum(weights * (window_labels == c))

        return soft_labels

    def _get_weights(self, window_size: int) -> np.ndarray:
        """
        Generate weights for aggregating labels within a window.

        Args:
            window_size: Size of the token window

        Returns:
            weights: [window_size] array that sums to 1.0
        """
        if self.weighting == "uniform":
            return np.ones(window_size, dtype=np.float32) / window_size
        else:  # gaussian
            center = (window_size - 1) / 2.0
            sigma = window_size * self.sigma_ratio
            positions = np.arange(window_size, dtype=np.float32)
            weights = np.exp(-((positions - center) ** 2) / (2 * sigma ** 2))
            return weights / weights.sum()


def window_hard_labels(hard_labels: np.ndarray, window_size: int, stride: int,
                       method: str = "majority") -> np.ndarray:
    """
    Convert per-sample hard labels to per-token hard labels using windowing.

    Args:
        hard_labels: [pulses] array with integer class labels
        window_size: Token window size
        stride: Token stride
        method: Aggregation method - "majority" (most common) or "center" (center sample)

    Returns:
        token_labels: [num_tokens] integer array
    """
    num_samples = len(hard_labels)
    num_tokens = (num_samples - window_size) // stride + 1

    if num_tokens <= 0:
        raise ValueError(f"Not enough samples ({num_samples}) for window_size={window_size}")

    token_labels = np.zeros(num_tokens, dtype=np.int64)

    for i in range(num_tokens):
        start = i * stride
        end = start + window_size
        window_labels = hard_labels[start:end]

        if method == "majority":
            # Most common label in window
            counts = np.bincount(window_labels.astype(np.int64))
            token_labels[i] = np.argmax(counts)
        else:  # center
            # Label at center of window
            center_idx = window_size // 2
            token_labels[i] = window_labels[center_idx]

    return token_labels
