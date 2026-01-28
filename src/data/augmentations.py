"""
Signal Augmentation Module for Ultrasound Data

Provides augmentation transforms for oversampling minority classes.
Augmentations are semantically preserving (movement UP is still UP after augmentation).

Usage:
    augmenter = SignalAugmenter(config)
    augmented_signal = augmenter(original_signal)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import random


class SignalAugmenter:
    """
    Augmenter for 1D ultrasound signals.

    Applies random combinations of:
    - Additive Gaussian noise
    - Amplitude scaling
    - Temporal shifting

    All augmentations are designed to preserve the semantic meaning
    of the signal (movement class remains the same).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """
        Initialize augmenter with configuration.

        Args:
            config: Augmentation configuration dict with structure:
                {
                    'noise': {'enabled': True, 'std': 0.02},
                    'scale': {'enabled': True, 'range': [0.9, 1.1]},
                    'shift': {'enabled': True, 'max_shift': 3}
                }
            seed: Random seed for reproducibility
        """
        config = config or {}

        # Noise configuration
        noise_config = config.get('noise', {})
        self.noise_enabled = noise_config.get('enabled', True)
        self.noise_std = noise_config.get('std', 0.02)

        # Scale configuration
        scale_config = config.get('scale', {})
        self.scale_enabled = scale_config.get('enabled', True)
        self.scale_range = scale_config.get('range', [0.9, 1.1])

        # Shift configuration
        shift_config = config.get('shift', {})
        self.shift_enabled = shift_config.get('enabled', True)
        self.max_shift = shift_config.get('max_shift', 3)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to signal.

        Args:
            signal: Input signal array (any shape, augmentation applied to last axis)

        Returns:
            Augmented signal (same shape as input)
        """
        augmented = signal.copy()

        if self.noise_enabled:
            augmented = self.add_noise(augmented)

        if self.scale_enabled:
            augmented = self.apply_scale(augmented)

        if self.shift_enabled:
            augmented = self.apply_shift(augmented)

        return augmented

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to signal.

        Args:
            signal: Input signal

        Returns:
            Signal with added noise
        """
        noise = np.random.normal(0, self.noise_std, signal.shape)
        return signal + noise

    def apply_scale(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random amplitude scaling.

        Args:
            signal: Input signal

        Returns:
            Scaled signal
        """
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return signal * scale_factor

    def apply_shift(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random temporal shift (roll).

        Args:
            signal: Input signal

        Returns:
            Shifted signal
        """
        shift_amount = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift_amount == 0:
            return signal
        return np.roll(signal, shift_amount, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return current augmentation configuration."""
        return {
            'noise': {
                'enabled': self.noise_enabled,
                'std': self.noise_std
            },
            'scale': {
                'enabled': self.scale_enabled,
                'range': self.scale_range
            },
            'shift': {
                'enabled': self.shift_enabled,
                'max_shift': self.max_shift
            }
        }


def create_augmenter(config: Optional[Dict[str, Any]] = None,
                     seed: Optional[int] = None) -> SignalAugmenter:
    """
    Factory function to create an augmenter.

    Args:
        config: Augmentation configuration
        seed: Random seed

    Returns:
        Configured SignalAugmenter instance
    """
    return SignalAugmenter(config=config, seed=seed)
