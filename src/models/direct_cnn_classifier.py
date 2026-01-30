"""
Direct CNN Classifier

Simple 2-conv CNN classifier inspired by colleague's architecture.
Designed for direct classification without autoencoder bottleneck.

Key differences from AE approach:
- No reconstruction objective (focused gradients)
- Large flattened feature space (not compressed embedding)
- Larger receptive fields for pattern capture

Architecture for (B, 3, 10, 130) input:
    Input           (B, 3, 10, 130)
    Conv1 (1,13)    (B, 32, 10, 118)    # Depth pattern extraction
    Pool1 (1,3)     (B, 32, 10, 39)
    Conv2 (5,9)     (B, 64, 6, 31)      # Temporal + depth patterns
    Pool2 (1,3)     (B, 64, 6, 10)
    Flatten         (B, 3840)           # Large feature space
    FC              (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectCNNClassifier(nn.Module):
    """
    Direct CNN classifier without autoencoder bottleneck.

    Adapted from colleague's architecture for smaller input depth.
    Preserves large feature dimensionality for classification.

    Args:
        in_channels: Number of input channels (default: 3)
        input_pulses: Temporal dimension / number of pulses (default: 10)
        input_depth: Spatial dimension / depth samples (default: 130)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        use_batchnorm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_pulses: int = 10,
        input_depth: int = 130,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_pulses = input_pulses
        self.input_depth = input_depth
        self.num_classes = num_classes

        # Calculate kernel sizes based on input depth
        # Colleague used (1,51) for 1000 depth = 5.1% coverage
        # Scale proportionally but keep minimum useful size
        depth_kernel1 = max(7, int(input_depth * 0.10))  # ~10% of depth
        depth_kernel2 = max(5, int(input_depth * 0.07))  # ~7% of depth

        # Conv Block 1: Extract depth patterns (preserve temporal)
        # Kernel (1, K) operates only on depth dimension
        self.conv1 = nn.Conv2d(
            in_channels, 32,
            kernel_size=(1, depth_kernel1),
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Calculate intermediate dimensions
        depth_after_conv1 = input_depth - depth_kernel1 + 1
        depth_after_pool1 = depth_after_conv1 // 3

        # Conv Block 2: Extract temporal + depth patterns
        # Kernel (5, K) operates on both dimensions
        temporal_kernel = min(5, input_pulses)  # Don't exceed input
        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=(temporal_kernel, depth_kernel2),
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Calculate final dimensions
        pulses_after_conv2 = input_pulses - temporal_kernel + 1
        depth_after_conv2 = depth_after_pool1 - depth_kernel2 + 1
        depth_after_pool2 = max(1, depth_after_conv2 // 3)

        self.flatten_dim = 64 * pulses_after_conv2 * depth_after_pool2

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Store dimensions for logging
        self._dims = {
            'input': (in_channels, input_pulses, input_depth),
            'after_conv1': (32, input_pulses, depth_after_conv1),
            'after_pool1': (32, input_pulses, depth_after_pool1),
            'after_conv2': (64, pulses_after_conv2, depth_after_conv2),
            'after_pool2': (64, pulses_after_conv2, depth_after_pool2),
            'flatten': self.flatten_dim,
            'kernels': {
                'conv1': (1, depth_kernel1),
                'conv2': (temporal_kernel, depth_kernel2)
            }
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward_backbone(self, x):
        """
        Forward pass through backbone (conv layers).
        Returns flattened features before classification head.

        Args:
            x: Input tensor (B, C, Pulses, Depth)

        Returns:
            Flattened feature tensor (B, flatten_dim)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        """
        Full forward pass.

        Args:
            x: Input tensor (B, C, Pulses, Depth)

        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Backbone
        features = self.forward_backbone(x)

        # Classification head
        x = self.dropout(features)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def forward_with_features(self, x):
        """
        Forward pass returning both logits and features.
        Useful for embedding extraction after training.

        Returns:
            tuple: (logits, features)
        """
        features = self.forward_backbone(x)

        x = self.dropout(features)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, features

    def get_embedding_dim(self):
        """Return the dimension of the backbone features."""
        return self.flatten_dim

    def print_architecture(self):
        """Print architecture summary."""
        print("\n" + "=" * 70)
        print("DirectCNNClassifier Architecture")
        print("=" * 70)
        print(f"\nInput dimensions: {self._dims['input']}")
        print(f"Kernel sizes: Conv1={self._dims['kernels']['conv1']}, Conv2={self._dims['kernels']['conv2']}")
        print(f"\nFeature flow:")
        print(f"  Input:       {self._dims['input']}")
        print(f"  After Conv1: {self._dims['after_conv1']}")
        print(f"  After Pool1: {self._dims['after_pool1']}")
        print(f"  After Conv2: {self._dims['after_conv2']}")
        print(f"  After Pool2: {self._dims['after_pool2']}")
        print(f"  Flattened:   {self._dims['flatten']} features")
        print(f"\nClassification head: {self._dims['flatten']} -> 256 -> {self.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 70 + "\n")


class DirectCNNClassifierLarge(DirectCNNClassifier):
    """
    Larger variant with 3 conv blocks for more capacity.

    Use this if the 2-block version underfits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_pulses: int = 10,
        input_depth: int = 130,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        # Don't call parent __init__, we redefine everything
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self.input_pulses = input_pulses
        self.input_depth = input_depth
        self.num_classes = num_classes

        # Three conv blocks with increasing channels
        # Conv1: depth patterns only
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(1, 7), padding=0)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Conv2: small temporal + depth
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=0)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Conv3: larger temporal + depth
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), padding=0)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Calculate dimensions
        d = input_depth
        p = input_pulses

        d = (d - 7 + 1) // 2  # after conv1 + pool1
        d = (d - 5 + 1) // 2  # after conv2 + pool2
        p = p - 3 + 1         # after conv2
        d = (d - 5 + 1) // 2  # after conv3 + pool3
        p = p - 3 + 1         # after conv3

        self.flatten_dim = 128 * p * d

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self._dims = {'flatten': self.flatten_dim}
        self._init_weights()

    def forward_backbone(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.pool3(x)

        return x.view(x.size(0), -1)

    def forward(self, x):
        features = self.forward_backbone(x)

        x = self.dropout(features)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits


if __name__ == "__main__":
    # Test with default dimensions
    print("Testing DirectCNNClassifier...")

    model = DirectCNNClassifier(
        in_channels=3,
        input_pulses=10,
        input_depth=130,
        num_classes=3
    )
    model.print_architecture()

    # Test forward pass
    x = torch.randn(4, 3, 10, 130)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    logits, features = model.forward_with_features(x)
    print(f"Features shape: {features.shape}")

    # Test large variant
    print("\n\nTesting DirectCNNClassifierLarge...")
    model_large = DirectCNNClassifierLarge(
        in_channels=3,
        input_pulses=10,
        input_depth=130,
        num_classes=3
    )
    print(f"Large model flatten dim: {model_large.flatten_dim}")
    print(f"Large model parameters: {sum(p.numel() for p in model_large.parameters()):,}")

    logits = model_large(x)
    print(f"Output shape: {logits.shape}")
