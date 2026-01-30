"""
Direct CNN Classifier (USMModeCNN-style Architecture)

3-block CNN classifier with global average pooling for position-invariant features.
Inspired by colleague's USMModeCNN architecture, adapted for 130-depth decimated input.

Key design principles:
- Same-padding preserves edge information
- Progressive depth reduction with temporal preservation
- Global average pooling for position invariance (~25K params vs ~2.5M)
- Compact FC head (64 features)

Architecture for (B, 3, 10, 130) input:
    Input           (B, 3, 10, 130)
    Block1          (B, 16, 10, 65)    # depth halved
    Block2          (B, 32, 10, 32)    # depth halved
    Block3          (B, 64, 5, 16)     # both halved
    GlobalPool      (B, 64, 1, 1)      # spatial removed
    FC              (B, num_classes)
"""

import torch
import torch.nn as nn


class DirectCNNClassifier(nn.Module):
    """
    USMModeCNN-style classifier with 3 conv blocks and global average pooling.

    Adapted from colleague's architecture for smaller input depth (130 vs 1000).
    Uses same-padding to preserve edge information and global pooling for
    position-invariant features.

    Args:
        in_channels: Number of input channels (default: 3)
        input_pulses: Temporal dimension / number of pulses (default: 10)
        input_depth: Spatial dimension / depth samples (default: 130)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        in_channels=3,
        input_pulses=10,
        input_depth=130,
        num_classes=3,
        dropout=0.3,
        use_batchnorm=True  # kept for API compatibility, always uses batchnorm
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_pulses = input_pulses
        self.input_depth = input_depth
        self.num_classes = num_classes

        # Block 1: Preserve temporal, downsample depth
        # Kernel (3,13) with padding (1,6) → same output size
        # Pool (1,2) → depth: 130→65
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 13), padding=(1, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # Block 2: Continue depth downsampling
        # Kernel (3,7) with padding (1,3) → same output size
        # Pool (1,2) → depth: 65→32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # Block 3: Final refinement + temporal reduction
        # Kernel (3,5) with padding (1,2) → same output size
        # Pool (2,2) → depth: 32→16, pulses: 10→5
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Global average pooling (key for position invariance)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact classifier head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Feature dimension after global pooling
        self.flatten_dim = 64

        # Store dimensions for logging
        self._dims = {
            'input': (in_channels, input_pulses, input_depth),
            'after_block1': (16, input_pulses, input_depth // 2),
            'after_block2': (32, input_pulses, input_depth // 4),
            'after_block3': (64, input_pulses // 2, input_depth // 8),
            'after_global_pool': (64, 1, 1),
            'flatten': 64,
            'kernels': {
                'block1': (3, 13),
                'block2': (3, 7),
                'block3': (3, 5)
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
        Forward pass through backbone (conv layers + global pool).
        Returns pooled features before classification head.

        Args:
            x: Input tensor (B, C, Pulses, Depth)

        Returns:
            Feature tensor (B, 64)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
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
        features = self.forward_backbone(x)
        logits = self.fc(features)
        return logits

    def forward_with_features(self, x):
        """
        Forward pass returning both logits and features.
        Useful for embedding extraction after training.

        Returns:
            tuple: (logits, features)
        """
        features = self.forward_backbone(x)
        logits = self.fc(features)
        return logits, features

    def get_embedding_dim(self):
        """Return the dimension of the backbone features."""
        return self.flatten_dim

    def print_architecture(self):
        """Print architecture summary."""
        print("\n" + "=" * 70)
        print("DirectCNNClassifier Architecture (USMModeCNN-style)")
        print("=" * 70)
        print(f"\nInput dimensions: {self._dims['input']}")
        print(f"Kernel sizes: Block1={self._dims['kernels']['block1']}, "
              f"Block2={self._dims['kernels']['block2']}, "
              f"Block3={self._dims['kernels']['block3']}")
        print(f"\nFeature flow:")
        print(f"  Input:            {self._dims['input']}")
        print(f"  After Block1:     {self._dims['after_block1']}")
        print(f"  After Block2:     {self._dims['after_block2']}")
        print(f"  After Block3:     {self._dims['after_block3']}")
        print(f"  After GlobalPool: {self._dims['after_global_pool']}")
        print(f"  Features:         {self._dims['flatten']}")
        print(f"\nClassification head: {self._dims['flatten']} -> 64 -> {self.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test with default dimensions
    print("Testing DirectCNNClassifier (USMModeCNN-style)...")

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
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"

    logits, features = model.forward_with_features(x)
    print(f"Features shape: {features.shape}")
    assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"

    print("\nAll tests passed!")
