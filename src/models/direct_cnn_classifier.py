"""
Direct CNN Classifier (USMModeCNN-style Architecture)

3-block CNN classifier with global average pooling for position-invariant features.
Based on colleague's USMModeCNN architecture, adapted for 130-depth decimated input.

Key design principles:
- Input format: (B, C, Depth, Pulses) - same as USMModeCNN
- Same-padding preserves edge information
- Progressive depth reduction with temporal preservation
- Global average pooling for position invariance
- Compact FC head (64 features)

Architecture for (B, 3, 130, 10) input:
    Input           (B, 3, 130, 10)     # (B, C, Depth, Pulses)
    Block1          (B, 16, 65, 10)     # depth halved
    Block2          (B, 32, 32, 10)     # depth halved
    Block3          (B, 64, 16, 5)      # both halved
    GlobalPool      (B, 64, 1, 1)       # spatial removed
    FC              (B, num_classes)
"""

import torch
import torch.nn as nn


class DirectCNNClassifier(nn.Module):
    """
    USMModeCNN-style classifier with 3 conv blocks and global average pooling.

    Input format: (B, C, Depth, Pulses) - matches USMModeCNN convention.

    Kernel design follows USMModeCNN philosophy:
    - Larger kernels along depth (spatial patterns)
    - Smaller kernels along pulses (temporal preservation)

    Args:
        in_channels: Number of input channels (default: 3)
        input_depth: Spatial dimension / depth samples (default: 130)
        input_pulses: Temporal dimension / number of pulses (default: 10)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout probability for FC head (default: 0.5)
        spatial_dropout: Dropout2d probability between conv blocks (default: 0.1)
    """

    def __init__(
        self,
        in_channels=3,
        input_depth=130,
        input_pulses=10,
        num_classes=5,
        dropout=0.5,
        spatial_dropout=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_depth = input_depth
        self.input_pulses = input_pulses
        self.num_classes = num_classes
        self.spatial_dropout_p = spatial_dropout
        self.fc_dropout_p = dropout

        # Spatial dropout (applied between conv blocks)
        self.spatial_drop = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

        # Block 1: Downsample depth, preserve pulses
        # Kernel (13, 3): 13 across depth (spatial), 3 across pulses (temporal)
        # Pool (2, 1): depth halved, pulses preserved
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(13, 3), padding=(6, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))  # depth: 130->65, pulses: 10->10
        )

        # Block 2: Continue depth downsampling
        # Kernel (7, 3): 7 across depth, 3 across pulses
        # Pool (2, 1): depth halved, pulses preserved
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))  # depth: 65->32, pulses: 10->10
        )

        # Block 3: Final refinement + temporal reduction
        # Kernel (5, 3): 5 across depth, 3 across pulses
        # Pool (2, 2): both halved
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # depth: 32->16, pulses: 10->5
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
            'input': (in_channels, input_depth, input_pulses),
            'after_block1': (16, input_depth // 2, input_pulses),
            'after_block2': (32, input_depth // 4, input_pulses),
            'after_block3': (64, input_depth // 8, input_pulses // 2),
            'after_global_pool': (64, 1, 1),
            'flatten': 64,
            'kernels': {
                'block1': (13, 3),
                'block2': (7, 3),
                'block3': (5, 3)
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

        Args:
            x: Input tensor (B, C, Depth, Pulses)

        Returns:
            Feature tensor (B, 64)
        """
        x = self.block1(x)
        x = self.spatial_drop(x)
        x = self.block2(x)
        x = self.spatial_drop(x)
        x = self.block3(x)
        x = self.spatial_drop(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        Full forward pass.

        Args:
            x: Input tensor (B, C, Depth, Pulses)

        Returns:
            logits: Classification logits (B, num_classes)
        """
        features = self.forward_backbone(x)
        logits = self.fc(features)
        return logits

    def forward_with_features(self, x):
        """
        Forward pass returning both logits and features.

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
        print(f"\nInput format: (B, C, Depth, Pulses) = (B, {self.in_channels}, {self.input_depth}, {self.input_pulses})")
        print(f"\nKernel sizes (Depth, Pulses):")
        print(f"  Block1: {self._dims['kernels']['block1']} - large depth, small temporal")
        print(f"  Block2: {self._dims['kernels']['block2']}")
        print(f"  Block3: {self._dims['kernels']['block3']}")
        print(f"\nRegularization:")
        print(f"  Spatial dropout (between blocks): {self.spatial_dropout_p}")
        print(f"  FC dropout (classifier head):     {self.fc_dropout_p}")
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
    # Test with dimensions matching h5 data: (B, C, Depth, Pulses)
    print("Testing DirectCNNClassifier (USMModeCNN-style)...")

    model = DirectCNNClassifier(
        in_channels=3,
        input_depth=130,
        input_pulses=10,
        num_classes=5
    )
    model.print_architecture()

    # Test forward pass with h5 data format
    x = torch.randn(4, 3, 130, 10)  # (B, C, Depth, Pulses)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    logits, features = model.forward_with_features(x)
    print(f"Features shape: {features.shape}")
    assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"

    print("\nAll tests passed!")
