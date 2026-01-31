"""
Direct CNN Classifier (USMModeCNN-style Architecture)

3-block CNN classifier with global average pooling for position-invariant features.
Based on colleague's USMModeCNN architecture, adapted for 130-depth decimated input.

Key design principles:
- Input format: (B, C, Depth, Pulses) - same as USMModeCNN
- Same-padding preserves edge information
- Progressive depth reduction with temporal preservation
- Global average pooling for position invariance
- Configurable width via width_multiplier

Architecture for (B, 3, 130, 10) input with width_multiplier=1:
    Input           (B, 3, 130, 10)     # (B, C, Depth, Pulses)
    Block1          (B, 16, 65, 10)     # depth halved
    Block2          (B, 32, 32, 10)     # depth halved
    Block3          (B, 64, 16, 5)      # both halved
    GlobalPool      (B, 64, 1, 1)       # spatial removed
    FC              (B, num_classes)

Width multiplier scaling:
    multiplier=1: 16→32→64 channels,  ~48K params (default)
    multiplier=2: 32→64→128 channels, ~187K params
    multiplier=4: 64→128→256 channels, ~740K params
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
        width_multiplier: Channel width multiplier (default: 1)
                         1 = 16→32→64 channels (~48K params)
                         2 = 32→64→128 channels (~187K params)
                         4 = 64→128→256 channels (~740K params)
        kernel_scale: Depth kernel size multiplier (default: 1)
                     Scale kernels to maintain ~10% receptive field at different depths:
                     1 = (13,3)→(7,3)→(5,3) for depth=130
                     2 = (26,3)→(14,3)→(10,3) for depth=260
                     5 = (65,3)→(35,3)→(25,3) for depth=650
                     10 = (130,3)→(70,3)→(50,3) for depth=1300
    """

    def __init__(
        self,
        in_channels=3,
        input_depth=130,
        input_pulses=10,
        num_classes=5,
        dropout=0.5,
        spatial_dropout=0.1,
        width_multiplier=1,
        kernel_scale=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_depth = input_depth
        self.input_pulses = input_pulses
        self.num_classes = num_classes
        self.spatial_dropout_p = spatial_dropout
        self.fc_dropout_p = dropout
        self.width_multiplier = width_multiplier
        self.kernel_scale = kernel_scale

        # Base channel counts scaled by width_multiplier
        c1 = 16 * width_multiplier  # Block 1 output channels
        c2 = 32 * width_multiplier  # Block 2 output channels
        c3 = 64 * width_multiplier  # Block 3 output channels

        # Kernel sizes scaled by kernel_scale (depth dimension only)
        # Base kernels: (13, 3), (7, 3), (5, 3) for depth=130
        k1_depth = 13 * kernel_scale
        k2_depth = 7 * kernel_scale
        k3_depth = 5 * kernel_scale
        # Padding = (kernel - 1) // 2 for same-padding
        p1_depth = (k1_depth - 1) // 2
        p2_depth = (k2_depth - 1) // 2
        p3_depth = (k3_depth - 1) // 2

        # Spatial dropout (applied between conv blocks)
        self.spatial_drop = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

        # Block 1: Downsample depth, preserve pulses
        # Kernel scaled by kernel_scale on depth, fixed 3 on pulses
        # Pool (2, 1): depth halved, pulses preserved
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=(k1_depth, 3), padding=(p1_depth, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Block 2: Continue depth downsampling
        # Pool (2, 1): depth halved, pulses preserved
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(k2_depth, 3), padding=(p2_depth, 1)),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Block 3: Final refinement + temporal reduction
        # Pool (2, 2): both halved
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(k3_depth, 3), padding=(p3_depth, 1)),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Global average pooling (key for position invariance)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head (also scaled by width_multiplier)
        fc_hidden = 64 * width_multiplier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )

        # Feature dimension after global pooling
        self.flatten_dim = c3

        # Store dimensions for logging
        self._dims = {
            'input': (in_channels, input_depth, input_pulses),
            'after_block1': (c1, input_depth // 2, input_pulses),
            'after_block2': (c2, input_depth // 4, input_pulses),
            'after_block3': (c3, input_depth // 8, input_pulses // 2),
            'after_global_pool': (c3, 1, 1),
            'flatten': c3,
            'fc_hidden': fc_hidden,
            'width_multiplier': width_multiplier,
            'kernel_scale': kernel_scale,
            'kernels': {
                'block1': (k1_depth, 3),
                'block2': (k2_depth, 3),
                'block3': (k3_depth, 3)
            }
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate initialization per layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Final classification layer (no ReLU after) uses Xavier
                # Kaiming is designed for ReLU and creates too-large initial logits
                nn.init.xavier_uniform_(m.weight)
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
        print(f"\nWidth multiplier: {self.width_multiplier}x, Kernel scale: {self.kernel_scale}x")
        print(f"Input format: (B, C, Depth, Pulses) = (B, {self.in_channels}, {self.input_depth}, {self.input_pulses})")
        print(f"\nChannel progression: {self._dims['after_block1'][0]} → {self._dims['after_block2'][0]} → {self._dims['after_block3'][0]}")
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
        print(f"\nClassification head: {self._dims['flatten']} -> {self._dims['fc_hidden']} -> {self.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test with dimensions matching h5 data: (B, C, Depth, Pulses)
    print("Testing DirectCNNClassifier (USMModeCNN-style)...")

    # Test default width (multiplier=1)
    print("\n--- Width multiplier = 1 (default) ---")
    model = DirectCNNClassifier(
        in_channels=3,
        input_depth=130,
        input_pulses=10,
        num_classes=5,
        width_multiplier=1
    )
    model.print_architecture()

    x = torch.randn(4, 3, 130, 10)  # (B, C, Depth, Pulses)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    # Test wider model (multiplier=2)
    print("\n--- Width multiplier = 2 (wider) ---")
    model_wide = DirectCNNClassifier(
        in_channels=3,
        input_depth=130,
        input_pulses=10,
        num_classes=5,
        width_multiplier=2
    )
    model_wide.print_architecture()

    logits_wide, features_wide = model_wide.forward_with_features(x)
    print(f"Features shape: {features_wide.shape}")
    assert features_wide.shape == (4, 128), f"Expected (4, 128), got {features_wide.shape}"

    print("\nAll tests passed!")
