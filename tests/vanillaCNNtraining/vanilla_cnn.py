"""
Vanilla CNN Classifier - Minimal version for debugging.

No dropout, no regularization, just pure convolutions.
Input: (B, 3, 130, 10) -> Output: (B, 5) logits
"""

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    """
    Minimal 3-block CNN classifier.

    Architecture:
        Input           (B, 3, 130, 10)
        Block1          (B, 16, 65, 10)   - conv + bn + relu + pool
        Block2          (B, 32, 32, 10)   - conv + bn + relu + pool
        Block3          (B, 64, 16, 5)    - conv + bn + relu + pool
        GlobalPool      (B, 64)           - adaptive avg pool + flatten
        FC              (B, 5)            - linear
    """

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        # Block 1: (B, 3, 130, 10) -> (B, 16, 65, 10)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(13, 3), padding=(6, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Block 2: (B, 16, 65, 10) -> (B, 32, 32, 10)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Block 3: (B, 32, 32, 10) -> (B, 64, 16, 5)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (no hidden layer, no dropout - minimal)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Quick test
    model = VanillaCNN()
    x = torch.randn(4, 3, 130, 10)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
