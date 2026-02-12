"""
U-Net Autoencoder for Ultrasound M-mode Data

A vanilla U-Net architecture with skip connections for better reconstruction quality.
Preserves fine details through concatenation-based skip connections.

Input format: [Batch, US_Channels, Depth, Pulses] = [B, 3, 130, 18]

Supports optional classification head for joint reconstruction + classification training.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    MLP classification head for embedding-based classification.
    """

    def __init__(self, embedding_dim, num_classes=3, hidden_dim=None, dropout=0.3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embedding):
        return self.classifier(embedding)


class EncoderBlock(nn.Module):
    """
    Encoder block: Conv (stride=2 downsample) + Conv (refine) with BatchNorm, LeakyReLU,
    and optional spatial dropout (Dropout2d).
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True, spatial_dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if spatial_dropout > 0:
            layers.append(nn.Dropout2d(spatial_dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block: ConvTranspose (upsample) + Conv (after skip concat) with BatchNorm and LeakyReLU.

    Input channels are doubled due to skip connection concatenation.
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        # Upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # After concatenation with skip, channels are doubled
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            # Crop or pad to match
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]

            if diff_h > 0 or diff_w > 0:
                # x is smaller, pad it
                x = nn.functional.pad(x, [0, diff_w, 0, diff_h])
            elif diff_h < 0 or diff_w < 0:
                # x is larger, crop it
                x = x[:, :, :skip.shape[2], :skip.shape[3]]

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetAutoencoder(nn.Module):
    """
    U-Net Autoencoder with skip connections for ultrasound M-mode data.

    Features:
    - Skip connections preserve fine details (edges, texture)
    - Configurable depth and channel sizes
    - Embedding bottleneck for downstream classification
    - LeakyReLU activation throughout

    Input:  [B, C, H, W] = [Batch, US_Channels, Depth, Pulses]
    Output: (reconstruction, embedding)

    Args:
        in_channels: Number of input channels (default: 3 for US channels)
        input_height: Height of input (default: 130 for depth samples)
        input_width: Width of input (default: 18 for pulses)
        channels: List of channel sizes for encoder levels (default: [32, 64, 128, 256])
        embedding_dim: Size of bottleneck embedding (default: 512)
        use_batchnorm: Whether to use batch normalization (default: True)
        num_classes: Number of classes for classification head (default: 3, set to 0 to disable)
        classifier_dropout: Dropout rate for classification head (default: 0.3)
        spatial_dropout: Dropout2d rate for encoder blocks (default: 0.0, disabled)
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_height: int = 130,
        input_width: int = 18,
        channels: list = None,
        embedding_dim: int = 512,
        use_batchnorm: bool = True,
        num_classes: int = 3,
        classifier_dropout: float = 0.3,
        spatial_dropout: float = 0.0,
        skip_drop_prob: float = 0.0,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.use_batchnorm = use_batchnorm
        self.num_levels = len(channels)
        self.num_classes = num_classes
        self.skip_drop_prob = skip_drop_prob

        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        ch_in = in_channels
        for ch_out in channels:
            self.encoder_blocks.append(EncoderBlock(ch_in, ch_out, use_batchnorm, spatial_dropout))
            ch_in = ch_out

        # Calculate bottleneck size
        self._bottleneck_h, self._bottleneck_w = self._get_bottleneck_size()
        self._flat_size = channels[-1] * self._bottleneck_h * self._bottleneck_w

        # Bottleneck: flatten → embedding → unflatten
        self.fc_encode = nn.Linear(self._flat_size, embedding_dim)
        self.fc_decode = nn.Sequential(
            nn.Linear(embedding_dim, self._flat_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build decoder blocks (reverse order)
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            ch_in = reversed_channels[i]
            ch_out = reversed_channels[i + 1]
            self.decoder_blocks.append(DecoderBlock(ch_in, ch_out, use_batchnorm))

        # Final decoder block and output layer
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels[0]) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1)

        # Classification head (optional)
        if num_classes > 0:
            self.classifier = ClassificationHead(
                embedding_dim, num_classes, dropout=classifier_dropout
            )
        else:
            self.classifier = None

    def _get_bottleneck_size(self):
        """Calculate spatial size at bottleneck after all encoder blocks."""
        h, w = self.input_height, self.input_width
        for _ in self.channels:
            h = (h + 1) // 2  # stride=2 with padding=1
            w = (w + 1) // 2
        return h, w

    def encode(self, x):
        """
        Encode input to embedding.

        Returns:
            embedding: [B, embedding_dim]
        """
        # Pass through encoder, collecting skip connections
        skips = []
        h = x
        for encoder_block in self.encoder_blocks:
            h = encoder_block(h)
            skips.append(h)

        # Flatten and project to embedding
        h_flat = h.view(h.size(0), -1)
        embedding = self.fc_encode(h_flat)

        return embedding

    def encode_with_skips(self, x):
        """
        Encode input to embedding, also returning skip connections for decoder.

        Returns:
            embedding: [B, embedding_dim]
            skips: List of intermediate feature maps
        """
        skips = []
        h = x
        for encoder_block in self.encoder_blocks:
            h = encoder_block(h)
            skips.append(h)

        h_flat = h.view(h.size(0), -1)
        embedding = self.fc_encode(h_flat)

        return embedding, skips

    def decode(self, embedding, skips=None):
        """
        Decode embedding to reconstruction.

        Args:
            embedding: [B, embedding_dim]
            skips: List of skip connections (required for full reconstruction quality)

        Returns:
            reconstruction: [B, in_channels, H, W]
        """
        # Unflatten
        h = self.fc_decode(embedding)
        h = h.view(-1, self.channels[-1], self._bottleneck_h, self._bottleneck_w)

        # Pass through decoder with skip connections
        if skips is not None:
            # Use skip connections (skips are in encoder order, need to reverse for decoder)
            reversed_skips = list(reversed(skips[:-1]))  # Skip the last one (bottleneck input)

            for decoder_block, skip in zip(self.decoder_blocks, reversed_skips):
                h = decoder_block(h, skip)
        else:
            # No skip connections - forces embedding to carry all information
            for decoder_block in self.decoder_blocks:
                h = decoder_block.upsample(h)
                dummy_skip = torch.zeros_like(h)
                h = torch.cat([h, dummy_skip], dim=1)
                h = decoder_block.conv(h)

        # Final upsample and output
        h = self.final_upsample(h)

        # Crop to exact input size
        h = h[:, :, :self.input_height, :self.input_width]

        reconstruction = self.final_conv(h)

        return reconstruction

    def forward(self, x):
        """
        Forward pass.

        Uses skip connections for best reconstruction quality.

        Returns:
            If classifier enabled: (reconstruction, embedding, logits)
            If classifier disabled: (reconstruction, embedding)
        """
        embedding, skips = self.encode_with_skips(x)

        # Skip dropout: during training, randomly drop ALL skip connections
        # to force the embedding to carry spatial/structural information
        if self.training and self.skip_drop_prob > 0 and torch.rand(1).item() < self.skip_drop_prob:
            reconstruction = self.decode(embedding, skips=None)
        else:
            reconstruction = self.decode(embedding, skips)

        if self.classifier is not None:
            logits = self.classifier(embedding)
            return reconstruction, embedding, logits

        return reconstruction, embedding

    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    # Test with default config (classification enabled)
    model = UNetAutoencoder()
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Channels: {model.channels}")
    print(f"Embedding dim: {model.embedding_dim}")
    print(f"Bottleneck size: {model._bottleneck_h}x{model._bottleneck_w}")

    # Test forward pass
    x = torch.randn(4, 3, 130, 18)
    recon, emb, logits = model(x)
    print(f"\nInput:          {x.shape}")
    print(f"Embedding:      {emb.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"Logits:         {logits.shape}")

    # Test encode only
    emb_only = model.encode(x)
    print(f"Encode only:    {emb_only.shape}")

    # Verify reconstruction matches input size
    assert recon.shape == x.shape, f"Shape mismatch: {recon.shape} vs {x.shape}"

    # Test without classification head
    model2 = UNetAutoencoder(num_classes=0)
    print(f"\nModel without classifier: {model2.get_num_parameters():,}")
    recon2, emb2 = model2(x)
    print(f"Embedding:      {emb2.shape}")
    print(f"Reconstruction: {recon2.shape}")

    print("\n✓ All tests passed!")