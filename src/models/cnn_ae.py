"""
Vanilla CNN Autoencoder for Ultrasound M-mode Data

Input format: [Batch, US_Channels, Depth, Pulses] = [B, 3, 130, 18]
"""

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    Vanilla CNN Autoencoder with configurable architecture.

    Input:  [B, C, H, W] = [Batch, US_Channels, Depth, Pulses]
    Output: (reconstruction, embedding)

    Args:
        in_channels: Number of input channels (default: 3 for US channels)
        input_height: Height of input (default: 130 for depth samples)
        input_width: Width of input (default: 18 for pulses)
        channels: List of channel sizes for encoder layers (default: [16, 32, 64])
        embedding_dim: Size of bottleneck embedding (default: 256)
        use_batchnorm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_height: int = 130,
        input_width: int = 18,
        channels: list = None,
        embedding_dim: int = 256,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]

        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.use_batchnorm = use_batchnorm

        # Build encoder
        encoder_layers = []
        ch_in = in_channels
        for ch_out in channels:
            encoder_layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm2d(ch_out))
            encoder_layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate flattened size after conv layers
        self._flat_size, self._enc_h, self._enc_w = self._get_conv_output_size()

        # Encoder projection to embedding
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, embedding_dim),
        )

        # Decoder projection from embedding
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, self._flat_size),
            nn.ReLU(inplace=True),
        )

        # Build decoder (mirror of encoder)
        decoder_layers = []
        reversed_channels = list(reversed(channels))
        for i, (ch_in, ch_out) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            decoder_layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm2d(ch_out))
            decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_conv = nn.Sequential(*decoder_layers)

        # Final layer to reconstruct original channels
        self.decoder_final = nn.ConvTranspose2d(
            channels[0], in_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def _get_conv_output_size(self):
        """Calculate output size after encoder conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            out = self.encoder_conv(dummy)
            return out.numel(), out.shape[2], out.shape[3]

    def encode(self, x):
        """Encode input to embedding."""
        h = self.encoder_conv(x)
        embedding = self.encoder_fc(h)
        return embedding

    def decode(self, embedding):
        """Decode embedding to reconstruction."""
        h = self.decoder_fc(embedding)
        h = h.view(-1, self.channels[-1], self._enc_h, self._enc_w)
        h = self.decoder_conv(h)
        reconstruction = self.decoder_final(h)
        # Crop to exact input size (handles odd dimensions)
        reconstruction = reconstruction[:, :, :self.input_height, :self.input_width]
        return reconstruction

    def forward(self, x):
        """Forward pass returning (reconstruction, embedding)."""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding


# Quick test
if __name__ == "__main__":
    # Test with default config
    model = CNNAutoencoder()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 3, 130, 18)
    recon, emb = model(x)
    print(f"Input:          {x.shape}")
    print(f"Embedding:      {emb.shape}")
    print(f"Reconstruction: {recon.shape}")

    # Test with custom config
    model2 = CNNAutoencoder(channels=[32, 64, 128, 256], embedding_dim=512)
    print(f"\nDeeper model parameters: {sum(p.numel() for p in model2.parameters()):,}")
    recon2, emb2 = model2(x)
    print(f"Embedding:      {emb2.shape}")
    print(f"Reconstruction: {recon2.shape}")