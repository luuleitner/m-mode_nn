"""
Transformer Autoencoder for Ultrasound M-Mode Data
Transformer-based alternative to CNN autoencoder for sequence reconstruction.
"""

import torch
import torch.nn as nn
import math

import utils.logging_config as logconf
logger = logconf.get_logger("TransformerAE")


# ============================================================================
# TRANSFORMER AUTOENCODER MODEL
# ============================================================================

class TransformerAutoencoder(nn.Module):
    """Transformer-based autoencoder for ultrasound sequences"""
    
    def __init__(self, 
                 seq_length=10,
                 input_channels=3,
                 height=130,
                 width=5,
                 embedding_dim=256,
                 num_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        self.seq_length = seq_length
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim
        
        # Calculate flattened input size
        self.input_size = input_channels * height * width
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=seq_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Bottleneck - optional compression
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, self.input_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, x):
        """
        Encode input sequences
        Input: x of shape (B, T, C, H, W)
        Output: latent of shape (B, T, embedding_dim)
        """
        B, T, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.view(B, T, -1)  # (B, T, C*H*W)
        
        # Project to embedding dimension
        x = self.input_projection(x)  # (B, T, embedding_dim)
        
        # Add temporal positional encoding
        #TODO: add PE of tokem in seqence
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        memory = self.transformer_encoder(x)  # (B, T, embedding_dim)
        
        # Apply bottleneck
        latent = self.bottleneck(memory)  # (B, T, embedding_dim)
        
        return latent
    
    def decode(self, latent):
        """
        Decode latent representation
        Input: latent of shape (B, T, embedding_dim)
        Output: reconstruction of shape (B, T, C, H, W)
        """
        B, T, _ = latent.shape
        
        # Use latent as both target and memory for decoder
        # This creates an autoregressive-like reconstruction
        output = self.transformer_decoder(latent, latent)  # (B, T, embedding_dim)
        
        # Project back to original dimensions
        output = self.output_projection(output)  # (B, T, C*H*W)
        
        # Reshape to original format
        output = output.view(B, T, self.input_channels, self.height, self.width)
        
        return output
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        Input: x of shape (B, T, C, H, W)
        Output: (reconstruction, latent)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# CNN-TRANSFORMER AUTOENCODER (CNN ENCODING VERSION)
# ============================================================================

class CNNTransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder that uses CNN for initial spatial encoding
    instead of linear projection, then transformer for temporal processing
    """
    
    def __init__(self, 
                 seq_length=10,
                 embedding_dim=256,
                 num_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        
        # CNN Encoder for spatial features (exact copy from CNNAutoencoder)
        self.width_reducer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=seq_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Bottleneck - optional compression
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # CNN Decoder (exact copy from CNNAutoencoder)
        self.decoder_projection = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 5),
            nn.ReLU(inplace=True)
        )
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 1), stride=(2, 1),
                             padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(2, 1),
                             padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1),
                             padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(3, 1),
                             padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.width_restorer = nn.Sequential(
            nn.Conv2d(32, 32 * 5, kernel_size=(1, 1)),
            nn.PixelShuffle(1),
        )
        
        self.final_projection = nn.Conv2d(32, 3, kernel_size=(1, 1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, x):
        """
        Encode input sequences using CNN + Transformer
        Input: x of shape (B, T, C, H, W)
        Output: latent of shape (B, T, embedding_dim)
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through CNN encoder
        x = x.view(B * T, C, H, W)  # Flatten batch and time
        x = self.width_reducer(x)   # Width reduction
        cnn_features = self.cnn_encoder(x)  # CNN encoding: (B*T, embedding_dim)
        
        # Reshape back to sequence format
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoder(cnn_features)
        
        # Apply transformer encoder for temporal modeling
        memory = self.transformer_encoder(x)  # (B, T, embedding_dim)
        
        # Apply bottleneck
        latent = self.bottleneck(memory)  # (B, T, embedding_dim)
        
        return latent
    
    def decode(self, latent):
        """
        Decode latent representation using Transformer + CNN
        Input: latent of shape (B, T, embedding_dim)
        Output: reconstruction of shape (B, T, C, H, W)
        """
        B, T, _ = latent.shape
        
        # Use latent as both target and memory for transformer decoder
        output = self.transformer_decoder(latent, latent)  # (B, T, embedding_dim)
        
        # Process each frame through CNN decoder (like CNNAutoencoder.decode)
        output = output.view(B * T, -1)  # Flatten for processing
        x = self.decoder_projection(output)  # Linear projection
        x = x.view(B * T, 512, 5, 1)  # Reshape for deconv
        x = self.cnn_decoder(x)  # Deconvolutions
        x = self.width_restorer(x)  # Width restoration
        x = x.view(B * T, 32, 130, 5)  # Reshape after width restoration
        x = self.final_projection(x)  # Final projection to 3 channels
        
        # Reshape to original format
        return x.view(B, T, 3, 130, 5)
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        Input: x of shape (B, T, C, H, W)
        Output: (reconstruction, latent)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent