import torch
import torch.nn as nn
import torchaudio

from modules import ResFrontEnd, Transformer


class FrameEncoder(nn.Module):
    """
    Audio encoder for processing mel-spectrogram slices and extracting meaningful embeddings.
    Adapted from https://github.com/minzwon/semi-supervised-music-tagging-transformer.
    Copyright (c) 2021 ByteDance. Code developed by Minz Won.
    
    This model combines convolutional layers (frontend) and transformer layers to encode 
    mel-spectrogram slices into a latent embedding space.
    """
    def __init__(
        self,
        n_mels,
        conv_ndim,
        sample_rate,
        n_fft,
        hop_length,
        n_embedding,
        f_min,
        f_max,
        dropout,
        hidden_dim,
        attention_ndim,
        attention_nlayers,
        attention_nheads,
    ):
        """
        Initializes the FrameEncoder.

        Args:
            n_mels (int): Number of mel filterbanks.
            conv_ndim (int): Number of channels in the convolutional frontend.
            sample_rate (int): Sample rate of the audio.
            n_fft (int): FFT size for the mel-spectrogram.
            hop_length (int): Hop length for the mel-spectrogram.
            n_embedding (int): Size of the embedding for positional encoding.
            f_min (float): Minimum frequency for the mel-spectrogram.
            f_max (float): Maximum frequency for the mel-spectrogram.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the final output embedding.
            attention_ndim (int): Dimension of the transformer model.
            attention_nlayers (int): Number of transformer layers.
            attention_nheads (int): Number of attention heads in the transformer.
        """
        super(FrameEncoder, self).__init__()

        # Mel-spectrogram transformation
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            hop_length=hop_length,
            power=2,
        )

        # Convert amplitudes to decibels
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Convolutional frontend for initial feature extraction
        self.frontend = ResFrontEnd(
            conv_ndim=conv_ndim, 
            nharmonics=1, 
            nmels=n_mels, 
            output_size=attention_ndim, 
            dropout=dropout
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Positional embedding for transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_embedding // 4, attention_ndim)
        )

        # Transformer for capturing temporal dependencies
        self.transformer = Transformer(
            dim=attention_ndim,
            depth=attention_nlayers,
            heads=attention_nheads,
            dim_head=attention_ndim // attention_nheads,
            mlp_dim=attention_ndim * 4,
            dropout=dropout,
        )

        # Hidden dimension for the final embedding
        self.hidden_dim = hidden_dim

        # MLP head to project the transformer output to the desired hidden dimension
        self.mlp_head = nn.Linear(attention_ndim, hidden_dim)

    def forward(self, x):
        """
        Forward pass of the FrameEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps).

        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, hidden_dim).
        """
        # Step 1: Transform audio into mel-spectrogram and convert to decibels
        x = self.spec(x)  # Shape: (batch_size, n_mels, time_steps)
        x = self.amplitude_to_db(x)  # Shape: (batch_size, n_mels, time_steps)
        x = x.unsqueeze(1)  # Add channel dimension. Shape: (batch_size, 1, n_mels, time_steps)

        # Step 2: Apply convolutional frontend
        x = self.frontend(x)  # Shape: (batch_size, num_frames, attention_ndim)

        # Step 3: Add positional encoding
        x += self.pos_embedding[:, : x.size(1)]  # Broadcast positional encoding

        # Step 4: Apply dropout for regularization
        x = self.dropout(x)

        # Step 5: Process with transformer layers
        x = self.transformer(x)  # Shape: (batch_size, num_frames, attention_ndim)

        # Step 6: Temporal pooling (mean pooling along the frame dimension)
        x = x.mean(dim=1).squeeze(1)  # Shape: (batch_size, attention_ndim)

        # Step 7: Project to hidden_dim using MLP head
        x = self.mlp_head(x)  # Shape: (batch_size, hidden_dim)

        return x
    


