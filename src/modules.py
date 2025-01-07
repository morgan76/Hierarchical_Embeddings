import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import math

# Residual 2D Max Pooling Module
class Res2DMaxPoolModule(nn.Module):
    """
    Residual 2D convolutional module with max pooling.
    This module applies two convolutional layers with residual connections, followed by max pooling.
    """
    def __init__(self, input_channels, output_channels, pooling=2):
        """
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            pooling (int or tuple): Kernel size for max pooling.
        """
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ELU()
        self.mp = nn.MaxPool2d(pooling)

        # Residual connection for input-output channel mismatch
        self.diff = input_channels != output_channels
        if self.diff:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
            self.bn_3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        """
        Forward pass for the Res2DMaxPoolModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after convolution, residual addition, and max pooling.
        """
        residual = self.bn_3(self.conv_3(x)) if self.diff else x
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x))
        x = self.relu(x + residual)  # Add residual connection
        x = self.mp(x)  # Max pooling
        return x


# Residual Front End
class ResFrontEnd(nn.Module):
    """
    Residual front-end feature extractor with multiple 2D convolutional layers.
    Processes input harmonics or mel-spectrograms into a latent feature space.
    """
    def __init__(self, conv_ndim=64, nharmonics=1, nmels=64, output_size=32, dropout=0):
        """
        Args:
            conv_ndim (int): Number of convolutional filters.
            nharmonics (int): Number of input harmonic channels.
            nmels (int): Number of mel bands in the input.
            output_size (int): Dimensionality of the output feature vector.
            dropout (float): Dropout rate for regularization.
        """
        super(ResFrontEnd, self).__init__()
        self.input_bn = nn.BatchNorm2d(nharmonics)

        # Three residual convolutional layers with pooling
        self.layer1 = Res2DMaxPoolModule(nharmonics, conv_ndim, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 1))
        
        # Fully connected layer for dimensionality reduction
        fc_dim = nmels // 2 // 2 // 2 * conv_ndim
        self.fc = nn.Linear(fc_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for ResFrontEnd.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nharmonics, nmels, time_steps).
        
        Returns:
            torch.Tensor: Latent feature tensor of shape (batch_size, time_steps, output_size).
        """
        x = self.input_bn(x)  # Normalize input
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Reshape and reduce dimensions
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)  # Reshape to (batch, time_steps, features)
        x = self.dropout(x)
        x = self.fc(x)  # Project to output_size
        return x


# Transformer Modules
class Residual(nn.Module):
    """
    Residual connection module.
    Adds the input to the output of a function (`fn`).
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Pre-normalization module.
    Applies LayerNorm before passing input to a function (`fn`).
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    FeedForward module for transformers.
    Implements a two-layer MLP with GELU activation and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (torch.Tensor): Optional mask for attention.
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into query, key, value
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer encoder with multiple layers of attention and feedforward modules.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        """
        Args:
            dim (int): Dimension of input features.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Dimension of feedforward layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads, dim_head, dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))
            ]) for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
    

class RelativeTransformer(nn.Module):
    """
    Transformer encoder with multiple layers of attention and feedforward modules.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        """
        Args:
            dim (int): Dimension of input features.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Dimension of feedforward layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, RelativeGlobalAttention(d_model=dim, num_heads=heads, max_len=2000, dropout=0.1))),  
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x