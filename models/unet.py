"""
Tiny UNet architecture for 32x32 maze diffusion.
Conditional on timestep and unsolved maze image.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timestep tensor (batch_size,)
        Returns:
            Embeddings (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, height, width)
            time_emb: Time embedding (batch, time_emb_dim)
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + residual


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple:
        """
        Returns: (downsampled, skip_connection)
        """
        x = self.res_block(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous layer
            skip: Skip connection from encoder
            time_emb: Time embedding
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, time_emb)
        return x


class TinyUNet(nn.Module):
    """
    Tiny UNet for 32x32 maze diffusion.
    Takes 2-channel input: noisy solved maze + unsolved maze (conditioning).
    Outputs 1-channel: predicted noise.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        # 32x32 -> 16x16 -> 8x8
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)  # 64 -> 64, skip: 64
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)  # 64 -> 128, skip: 128
        
        # Bottleneck (8x8)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)  # 128
        
        # Decoder (upsampling)
        # 8x8 -> 16x16 -> 32x32
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)  # 128 + skip 128 -> 64
        self.up2 = UpBlock(base_channels, base_channels, base_channels, time_emb_dim)  # 64 + skip 64 -> 64
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, 2, 32, 32) - noisy + conditioning
            timestep: Timestep tensor (batch,)
        Returns:
            Predicted noise (batch, 1, 32, 32)
        """
        # Get time embeddings
        t_emb = self.time_mlp(timestep)
        
        # Initial convolution
        x = self.init_conv(x)  # (batch, 64, 32, 32)
        
        # Encoder
        x, skip1 = self.down1(x, t_emb)  # (batch, 64, 16, 16), skip: (batch, 64, 32, 32)
        x, skip2 = self.down2(x, t_emb)  # (batch, 128, 8, 8), skip: (batch, 128, 16, 16)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)  # (batch, 128, 8, 8)
        
        # Decoder
        x = self.up1(x, skip2, t_emb)  # (batch, 64, 16, 16)
        x = self.up2(x, skip1, t_emb)  # (batch, 64, 32, 32)
        
        # Final convolution
        x = self.final_conv(x)  # (batch, 1, 32, 32)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing TinyUNet...\n")
    
    # Create model
    model = TinyUNet(
        in_channels=2,
        out_channels=1,
        base_channels=64,
        time_emb_dim=128
    )
    
    print(f"1. Model info:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Input channels: {model.in_channels}")
    print(f"  Output channels: {model.out_channels}")
    
    # Test forward pass
    print(f"\n2. Testing forward pass:")
    batch_size = 4
    x = torch.randn(batch_size, 2, 32, 32)  # 2 channels: noisy + conditioning
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"  Input shape: {x.shape}")
    print(f"  Timestep shape: {t.shape}")
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test different batch sizes
    print(f"\n3. Testing different batch sizes:")
    for bs in [1, 8, 32]:
        x = torch.randn(bs, 2, 32, 32)
        t = torch.randint(0, 1000, (bs,))
        with torch.no_grad():
            output = model(x, t)
        print(f"  Batch size {bs}: input {x.shape} -> output {output.shape}")
    
    print("\n✓ UNet test complete!")

