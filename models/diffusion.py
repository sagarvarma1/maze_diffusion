"""
Diffusion process components: noise schedule, forward/reverse process.
Implements DDPM and DDIM sampling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better than linear schedule for small images.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta from being too small near t=0
    
    Returns:
        Beta values for each timestep
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule (original DDPM).
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta values for each timestep
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    """
    Gaussian Diffusion process for training and sampling.
    Handles noise scheduling and forward/reverse processes.
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = 'cosine',
        device: str = 'cpu'
    ):
        """
        Initialize diffusion process.
        
        Args:
            timesteps: Number of diffusion steps (default: 1000)
            schedule: 'cosine' or 'linear' (default: 'cosine')
            device: Device to place tensors on
        """
        self.timesteps = timesteps
        self.device = device
        
        # Create noise schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute useful values (move everything to device)
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]).to(device)
        
        # Calculations for forward process (adding noise)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        
        # Calculations for reverse process (denoising)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(device)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean image.
        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        
        Args:
            x_start: Clean image (batch, channels, height, width)
            t: Timestep for each sample in batch (batch,)
            noise: Optional pre-sampled noise (same shape as x_start)
        
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get coefficients for this timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Add noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion step (DDPM): remove noise from image.
        
        Args:
            model: Denoising model
            x_t: Noisy image at timestep t
            t: Current timestep
            condition: Conditioning image (unsolved maze)
        
        Returns:
            Slightly less noisy image at timestep t-1
        """
        # Get model prediction
        with torch.no_grad():
            # Concatenate noisy image with conditioning
            model_input = torch.cat([x_t, condition], dim=1)
            predicted_noise = model(model_input, t)
        
        # Get coefficients
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            # Add noise (except at last step)
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        progress: bool = False
    ) -> torch.Tensor:
        """
        Full reverse diffusion loop (DDPM sampling).
        Start from pure noise and iteratively denoise.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate (batch, channels, height, width)
            condition: Conditioning image (unsolved maze)
            progress: Whether to show progress bar
        
        Returns:
            Generated images (solved mazes)
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Iteratively denoise
        timesteps_list = list(reversed(range(self.timesteps)))
        
        if progress:
            from tqdm import tqdm
            timesteps_list = tqdm(timesteps_list, desc='Sampling')
        
        for t_idx in timesteps_list:
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, condition)
        
        return img
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = False
    ) -> torch.Tensor:
        """
        DDIM sampling (faster, deterministic if eta=0).
        Allows sampling with fewer steps than training timesteps.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate
            condition: Conditioning image (unsolved maze)
            ddim_steps: Number of sampling steps (default: 50)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
            progress: Whether to show progress bar
        
        Returns:
            Generated images
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Create subset of timesteps for DDIM
        step_size = self.timesteps // ddim_steps
        timesteps = np.arange(0, self.timesteps, step_size)
        timesteps = timesteps[::-1].copy()  # Reverse order
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc='DDIM Sampling')
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Get model prediction
            model_input = torch.cat([img, condition], dim=1)
            predicted_noise = model(model_input, t)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t_idx]
            
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Clip to valid range
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * predicted_noise
            
            # Add noise if eta > 0
            noise = torch.randn_like(img) if eta > 0 else 0
            
            # Update image
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise
        
        return img


if __name__ == '__main__':
    print("Testing Gaussian Diffusion...\n")
    
    # Test noise schedule
    print("1. Testing noise schedules:")
    cosine_betas = cosine_beta_schedule(1000)
    linear_betas = linear_beta_schedule(1000)
    print(f"  Cosine betas - min: {cosine_betas.min():.6f}, max: {cosine_betas.max():.6f}")
    print(f"  Linear betas - min: {linear_betas.min():.6f}, max: {linear_betas.max():.6f}")
    
    # Test diffusion
    print("\n2. Testing GaussianDiffusion:")
    diffusion = GaussianDiffusion(timesteps=1000, schedule='cosine')
    print(f"  Timesteps: {diffusion.timesteps}")
    print(f"  Alpha_cumprod range: [{diffusion.alphas_cumprod.min():.4f}, {diffusion.alphas_cumprod.max():.4f}]")
    
    # Test forward process (adding noise)
    print("\n3. Testing forward process (q_sample):")
    x_start = torch.randn(4, 1, 32, 32)  # Batch of 4 images
    t = torch.tensor([0, 250, 500, 999])  # Different timesteps
    x_noisy = diffusion.q_sample(x_start, t)
    print(f"  Input shape: {x_start.shape}")
    print(f"  Noisy output shape: {x_noisy.shape}")
    print(f"  Noise level at t=0: {(x_noisy[0] - x_start[0]).abs().mean():.4f}")
    print(f"  Noise level at t=999: {(x_noisy[3] - x_start[3]).abs().mean():.4f}")
    
    print("\n✓ Diffusion components test complete!")

