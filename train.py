"""
Training script for maze diffusion model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

from data.dataset import create_dataloaders
from models.unet import TinyUNet
from models.diffusion import GaussianDiffusion
from models.ema import EMA


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train maze diffusion model')
    
    # Data
    parser.add_argument('--train_path', type=str, default='data/raw/train.npz',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, default='data/raw/test.npz',
                        help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in UNet')
    parser.add_argument('--time_emb_dim', type=int, default=128,
                        help='Dimension of time embeddings')
    
    # Diffusion
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'],
                        help='Noise schedule')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda/mps)')
    
    # Logging
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--sample_every', type=int, default=1000,
                        help='Generate samples every N steps')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory to save sample images')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def save_checkpoint(path: str, model, ema, optimizer, step: int, loss: float, args):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, model, ema, optimizer, device):
    """Load checkpoint and resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f"Resumed from checkpoint: {path} (step {step})")
    return step


def save_samples(samples_dir: Path, step: int, unsolved, predicted, ground_truth):
    """Save sample images for visualization."""
    import matplotlib.pyplot as plt
    
    # Convert to numpy and denormalize from [-1, 1] to [0, 1]
    unsolved = (unsolved.cpu().numpy() + 1) / 2
    predicted = (predicted.cpu().numpy() + 1) / 2
    ground_truth = (ground_truth.cpu().numpy() + 1) / 2
    
    # Show 4 examples
    num_samples = min(4, unsolved.shape[0])
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Unsolved
        axes[i, 0].imshow(unsolved[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Unsolved')
        axes[i, 0].axis('off')
        
        # Predicted
        axes[i, 1].imshow(predicted[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Predicted')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(ground_truth[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = samples_dir / f'step_{step:06d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Samples saved: {save_path}")


def train_one_epoch(
    model, ema, diffusion, train_loader, optimizer, device, 
    epoch, global_step, args, samples_dir, test_batch
):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, (unsolved, solved) in enumerate(pbar):
        # Move to device
        unsolved = unsolved.to(device)
        solved = solved.to(device)
        
        # Sample random timesteps
        batch_size = solved.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(solved)
        
        # Add noise to solved mazes (forward diffusion)
        noisy_solved = diffusion.q_sample(solved, t, noise)
        
        # Concatenate with conditioning (unsolved maze)
        model_input = torch.cat([noisy_solved, unsolved], dim=1)
        
        # Predict noise
        predicted_noise = model(model_input, t)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update EMA
        ema.update(model)
        
        # Logging
        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{epoch_loss / num_batches:.4f}'
        })
        
        # Log periodically
        if global_step % args.log_every == 0:
            avg_loss = epoch_loss / num_batches
            print(f'\n[Step {global_step}] Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}')
        
        # Save checkpoint
        if global_step % args.save_every == 0:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                save_dir / f'checkpoint_step_{global_step}.pt',
                model, ema, optimizer, global_step, loss.item(), args
            )
            # Also save as latest
            save_checkpoint(
                save_dir / 'latest.pt',
                model, ema, optimizer, global_step, loss.item(), args
            )
        
        # Generate samples
        if global_step % args.sample_every == 0:
            print(f'\n[Step {global_step}] Generating samples...')
            generate_samples(
                ema, diffusion, test_batch, device, 
                samples_dir, global_step
            )
            model.train()  # Back to training mode
    
    avg_epoch_loss = epoch_loss / num_batches
    return global_step, avg_epoch_loss


@torch.no_grad()
def generate_samples(ema, diffusion, test_batch, device, samples_dir, step):
    """Generate sample mazes using DDIM."""
    ema.eval()
    
    # Get a batch of test data
    unsolved, solved = test_batch
    unsolved = unsolved[:4].to(device)  # Just 4 samples
    solved = solved[:4].to(device)
    
    # Generate using DDIM (fast sampling)
    shape = solved.shape
    predicted = diffusion.ddim_sample(
        ema.ema_model,
        shape,
        unsolved,
        ddim_steps=50,
        progress=False
    )
    
    # Clamp to valid range
    predicted = torch.clamp(predicted, -1, 1)
    
    # Save samples
    save_samples(samples_dir, step, unsolved, predicted, solved)


@torch.no_grad()
def evaluate(model, diffusion, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for unsolved, solved in tqdm(test_loader, desc='Evaluating'):
        unsolved = unsolved.to(device)
        solved = solved.to(device)
        
        batch_size = solved.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(solved)
        
        noisy_solved = diffusion.q_sample(solved, t, noise)
        model_input = torch.cat([noisy_solved, unsolved], dim=1)
        predicted_noise = model(model_input, t)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"\nMaze Diffusion Training")
    print("=" * 50)
    print(f"Device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    samples_dir = Path(args.sample_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader = create_dataloaders(
        train_path=args.train_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Get a test batch for sampling
    test_batch = next(iter(test_loader))
    
    # Create model
    print("\nCreating model...")
    model = TinyUNet(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device
    )
    print(f"Diffusion timesteps: {args.timesteps}")
    print(f"Noise schedule: {args.schedule}")
    
    # Create EMA
    ema = EMA(model, decay=args.ema_decay, device=device)
    print(f"EMA decay: {args.ema_decay}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 1
    if args.resume:
        global_step = load_checkpoint(args.resume, model, ema, optimizer, device)
        start_epoch = (global_step // len(train_loader)) + 1
    
    print("\n" + "=" * 50)
    print("Starting training...\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        global_step, avg_loss = train_one_epoch(
            model, ema, diffusion, train_loader, optimizer, device,
            epoch, global_step, args, samples_dir, test_batch
        )
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f}min")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss = evaluate(ema.ema_model, diffusion, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}\n")
        print("-" * 50)
    
    # Save final checkpoint
    print("\nTraining complete!")
    save_checkpoint(
        save_dir / 'final.pt',
        model, ema, optimizer, global_step, avg_loss, args
    )
    print(f"Final checkpoint saved to: {save_dir / 'final.pt'}")


if __name__ == '__main__':
    main()

