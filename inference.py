"""
Inference script for solving mazes with trained diffusion model.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from models.unet import TinyUNet
from models.diffusion import GaussianDiffusion
from data.maze_generator import generate_maze


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        (model, diffusion, checkpoint_info)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model args from checkpoint
    args = checkpoint.get('args', {})
    base_channels = args.get('base_channels', 64)
    time_emb_dim = args.get('time_emb_dim', 128)
    timesteps = args.get('timesteps', 1000)
    schedule = args.get('schedule', 'cosine')
    
    # Create model
    model = TinyUNet(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim
    ).to(device)
    
    # Load EMA weights (better quality than regular weights)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    
    model.eval()
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule,
        device=device
    )
    
    step = checkpoint.get('step', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"Checkpoint info: step={step}, loss={loss}")
    
    return model, diffusion, checkpoint


def load_maze_from_test_set(test_path: str, index: int):
    """
    Load a maze from the test set.
    
    Args:
        test_path: Path to test NPZ file
        index: Index of maze to load
    
    Returns:
        (unsolved, solved) as numpy arrays
    """
    data = np.load(test_path)
    unsolved = data['unsolved'][index]
    solved = data['solved'][index]
    return unsolved, solved


def load_maze_from_image(image_path: str):
    """
    Load and preprocess a maze from an image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        unsolved maze as numpy array (32, 32)
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 32x32
    img = img.resize((32, 32), Image.NEAREST)
    
    # Convert to numpy array
    maze = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    
    # Binarize (simple threshold)
    maze = (maze > 0.5).astype(np.float32)
    
    return maze


def generate_random_maze():
    """
    Generate a random maze.
    
    Returns:
        (unsolved, solved) as numpy arrays
    """
    unsolved, solved = generate_maze(pixel_size=32)
    return unsolved, solved


@torch.no_grad()
def solve_maze(
    model,
    diffusion,
    unsolved_maze,
    device,
    steps=50,
    eta=0.0,
    seed=None
):
    """
    Solve a maze using the trained diffusion model.
    
    Args:
        model: Trained UNet model
        diffusion: GaussianDiffusion instance
        unsolved_maze: Unsolved maze (H, W) or (1, H, W) numpy array [0, 1]
        device: Device to run on
        steps: Number of DDIM sampling steps
        eta: DDIM eta parameter (0=deterministic, 1=stochastic)
        seed: Random seed for reproducibility
    
    Returns:
        Solved maze as numpy array (H, W) [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Prepare input
    if unsolved_maze.ndim == 2:
        unsolved_maze = unsolved_maze[None, :]  # Add channel dim
    
    # Normalize to [-1, 1]
    unsolved_normalized = unsolved_maze * 2.0 - 1.0
    
    # Convert to tensor and add batch dim
    unsolved_tensor = torch.from_numpy(unsolved_normalized).float().unsqueeze(0).to(device)
    
    # Sample using DDIM
    print(f"Solving maze with {steps} DDIM steps...")
    shape = (1, 1, 32, 32)
    solved_tensor = diffusion.ddim_sample(
        model,
        shape,
        unsolved_tensor,
        ddim_steps=steps,
        eta=eta,
        progress=True
    )
    
    # Denormalize from [-1, 1] to [0, 1]
    solved = (solved_tensor.cpu().numpy()[0, 0] + 1.0) / 2.0
    
    # Clamp to valid range
    solved = np.clip(solved, 0, 1)
    
    return solved


def save_comparison(
    unsolved,
    predicted,
    ground_truth,
    output_path,
    title=None
):
    """
    Save side-by-side comparison of input, output, and ground truth.
    
    Args:
        unsolved: Input maze (H, W)
        predicted: Predicted solution (H, W)
        ground_truth: Ground truth solution (H, W) or None
        output_path: Path to save image
        title: Optional title for the figure
    """
    # Determine number of columns
    num_cols = 3 if ground_truth is not None else 2
    
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))
    
    if num_cols == 2:
        axes = [axes[0], axes[1]]
    
    # Input
    axes[0].imshow(unsolved, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input (Unsolved)', fontsize=14)
    axes[0].axis('off')
    
    # Predicted
    axes[1].imshow(predicted, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predicted (Solved)', fontsize=14)
    axes[1].axis('off')
    
    # Ground truth (if available)
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth', fontsize=14)
        axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {output_path}")


def save_grid(samples, output_path, title='Maze Solutions'):
    """
    Save a grid of multiple maze solutions.
    
    Args:
        samples: List of (unsolved, predicted, ground_truth) tuples
        output_path: Path to save grid
        title: Title for the figure
    """
    num_samples = len(samples)
    num_cols = 3
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (unsolved, predicted, ground_truth) in enumerate(samples):
        # Input
        axes[i, 0].imshow(unsolved, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title('Input', fontsize=12)
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(f'Sample {i}', fontsize=12)
        
        # Predicted
        axes[i, 1].imshow(predicted, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 1].set_title('Predicted', fontsize=12)
        axes[i, 1].axis('off')
        
        # Ground truth
        if ground_truth is not None:
            axes[i, 2].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 2].set_title('Ground Truth', fontsize=12)
            axes[i, 2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve mazes with trained diffusion model')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--index', type=int, default=None,
                             help='Index of maze from test set')
    input_group.add_argument('--image', type=str, default=None,
                             help='Path to custom maze image')
    input_group.add_argument('--random', action='store_true',
                             help='Generate random maze')
    
    # Batch processing
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of mazes to solve from test set')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index for batch processing')
    
    # Sampling
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0=deterministic)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Paths
    parser.add_argument('--test_path', type=str, default='data/raw/test.npz',
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda/mps)')
    
    # Visualization
    parser.add_argument('--show', action='store_true',
                        help='Display results in matplotlib window')
    
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


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMaze Diffusion Inference")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"DDIM steps: {args.steps}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load model
    model, diffusion, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Determine input source
    if args.image:
        # Single custom image
        print(f"\nLoading maze from image: {args.image}")
        unsolved = load_maze_from_image(args.image)
        ground_truth = None
        num_samples = 1
        
    elif args.random:
        # Random maze(s)
        num_samples = args.num_samples
        print(f"\nGenerating {num_samples} random maze(s)")
        
    else:
        # From test set (default)
        num_samples = args.num_samples
        start_idx = args.start_index if args.index is None else args.index
        print(f"\nLoading {num_samples} maze(s) from test set (starting at index {start_idx})")
    
    # Process maze(s)
    samples = []
    
    for i in range(num_samples):
        print(f"\n--- Processing maze {i+1}/{num_samples} ---")
        
        # Get input maze
        if args.image:
            # Already loaded above
            pass
        elif args.random:
            print("Generating random maze...")
            unsolved, ground_truth = generate_random_maze()
        else:
            idx = start_idx + i
            print(f"Loading maze {idx} from test set...")
            unsolved, ground_truth = load_maze_from_test_set(args.test_path, idx)
        
        # Solve maze
        predicted = solve_maze(
            model, diffusion, unsolved, device,
            steps=args.steps, eta=args.eta, seed=args.seed
        )
        
        # Save individual result
        if num_samples == 1:
            output_path = output_dir / 'comparison.png'
        else:
            output_path = output_dir / f'maze_{i:03d}_comparison.png'
        
        save_comparison(unsolved, predicted, ground_truth, output_path,
                       title=f'Maze Solution {i}' if num_samples > 1 else None)
        
        # Also save individual images
        if num_samples == 1:
            Image.fromarray((unsolved * 255).astype(np.uint8)).save(
                output_dir / 'input.png'
            )
            Image.fromarray((predicted * 255).astype(np.uint8)).save(
                output_dir / 'output.png'
            )
            if ground_truth is not None:
                Image.fromarray((ground_truth * 255).astype(np.uint8)).save(
                    output_dir / 'ground_truth.png'
                )
        
        samples.append((unsolved, predicted, ground_truth))
    
    # Save grid if multiple samples
    if num_samples > 1:
        grid_path = output_dir / 'grid.png'
        save_grid(samples, grid_path, title=f'{num_samples} Maze Solutions')
    
    print(f"\n{'='*50}")
    print(f"✓ Solved {num_samples} maze(s)")
    print(f"Results saved to: {output_dir}")
    
    # Show results if requested
    if args.show:
        print("\nDisplaying results...")
        if num_samples == 1:
            img = Image.open(output_dir / 'comparison.png')
            plt.figure(figsize=(12, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            img = Image.open(output_dir / 'grid.png')
            plt.figure(figsize=(12, 4 * num_samples))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()

