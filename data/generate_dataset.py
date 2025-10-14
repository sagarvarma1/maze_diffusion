"""
Generate training and test datasets of maze pairs.
Creates train/test split automatically.
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from maze_generator import generate_maze


def generate_dataset(
    num_samples: int = 100000,
    test_split: float = 0.1,
    output_dir: str = 'data/raw',
    pixel_size: int = 32,
    seed: int = 42,
    save_format: str = 'npz'
):
    """
    Generate maze dataset with train/test split.
    
    Args:
        num_samples: Total number of mazes to generate
        test_split: Fraction of data for test set (default 0.1)
        output_dir: Directory to save generated data
        pixel_size: Size of maze images (default 32x32)
        seed: Random seed for reproducibility
        save_format: 'npz' for numpy arrays or 'png' for images
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    
    if save_format == 'png':
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate splits
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_test
    
    print(f"Generating {num_samples} mazes...")
    print(f"  Train: {num_train}")
    print(f"  Test: {num_test}")
    print(f"  Format: {save_format}")
    print(f"  Seed: {seed}\n")
    
    if save_format == 'npz':
        # Generate all and store in memory (for npz)
        train_unsolved = []
        train_solved = []
        test_unsolved = []
        test_solved = []
        
        for i in tqdm(range(num_samples), desc="Generating mazes"):
            unsolved, solved = generate_maze(pixel_size=pixel_size)
            
            # Split into train/test
            if i < num_train:
                train_unsolved.append(unsolved)
                train_solved.append(solved)
            else:
                test_unsolved.append(unsolved)
                test_solved.append(solved)
        
        # Convert to numpy arrays
        print("\nConverting to numpy arrays...")
        train_unsolved = np.array(train_unsolved, dtype=np.float32)
        train_solved = np.array(train_solved, dtype=np.float32)
        test_unsolved = np.array(test_unsolved, dtype=np.float32)
        test_solved = np.array(test_solved, dtype=np.float32)
        
        # Save
        print("Saving train set...")
        np.savez_compressed(
            output_path / 'train.npz',
            unsolved=train_unsolved,
            solved=train_solved
        )
        
        print("Saving test set...")
        np.savez_compressed(
            output_path / 'test.npz',
            unsolved=test_unsolved,
            solved=test_solved
        )
        
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"  Train shapes: {train_unsolved.shape}, {train_solved.shape}")
        print(f"  Test shapes: {test_unsolved.shape}, {test_solved.shape}")
    
    else:  # PNG format
        from PIL import Image
        
        train_count = 0
        test_count = 0
        
        for i in tqdm(range(num_samples), desc="Generating mazes"):
            unsolved, solved = generate_maze(pixel_size=pixel_size)
            
            # Convert to uint8 for saving
            unsolved_img = (unsolved * 255).astype(np.uint8)
            solved_img = (solved * 255).astype(np.uint8)
            
            # Split into train/test
            if i < num_train:
                Image.fromarray(unsolved_img).save(train_dir / f'{train_count:06d}_unsolved.png')
                Image.fromarray(solved_img).save(train_dir / f'{train_count:06d}_solved.png')
                train_count += 1
            else:
                Image.fromarray(unsolved_img).save(test_dir / f'{test_count:06d}_unsolved.png')
                Image.fromarray(solved_img).save(test_dir / f'{test_count:06d}_solved.png')
                test_count += 1
        
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"  Train: {train_count} pairs in {train_dir}")
        print(f"  Test: {test_count} pairs in {test_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate maze dataset')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Total number of mazes to generate (default: 100000)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction for test set (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory (default: data/raw)')
    parser.add_argument('--pixel_size', type=int, default=32,
                        help='Maze image size (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--format', type=str, choices=['npz', 'png'], default='npz',
                        help='Save format: npz or png (default: npz)')
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples,
        test_split=args.test_split,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        seed=args.seed,
        save_format=args.format
    )


if __name__ == '__main__':
    main()


