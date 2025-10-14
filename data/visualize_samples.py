"""
Visualize sample mazes from the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def visualize_samples(data_path: str = 'data/raw/train.npz', num_samples: int = 9):
    """
    Display a grid of maze samples.
    
    Args:
        data_path: Path to npz file
        num_samples: Number of samples to show (default 9)
    """
    # Load data
    data = np.load(data_path)
    unsolved = data['unsolved']
    solved = data['solved']
    
    print(f"Loaded {len(unsolved)} maze pairs")
    print(f"Shape: {unsolved.shape}")
    
    # Determine grid size
    n = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(n, n * 2, figsize=(n * 4, n * 2))
    fig.suptitle('Maze Samples: Unsolved (left) | Solved (right)', fontsize=14)
    
    # Flatten axes for easy indexing
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        if i >= len(unsolved):
            break
        
        row = i // n
        col = (i % n) * 2
        
        # Unsolved
        axes[row, col].imshow(unsolved[i], cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'Maze {i} - Unsolved')
        axes[row, col].axis('off')
        
        # Solved
        axes[row, col + 1].imshow(solved[i], cmap='gray', vmin=0, vmax=1)
        axes[row, col + 1].set_title(f'Maze {i} - Solved')
        axes[row, col + 1].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, n * n):
        row = i // n
        col = (i % n) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('maze_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to maze_samples.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize maze samples')
    parser.add_argument('--data_path', type=str, default='data/raw/train.npz',
                        help='Path to data file')
    parser.add_argument('--num_samples', type=int, default=9,
                        help='Number of samples to display')
    
    args = parser.parse_args()
    visualize_samples(args.data_path, args.num_samples)


if __name__ == '__main__':
    main()

