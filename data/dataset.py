"""
PyTorch Dataset for loading maze pairs from NPZ files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MazeDataset(Dataset):
    """
    Dataset for maze diffusion training.
    Loads unsolved and solved maze pairs from NPZ files.
    """
    
    def __init__(self, npz_path: str, normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            npz_path: Path to NPZ file containing 'unsolved' and 'solved' arrays
            normalize: If True, normalize to [-1, 1] range (default: True)
        """
        self.npz_path = Path(npz_path)
        self.normalize = normalize
        
        # Load data
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        self.unsolved = data['unsolved']  # (N, 32, 32)
        self.solved = data['solved']      # (N, 32, 32)
        
        print(f"  Loaded {len(self.unsolved)} maze pairs")
        print(f"  Unsolved shape: {self.unsolved.shape}")
        print(f"  Solved shape: {self.solved.shape}")
        print(f"  Unsolved range: [{self.unsolved.min():.2f}, {self.unsolved.max():.2f}]")
        print(f"  Solved range: [{self.solved.min():.2f}, {self.solved.max():.2f}]")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.unsolved)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single maze pair.
        
        Args:
            idx: Index of sample
        
        Returns:
            (unsolved, solved) as torch tensors with shape (1, 32, 32)
        """
        # Get maze pair
        unsolved = self.unsolved[idx].astype(np.float32)
        solved = self.solved[idx].astype(np.float32)
        
        # Normalize to [-1, 1] if requested (better for diffusion)
        if self.normalize:
            unsolved = unsolved * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            solved = solved * 2.0 - 1.0
        
        # Convert to torch tensors
        unsolved = torch.from_numpy(unsolved).unsqueeze(0)  # (1, 32, 32)
        solved = torch.from_numpy(solved).unsqueeze(0)      # (1, 32, 32)
        
        return unsolved, solved


def create_dataloaders(
    train_path: str = 'data/raw/train.npz',
    test_path: str = 'data/raw/test.npz',
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = True
) -> tuple:
    """
    Create train and test dataloaders.
    
    Args:
        train_path: Path to training NPZ file
        test_path: Path to test NPZ file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize data to [-1, 1]
    
    Returns:
        (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MazeDataset(train_path, normalize=normalize)
    test_dataset = MazeDataset(test_path, normalize=normalize)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing MazeDataset...\n")
    
    # Create dataset
    dataset = MazeDataset('data/raw/train.npz', normalize=True)
    
    # Get a single sample
    print("\nTesting single sample:")
    unsolved, solved = dataset[0]
    print(f"  Unsolved shape: {unsolved.shape}")
    print(f"  Solved shape: {solved.shape}")
    print(f"  Unsolved range: [{unsolved.min():.2f}, {unsolved.max():.2f}]")
    print(f"  Solved range: [{solved.min():.2f}, {solved.max():.2f}]")
    print(f"  Unsolved dtype: {unsolved.dtype}")
    print(f"  Solved dtype: {solved.dtype}")
    
    # Test dataloader
    print("\nTesting DataLoader:")
    train_loader, test_loader = create_dataloaders(
        batch_size=32,
        num_workers=0
    )
    
    # Get a batch
    unsolved_batch, solved_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Unsolved batch shape: {unsolved_batch.shape}")
    print(f"  Solved batch shape: {solved_batch.shape}")
    print(f"  Batch range: [{unsolved_batch.min():.2f}, {unsolved_batch.max():.2f}]")
    
    # Test iteration
    print("\nTesting iteration over 3 batches:")
    for i, (unsolved, solved) in enumerate(train_loader):
        if i >= 3:
            break
        print(f"  Batch {i}: unsolved {unsolved.shape}, solved {solved.shape}")
    
    print("\n✓ Dataset loader test complete!")

