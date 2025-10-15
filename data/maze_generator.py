"""
Simple maze generation and solving for diffusion model training.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List


def generate_simple_maze(size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple maze with solution using iterative backtracking.
    
    Args:
        size: Maze size (default 32x32 pixels)
    
    Returns:
        (unsolved_maze, solved_maze) as numpy arrays
    """
    # Create maze grid - odd dimensions for proper walls
    grid_size = (size // 2) - 1
    if grid_size < 3:
        grid_size = 3
    
    # Initialize maze: 1 = path, 0 = wall
    # Make it larger to have room for walls between cells
    maze_grid = np.zeros((grid_size * 2 + 1, grid_size * 2 + 1), dtype=np.float32)
    
    # Iterative backtracking to carve maze (avoids recursion limit)
    stack = [(1, 1)]
    maze_grid[1, 1] = 1.0  # Mark start as path
    
    while stack:
        x, y = stack[-1]
        
        # Random directions
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(directions)
        
        found = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < maze_grid.shape[1] and 
                0 <= ny < maze_grid.shape[0] and 
                maze_grid[ny, nx] == 0):
                
                # Carve path between cells
                maze_grid[y + dy // 2, x + dx // 2] = 1.0
                maze_grid[ny, nx] = 1.0
                
                # Add new cell to stack
                stack.append((nx, ny))
                found = True
                break
        
        if not found:
            # Backtrack
            stack.pop()
    
    # Add entrance and exit
    maze_grid[1, 0] = 1.0  # Left entrance
    maze_grid[-2, -1] = 1.0  # Right exit
    
    # Resize to target size
    from scipy.ndimage import zoom
    zoom_factor = size / maze_grid.shape[0]
    unsolved = zoom(maze_grid, zoom_factor, order=0)  # Nearest neighbor
    unsolved = unsolved[:size, :size]  # Ensure exact size
    
    # Find path using BFS
    start = (0, 1)
    end = (size - 1, size - 2)
    
    # Make sure start and end are on paths
    unsolved[start[1], start[0]] = 1.0
    unsolved[end[1], end[0]] = 1.0
    
    path = bfs_solve(unsolved, start, end)
    
    # Create solved version
    solved = unsolved.copy()
    for x, y in path:
        if 0 <= y < size and 0 <= x < size:
            solved[y, x] = 0.5  # Gray path
    
    return unsolved, solved


def bfs_solve(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Solve maze using BFS.
    
    Args:
        maze: Binary maze (0=wall, 1=path)
        start: Start position (x, y)
        end: End position (x, y)
    
    Returns:
        List of (x, y) coordinates forming path
    """
    height, width = maze.shape
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        x, y = current
        # Check 4 neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < width and 0 <= ny < height and
                (nx, ny) not in visited and
                maze[ny, nx] > 0.5):  # Is path
                
                visited.add((nx, ny))
                parent[(nx, ny)] = current
                queue.append((nx, ny))
    
    return []  # No path found


def generate_maze(pixel_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main entry point - generates a maze pair.
    
    Args:
        pixel_size: Size of output images (default 32)
    
    Returns:
        (unsolved_maze, solved_maze) as numpy arrays
    """
    return generate_simple_maze(pixel_size)


if __name__ == '__main__':
    # Test generation
    print("Testing simple maze generation...")
    
    for i in range(5):
        print(f"\nMaze {i+1}:")
        unsolved, solved = generate_maze(32)
        
        print(f"  Unsolved shape: {unsolved.shape}")
        print(f"  Solved shape: {solved.shape}")
        print(f"  Unsolved unique values: {np.unique(unsolved)}")
        print(f"  Solved unique values: {np.unique(solved)}")
        print(f"  Path pixels: {np.sum(solved == 0.5)}")
        
        # Check if solvable
        if np.sum(solved == 0.5) > 0:
            print("  ✓ Has solution path")
        else:
            print("  ✗ No solution found")
    
    print("\n✓ Test complete!")
