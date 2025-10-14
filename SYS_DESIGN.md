# Maze Diffusion Model - System Design Document

## Overview
A conditional diffusion model that solves mazes. Given an unsolved 32×32 maze image, the model outputs a solved maze image with the solution path overlaid.

---

## Problem Statement
- **Input**: Unsolved maze image (32×32, grayscale or binary)
- **Output**: Solved maze image (32×32) with the solution path drawn
- **Approach**: Conditional image-to-image diffusion model

---

## How It Works

### Core Idea
Diffusion models learn to remove noise from images. We train the model to denoise **solved maze images** while being conditioned on the **unsolved maze structure**. At inference, we start from pure noise and the model generates a solved maze by iteratively denoising.

### Training Flow
1. Generate pairs of `(unsolved_maze, solved_maze)` images
2. For each training iteration:
   - Take a `solved_maze` image (ground truth)
   - Add random noise at timestep `t` → `noisy_solved_maze`
   - Concatenate `unsolved_maze` + `noisy_solved_maze` as input
   - Model predicts the noise that was added
   - Compute loss: MSE between predicted noise and actual noise
3. Repeat for thousands of samples and timesteps

**Key insight**: The model learns "what solved mazes look like" conditioned on the maze structure.

### Inference Flow (Solving New Mazes)
1. Input: `unsolved_maze` (new maze to solve)
2. Initialize: `x = random_noise` (pure Gaussian noise, 32×32)
3. For each denoising step (20–50 iterations):
   - Concatenate `unsolved_maze` + `x` (current noisy state)
   - Model predicts the noise in `x`
   - Remove predicted noise → get slightly less noisy `x`
4. After all steps: `x` becomes the `solved_maze`
5. Post-process and visualize

---

## Data Generation

### Maze Generation
- **Algorithm**: DFS backtracking (simple, generates perfect mazes)
- **Size**: 32×32 pixels
- **Structure**: Binary image (walls = 0/black, paths = 1/white)
- **Start/End**: Two border openings (e.g., top-left and bottom-right)
- **Count**: 100k–300k training samples

### Solution Generation
- Use BFS (Breadth-First Search) to find shortest path from start to end
- Draw solution path on the maze → `solved_maze`
- Solution path in different intensity/color (e.g., gray line on white path, or colored overlay)

### Data Format
Each sample contains:
```
unsolved_maze: [32, 32, 1]  # Binary maze structure
solved_maze:   [32, 32, 1]  # Maze + solution path overlaid
```

### Augmentations (Optional, Light)
- Small Gaussian noise (σ=0.01–0.05)
- Slight blur (kernel size 3)
- Random wall thickness variation (1–2 pixels)

**Keep augmentations minimal** – diffusion models are already robust to noise.

---

## Model Architecture

### Network: Conditional UNet
- **Input**: 2 channels (noisy solved maze + unsolved maze concatenated)
- **Output**: 1 channel (predicted noise)
- **Architecture**:
  - Tiny UNet for 32×32 resolution
  - Base channels: 32–64
  - Depth: 3 levels (down/bottleneck/up)
  - No self-attention (too small resolution)
  - Residual blocks with GroupNorm
  - Timestep embedding via sinusoidal positional encoding

### Diffusion Setup
- **Schedule**: Cosine noise schedule (βₜ)
- **Timesteps**: T = 1000 during training
- **Prediction target**: Epsilon (noise) prediction
- **Objective**: MSE loss on predicted noise

```python
# Pseudocode
noise = sample_gaussian_noise()
t = random_timestep()
noisy_solved = add_noise(solved_maze, noise, t)
input = concat(unsolved_maze, noisy_solved)
pred_noise = model(input, t)
loss = MSE(pred_noise, noise)
```

---

## Training

### Hyperparameters
- **Batch size**: 128
- **Learning rate**: 2e-4 (AdamW)
- **Training steps**: 200k–400k
- **EMA**: Yes (decay 0.9999)
- **Mixed precision**: fp16/bf16
- **Hardware**: Single GPU (V100/A100) or multi-GPU

### Loss Function
- Simple MSE on noise prediction
- Optional: Add small weight to L1 loss for sharper paths

### Metrics (Validation)
- **Path IoU**: Intersection-over-Union of predicted path vs. ground truth
- **Success Rate**: % of mazes where start→end is connected
- **Visual inspection**: Log sample images every 5k steps

### Logging
- Track loss curves, IoU, success rate
- Save checkpoints every 10k steps
- Use Weights & Biases or TensorBoard

---

## Inference

### Sampling Algorithm: DDIM
- **Steps**: 20–50 (fewer steps = faster, slight quality trade-off)
- **Deterministic**: DDIM is deterministic given the same seed
- **Speed**: ~0.1–0.5 seconds on GPU for 32×32

### Inference Pseudocode
```python
def solve_maze(unsolved_maze, model, steps=50):
    x = random_noise(32, 32)  # Start from pure noise
    
    for t in reversed(timesteps):
        input = concat(unsolved_maze, x)
        pred_noise = model(input, t)
        x = ddim_step(x, pred_noise, t)  # Remove noise
    
    solved_maze = x
    return solved_maze
```

### Post-processing
1. **Threshold**: Convert continuous output to binary (threshold ≈ 0.5)
2. **Skeletonize** (optional): Thin path to 1-pixel width for clean visualization
3. **Connectivity check**: Run BFS to verify start→end connection
4. **Gap filling** (fallback): If small gaps exist, apply morphological closing or thin bridge

### Output Formats
- **Path mask**: Binary image of just the solution path
- **Overlay**: Original maze + colored solution path
- **Visualization**: Side-by-side (input | output)

### Why 32×32?
- **Fast iteration**: Small images = fast training and inference
- **Proof of concept**: Validate approach before scaling to 64×64, 128×128
- **Resource efficient**: Can train on modest hardware

### Why concatenate conditioning instead of other methods?
- **Simplicity**: No need for cross-attention, FiLM, or ControlNet
- **Reliability**: Model always sees maze structure, can't generate invalid solutions
- **Speed**: Single forward pass per denoising step

### Why diffusion instead of direct UNet regression?
- **Robustness**: Diffusion handles noisy/imperfect inputs better
- **Multi-modal solutions**: Can sample multiple valid paths if they exist
- **Quality**: Iterative refinement produces cleaner outputs
- **Research interest**: Learn diffusion model mechanics

### Why DDIM for sampling?
- **Speed**: Fewer steps needed (20–50 vs. 1000 for DDPM)
- **Deterministic**: Same maze always produces same solution (given seed)
- **Quality**: Minimal degradation compared to full DDPM

