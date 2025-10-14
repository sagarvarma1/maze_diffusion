"""
Exponential Moving Average (EMA) for model parameters.
Improves sampling quality by maintaining smoothed weights.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a moving average of model weights for better sampling.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = None):
        """
        Initialize EMA.
        
        Args:
            model: The model to track
            decay: EMA decay rate (default: 0.9999)
            device: Device to store EMA parameters (default: same as model)
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create a copy of the model for EMA
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        ema_param = decay * ema_param + (1 - decay) * model_param
        
        Args:
            model: Current model with updated parameters
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)
    
    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)
    
    def eval(self):
        """Set EMA model to eval mode."""
        self.ema_model.eval()
    
    def parameters(self):
        """Get EMA model parameters."""
        return self.ema_model.parameters()


if __name__ == '__main__':
    print("Testing EMA...\n")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create EMA
    ema = EMA(model, decay=0.999)
    
    print("1. Initial state:")
    print(f"  Model param mean: {model[0].weight.data.mean():.6f}")
    print(f"  EMA param mean: {list(ema.ema_model.parameters())[0].data.mean():.6f}")
    print(f"  Difference: {(model[0].weight.data - list(ema.ema_model.parameters())[0].data).abs().mean():.6f}")
    
    # Simulate training updates
    print("\n2. After 10 training steps:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(10):
        # Forward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update(model)
    
    print(f"  Model param mean: {model[0].weight.data.mean():.6f}")
    print(f"  EMA param mean: {list(ema.ema_model.parameters())[0].data.mean():.6f}")
    print(f"  Difference: {(model[0].weight.data - list(ema.ema_model.parameters())[0].data).abs().mean():.6f}")
    
    print("\n✓ EMA test complete!")

