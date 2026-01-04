import torch
from torch import nn, Tensor

@torch.jit.script
def snake_fused(x: Tensor, alpha: Tensor, log_scale: bool) -> Tensor:
    # Use broadcasting directly to avoid slow reshape/unsqueeze calls
    a = torch.exp(alpha) if log_scale else alpha
    # Use sin(x)^2 = (1 - cos(2x)) / 2 identity for faster computation
    # This reduces transcendental calls which are the main GPU bottleneck
    return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-9)

@torch.jit.script
def snake_beta_fused(x: Tensor, alpha: Tensor, beta: Tensor, log_scale: bool) -> Tensor:
    a = torch.exp(alpha) if log_scale else alpha
    b = torch.exp(beta) if log_scale else beta
    # Identical to: x + (1/b) * sin^2(ax)
    return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)

class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        # Shape (in_features) matches your pretrained weights
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable

    def forward(self, x: Tensor) -> Tensor:
        # unsqueeze(0) and unsqueeze(-1) converts [C] to [1, C, 1] for (B, C, T) input
        return snake_fused(x, self.alpha.unsqueeze(0).unsqueeze(-1), self.alpha_logscale)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: Tensor) -> Tensor:
        return snake_beta_fused(
            x, 
            self.alpha.unsqueeze(0).unsqueeze(-1), 
            self.beta.unsqueeze(0).unsqueeze(-1), 
            self.alpha_logscale
        )
