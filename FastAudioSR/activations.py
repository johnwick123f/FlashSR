import torch
from torch import nn, Tensor

@torch.jit.script
def snake_fast_inference(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    # x + (1 - cos(2ax)) * (1/2b)
    # Multiplication is faster than division on GPU
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        
        # Pre-calculated buffers for inference
        self.register_buffer('a_eff', torch.ones(1, in_features, 1))
        self.register_buffer('inv_2a', torch.ones(1, in_features, 1))
        self._is_prepared = False

    def prepare(self):
        """Bakes all constants into buffers for zero-overhead inference"""
        with torch.no_grad():
            a = torch.exp(self.alpha) if self.alpha_logscale else self.alpha
            a = a.view(1, -1, 1)
            self.a_eff.copy_(a)
            # Pre-calculate 1 / (2a + eps)
            self.inv_2a.copy_(1.0 / (2.0 * a + 1e-9))
        self._is_prepared = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()
        
        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2a)
        
        # Training fallback (slower but differentiable)
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-9)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.register_buffer('a_eff', torch.ones(1, in_features, 1))
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1))
        self._is_prepared = False

    def prepare(self):
        with torch.no_grad():
            a = torch.exp(self.alpha) if self.alpha_logscale else self.alpha
            b = torch.exp(self.beta) if self.alpha_logscale else self.beta
            self.a_eff.copy_(a.view(1, -1, 1))
            # Pre-calculate 1 / (2b + eps)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._is_prepared = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()

        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2b)
        
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)
