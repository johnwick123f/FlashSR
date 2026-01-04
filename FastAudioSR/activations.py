import torch
from torch import nn

class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        
        # Keep parameter names and shapes identical for weight loading
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        # Flattening the broadcast: [C] -> [1, C, 1]
        # Using [None, :, None] is a static op that torch.compile likes better than unsqueeze
        a = self.alpha[None, :, None]
        
        if self.alpha_logscale:
            a = torch.exp(a)
        
        # Optimization: sin(x)^2 is faster as s * s than pow(s, 2)
        # We also remove .detach() which was causing graph breaks
        s = torch.sin(x * a)
        return x + (s * s) / (a + self.no_div_by_zero)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        # Map parameters to [1, C, 1]
        a = self.alpha[None, :, None]
        b = self.beta[None, :, None]
        
        if self.alpha_logscale:
            a = torch.exp(a)
            b = torch.exp(b)
        
        # SnakeBeta := x + 1/b * sin^2 (xa)
        s = torch.sin(x * a)
        return x + (s * s) / (b + self.no_div_by_zero)
