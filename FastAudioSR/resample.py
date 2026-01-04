import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- HIGHLY OPTIMIZED MATH KERNELS ---

@torch.jit.script
def snake_op(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    # Fused operation: x + (1 - cos(2ax)) * (1/2b)
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

@torch.jit.script
def fast_upsample_kernel(x: Tensor, weight: Tensor, ratio: float, stride: int, padding: int, output_padding: int):
    # Optimized transpose conv with baked-in scaling
    # We use native padding inside the conv to avoid the Python slicing overhead
    x = F.conv_transpose1d(x, weight, stride=stride, padding=padding, output_padding=output_padding, groups=x.shape[1])
    return x * ratio

# --- MODULES ---

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        # Keep these names identical for state_dict compatibility
        self.alpha = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1), persistent=False)
        self._prepared = False

    def prepare(self):
        """Call this once after loading weights to bake parameters for inference."""
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._prepared = True

    def forward(self, x):
        if self.training:
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            return snake_op(x, a, 1.0 / (2.0 * b + 1e-9))
        
        if not self._prepared: self.prepare()
        return snake_op(x, self.a_eff, self.inv_2b)

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = kernel_size
        even = (kernel_size % 2 == 0)
        self.padding = kernel_size // 2 - int(even)
        
        # Original filter weights for state_dict
        from __main__ import kaiser_sinc_filter1d # Assuming helper is in scope
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        
        # Optimization buffer
        self.register_buffer("f_opt", torch.zeros(channels, 1, kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self.training:
            if not self._prepared: self.prepare()
            # Direct access to f_opt without slicing/expanding
            return F.conv1d(x, self.f_opt, stride=self.stride, padding=self.padding, groups=x.shape[1])
        
        f = self.filter.expand(x.shape[1], -1, -1)
        return F.conv1d(x, f, stride=self.stride, padding=self.padding, groups=x.shape[1])

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = float(ratio)
        self.stride = ratio
        self.channels = channels
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        
        # Pre-calculating padding to avoid slicing [..., pad_left:-pad_right]
        # Transpose conv padding math: 
        self.pad = (self.kernel_size - self.stride + 1) // 2
        self.out_pad = 2 * self.pad - (self.kernel_size - self.stride)
        
        from __main__ import kaiser_sinc_filter1d
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self.training:
            if not self._prepared: self.prepare()
            return fast_upsample_kernel(x, self.f_opt, self.ratio, self.stride, self.pad, self.out_pad)
        
        f = self.filter.expand(x.shape[1], -1, -1)
        return fast_upsample_kernel(x, f, self.ratio, self.stride, self.pad, self.out_pad)
        
class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        # Internalize the lowpass filter with the optimized LowPassFilter1d class
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, 
            half_width=0.6 / ratio, 
            stride=ratio, 
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
            channels=channels
        )

    def prepare(self):
        """Recursively prepares the internal lowpass filter"""
        self.lowpass.prepare()

    def forward(self, x):
        # The LowPassFilter1d already handles the optimized f_opt logic
        return self.lowpass(x)
