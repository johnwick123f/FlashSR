import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- MATH UTILS ---
@torch.jit.script
def sinc(x: Tensor):
    return torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: beta = 0.1102 * (A - 8.7)
    elif A >= 21.: beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: beta = 0.
    
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

# --- FUSED KERNELS ---
@torch.jit.script
def snake_fast(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

@torch.jit.script
def fast_upsample_forward(x: Tensor, weight: Tensor, ratio: int, stride: int, pad_inner: int, pad_left: int, pad_right: int):
    # Using 'zeros' (implicit padding) is way faster than 'replicate'
    x = F.conv_transpose1d(x, weight, stride=stride, padding=pad_inner, groups=x.shape[1])
    return x[..., pad_left:-pad_right] * float(ratio)

# --- MODULES ---

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        if not self.training: return snake_fast(x, self.a_eff, self.inv_2b)
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        even = (kernel_size % 2 == 0)
        self.padding = kernel_size // 2 - int(even) # Using implicit zero padding
        
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        f = self.f_opt[:C] if not self.training else self.filter.expand(C, -1, -1)
        # padding=self.padding here is 'zero' padding, handled by cuDNN (fast)
        return F.conv1d(x, f, stride=self.stride, padding=self.padding, groups=C)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio, self.stride, self.channels = ratio, ratio, channels
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        
        # Math for zero-padded transpose conv
        self.pad_inner = 0 
        self.pad_left = (self.kernel_size - self.stride) // 2
        self.pad_right = (self.kernel_size - self.stride + 1) // 2
        
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        f = self.f_opt[:C] if not self.training else self.filter.expand(C, -1, -1)
        return fast_upsample_forward(x, f, self.ratio, self.stride, self.pad_inner, self.pad_left, self.pad_right)

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
