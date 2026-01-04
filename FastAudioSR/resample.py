import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

@torch.jit.script
def snake_op(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    # Fused CUDA kernel for x + (1 - cos(2ax)) * (1/2b)
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

def get_kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int):
    """Faster initialization of the sinc filter."""
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    
    # Pre-calculate beta using the Kaiser formula
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: beta = 0.1102 * (A - 8.7)
    elif A >= 21.: beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: beta = 0.
    
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    
    # Vectorized time calculation
    if even:
        time = torch.arange(-half_size, half_size).float() + 0.5
    else:
        time = torch.arange(kernel_size).float() - half_size
        
    if cutoff == 0:
        return torch.zeros(1, 1, kernel_size)
    
    # Sinc function: sin(pi*x)/(pi*x)
    x = 2 * cutoff * time
    sinc = torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    
    res = 2 * cutoff * window * sinc
    return (res / res.sum()).view(1, 1, kernel_size)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        
        # Static buffers for inference (not saved in state_dict)
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
        self.padding = kernel_size // 2 - int(kernel_size % 2 == 0)
        
        self.register_buffer("filter", get_kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self.training:
            if not self._prepared: self.prepare()
            return F.conv1d(x, self.f_opt, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return F.conv1d(x, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, padding=self.padding, groups=x.shape[1])

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = float(ratio)
        self.stride = ratio
        self.channels = channels
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        
        # Baked padding logic to avoid Python-side slicing
        self.pad = (self.kernel_size - self.stride + 1) // 2
        self.out_pad = 2 * self.pad - (self.kernel_size - self.stride)
        
        self.register_buffer("filter", get_kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        # Using a JIT scripted kernel for the transpose conv + scale fusion
        if not self.training:
            if not self._prepared: self.prepare()
            return self._fast_call(x, self.f_opt)
        return self._fast_call(x, self.filter.expand(x.shape[1], -1, -1))

    def _fast_call(self, x, weight):
        return F.conv_transpose1d(x, weight, stride=int(self.stride), padding=self.pad, 
                                  output_padding=self.out_pad, groups=x.shape[1]) * self.ratio

class DownSample1d(nn.Module):
    """Directly optimized DownSample using the fast LowPass engine."""
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(0.5/ratio, 0.6/ratio, stride=ratio, kernel_size=ks, channels=channels)

    def prepare(self):
        self.lowpass.prepare()

    def forward(self, x):
        return self.lowpass(x)
