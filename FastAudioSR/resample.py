import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Dict

# --- FUSED KERNELS (UNCHANGED) ---

@torch.jit.script
def snake_op(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

@torch.jit.script
def fused_upsample_rescale(x: Tensor, weight: Tensor, stride: int, padding: int, out_padding: int, groups: int, ratio: float) -> Tensor:
    # We remove the padding from the conv_transpose call because we handle it via F.pad
    x = F.conv_transpose1d(x, weight, stride=stride, padding=0, output_padding=out_padding, groups=groups)
    return x * ratio

# --- HELPER (UNCHANGED) ---

def get_kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: beta = 0.1102 * (A - 8.7)
    elif A >= 21.: beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: beta = 0.
    
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size).float() + 0.5) if even else (torch.arange(kernel_size).float() - half_size)
    
    if cutoff == 0: return torch.zeros(1, 1, kernel_size)
    x = 2 * cutoff * time
    sinc = torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    res = 2 * cutoff * window * sinc
    return (res / res.sum()).view(1, 1, kernel_size)

# --- MODULES ---

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.in_features = in_features
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(init_val * alpha, requires_grad=alpha_trainable)
        
        self.register_buffer('a_eff', torch.empty(0), persistent=False)
        self.register_buffer('inv_2b', torch.empty(0), persistent=False)

    def prepare(self):
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff = a.contiguous()
            self.inv_2b = (1.0 / (2.0 * b + 1e-9)).contiguous()

    def forward(self, x):
        if self.training:
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            return snake_op(x, a, 1.0 / (2.0 * b + 1e-9))
        
        if self.a_eff.numel() == 0: self.prepare()
        return snake_op(x, self.a_eff, self.inv_2b)

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        # Logic for reflection padding
        self.pad = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.register_buffer("filter", get_kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        self.f_cache: Dict[int, Tensor] = {}

    def forward(self, x):
        C = x.shape[1]
        # Use reflection padding to avoid edge artifacts in audio
        x = F.pad(x, (self.pad, self.pad), mode='reflect')
        
        if self.training:
            return F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, padding=0, groups=C)
        
        if C not in self.f_cache or self.f_cache[C].device != x.device:
            self.f_cache[C] = self.filter.expand(C, -1, -1).contiguous()
            
        return F.conv1d(x, self.f_cache[C], stride=self.stride, padding=0, groups=C)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = float(ratio)
        self.stride = int(ratio)
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        
        # Calculate padding for internal logic
        self.pad = (self.kernel_size - self.stride + 1) // 2
        self.out_pad = 2 * self.pad - (self.kernel_size - self.stride)
        
        self.register_buffer("filter", get_kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))
        self.f_cache: Dict[int, Tensor] = {}

    def forward(self, x):
        C = x.shape[1]
        
        # Reflect pad the input slightly to handle edge transients during upsampling
        # We use a small pad (1) because Transpose Conv expands the signal anyway
        x = F.pad(x, (1, 1), mode='reflect')
        
        if self.training:
            f = self.filter.expand(C, -1, -1)
            # Adjust padding to account for the manual F.pad(1, 1)
            return fused_upsample_rescale(x, f, self.stride, 0, self.out_pad, C, self.ratio)[:, :, self.stride:-self.stride]
        
        if C not in self.f_cache or self.f_cache[C].device != x.device:
            self.f_cache[C] = self.filter.expand(C, -1, -1).contiguous()
            
        # Slicing at the end is fast and ensures the output length matches your original state_dict expectations
        return fused_upsample_rescale(x, self.f_cache[C], self.stride, 0, self.out_pad, C, self.ratio)[:, :, self.stride:-self.stride]

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(0.5/ratio, 0.6/ratio, stride=ratio, kernel_size=ks, channels=channels)

    def forward(self, x):
        return self.lowpass(x)
