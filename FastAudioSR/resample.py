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

# --- ACTIVATION MODULES ---

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride # Strictly 1 as requested
        self.channels = channels
        
        # We use a 6-tap window for speed, but keep 12 in the buffer for checkpoint compatibility
        self.target_k = 6
        self.padding = 2 # (6 // 2 - 1) for zero-padding symmetry
        
        # This matches your checkpoint [1, 1, 12]
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        # Optimized buffer [C, 1, 6]
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Slice middle 6 taps from the loaded 12-tap weights
            # [1, 1, 12] -> [1, 1, 6]
            short_f = self.filter[:, :, 3:9]
            # Pre-expand to match channel groups
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        # Using groups=C is the fastest way to apply the same filter to every channel
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=self.padding, groups=C)

class UpSample1d(nn.Module):
    """Modified to apply Kaiser smoothing at Ratio 1"""
    def __init__(self, ratio=1, kernel_size=12, channels=512):
        super().__init__()
        # We ignore the 'ratio' argument and force it to 1 for speed
        self.stride = 1 
        self.channels = channels
        self.target_k = 6
        
        # Load the weights the checkpoint expects (12 taps)
        # Even if ratio=1, we use the 0.5/ratio math to match the original filter's "shape"
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5/2.0, 0.6/2.0, kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            short_f = self.filter[:, :, 3:9]
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        # Standard convolution instead of transpose because ratio is 1
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=2, groups=C)

class DownSample1d(nn.Module):
    """Modified to apply Low-pass cleanup at Ratio 1"""
    def __init__(self, ratio=1, kernel_size=12, channels=512):
        super().__init__()
        # Force ratio 1 and reuse the LowPass logic
        self.lowpass = LowPassFilter1d(
            cutoff=0.5/2.0, 
            half_width=0.6/2.0, 
            stride=1, 
            kernel_size=kernel_size, 
            channels=channels
        )

    def forward(self, x):
        return self.lowpass(x)
