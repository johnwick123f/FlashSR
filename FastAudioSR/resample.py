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
def fast_upsample_forward(x: Tensor, weight: Tensor, stride: int, pad_left: int, pad_right: int):
    # padding=0 is the fastest path for conv_transpose1d
    x = F.conv_transpose1d(x, weight, stride=stride, padding=0, groups=x.shape[1])
    return x[..., pad_left:-pad_right]

# --- MODULES ---

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.target_k = 6
        
        # Checkpoint expects [1, 1, 12]
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        # Optimized buffer [C, 1, 6]
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Slice middle 6 taps
            short_f = self.filter[:, :, 3:9]
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        # padding=2 on a 6-tap kernel keeps the timing aligned (center-aligned)
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=2, groups=C)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.channels = channels
        self.target_k = 6 # Using 6-tap for speed
        
        # For a 6-tap kernel with stride 2, these slices keep the signal centered
        self.pad_left = (self.target_k - self.stride) // 2
        self.pad_right = (self.target_k - self.stride + 1) // 2
        
        # Load 12-tap filter from checkpoint
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Slice center 6 taps and FOLD the ratio (2.0) into the weights
            start = (self.filter.shape[-1] - self.target_k) // 2
            short_f = self.filter[:, :, start:start+self.target_k] * float(self.ratio)
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        # No more '* self.ratio' at the end - it's baked into f_opt
        return fast_upsample_forward(x, self.f_opt[:C], self.stride, self.pad_left, self.pad_right)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        # Ratio 2 downsampling is just a LowPass filter with stride 2
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, 
            half_width=0.6 / ratio, 
            stride=ratio, 
            kernel_size=kernel_size, 
            channels=channels
        )

    def forward(self, x):
        return self.lowpass(x)
