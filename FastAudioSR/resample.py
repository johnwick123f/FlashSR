import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- MATH UTILS (Used for Init only) ---
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
def fast_upsample_forward(x: Tensor, weight: Tensor, ratio: int, stride: int, pad_left: int, pad_right: int):
    # Groups = input channels. padding=0 because we handle edges with slicing
    x = F.conv_transpose1d(x, weight, stride=stride, padding=0, groups=x.shape[1])
    return x[..., pad_left:-pad_right] * float(ratio)

# --- MODULES ---

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.target_k = 3 # The speed magic number
        
        # Match checkpoint [1, 1, 12]
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        # Optimized invisible buffer [C, 1, 3]
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Slice middle 3 taps from the 12-tap filter (indices 5, 6, 7)
            short_f = self.filter[:, :, 5:8]
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        f = self.f_opt[:C] if not self.training else self.filter.expand(C, -1, -1)[:, :, 5:8]
        # padding=1 for a 3-tap filter keeps the output length identical
        return F.conv1d(x, f, stride=self.stride, padding=1, groups=C)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio, self.stride, self.channels = ratio, ratio, channels
        self.full_k = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.target_k = 3
        
        # New slicing logic for 3-tap transpose conv
        self.pad_left = (self.target_k - self.stride) // 2
        self.pad_right = (self.target_k - self.stride + 1) // 2
        
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.full_k))
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Slice center 3 taps
            start = (self.full_k - self.target_k) // 2
            short_f = self.filter[:, :, start:start+self.target_k]
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        f = self.f_opt[:C] if not self.training else self.filter.expand(C, -1, -1)[:, :, 5:8]
        return fast_upsample_forward(x, f, self.ratio, self.stride, self.pad_left, self.pad_right)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5/ratio, 0.6/ratio, stride=ratio, 
                                       kernel_size=int(6*ratio//2)*2 if kernel_size is None else kernel_size,
                                       channels=channels)
    def forward(self, x): return self.lowpass(x)
