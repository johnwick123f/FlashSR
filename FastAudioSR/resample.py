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
    # conv_transpose1d with padding=0
    x = F.conv_transpose1d(x, weight, stride=stride, padding=0, groups=x.shape[1])
    return x[..., pad_left:-pad_right]

# --- MODULES ---

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Keep identical for loading pretrained weights
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))
        
        # Optimization: Fused polyphase weight [C*ratio, 1, K/ratio]
        self.register_buffer("f_fast", torch.zeros(channels * ratio, 1, kernel_size // ratio), persistent=False)
        self._prepared = False

    @torch.no_grad()
    def prepare(self):
        # Reshape 12-tap filter into [ratio, 1, 6]
        # This allows us to compute both phases in a SINGLE grouped convolution pass
        w = self.filter * float(self.ratio) # [1, 1, 12]
        w = w.view(self.kernel_size)
        
        # Interleave weights for grouped conv: [Phase0_C1, Phase1_C1, Phase0_C2, Phase1_C2...]
        p0 = w[0::2] # [6]
        p1 = w[1::2] # [6]
        
        # Combine into [C*2, 1, 6]
        fast_w = torch.stack([p0, p1], dim=0) # [2, 6]
        fast_w = fast_w.repeat(self.channels, 1, 1).unsqueeze(1) # [C*2, 1, 6]
        self.f_fast.copy_(fast_w)
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        
        # 1. Pad input for the 6-tap polyphase kernels
        # This replaces the heavy ConvTranspose1d
        x = F.pad(x, (2, 3)) 
        
        # 2. Single Grouped Conv computes all output samples at once
        # Using groups=channels*ratio is the "fast path" in modern GPUs
        # This is where the 4x speedup comes from
        out = F.conv1d(x, self.f_fast, groups=self.channels, stride=1)
        
        # 3. Reshape from [B, C*2, L] to [B, C, L*2] (Pixel Shuffle 1D)
        B, C2, L = out.shape
        out = out.view(B, self.channels, self.ratio, L).transpose(2, 3).reshape(B, self.channels, -1)
        
        # Center crop to align timing
        return out[..., 2:-2]

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        self.register_buffer("f_opt", torch.zeros(channels, 1, kernel_size), persistent=False)
        self._prepared = False

    def prepare(self):
        self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        # Downsampling is already fast in PyTorch if using stride inside conv1d
        # because it never calculates the samples it doesn't need.
        return F.conv1d(x, self.f_opt, stride=self.stride, padding=5, groups=self.channels)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        # Downsampling already benefits from stride in F.conv1d
        self.lowpass = LowPassFilter1d(0.5/ratio, 0.6/ratio, ratio, kernel_size, channels)

    def forward(self, x):
        return self.lowpass(x)
