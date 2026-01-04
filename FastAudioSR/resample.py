import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter
    
# Using JIT for the forward passes to fuse slicing and convs
@torch.jit.script
def fast_upsample_forward(x: torch.Tensor, weight: torch.Tensor, ratio: int, stride: int, pad: int, pad_left: int, pad_right: int):
    # F.pad with replicate is a bottleneck. 
    # By JITing this, we reduce the overhead of the Python wrapper.
    x = F.pad(x, (pad, pad), mode='replicate')
    x = F.conv_transpose1d(x, weight, stride=stride, groups=weight.shape[0])
    return x[..., pad_left:-pad_right] * float(ratio)
    
class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride: int = 1, padding: bool = True, padding_mode: str = 'replicate', kernel_size: int = 12, channels: int = 512):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.channels = channels
        
        even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2

        # 1. Register the checkpoint weight as 'filter_raw'
        # This matches the [1, 1, K] shape in your checkpoint
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
        
        # 2. Register the optimized buffer with persistent=False
        # This makes it invisible to load_state_dict()
        self.register_buffer("filter_optimized", torch.zeros(channels, 1, kernel_size), persistent=False)
        self._is_prepared = False

    def prepare(self):
        """Expands the loaded weight into the optimized buffer once"""
        with torch.no_grad():
            # Expand [1, 1, K] -> [C, 1, K]
            self.filter_optimized.copy_(self.filter.expand(self.channels, -1, -1))
        self._is_prepared = True

    def forward(self, x):
        if not self._is_prepared and not self.training:
            self.prepare()

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        
        # Use the optimized pre-expanded buffer
        f = self.filter_optimized if not self.training else self.filter.expand(x.shape[1], -1, -1)
        return F.conv1d(x, f, stride=self.stride, groups=x.shape[1])

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.channels = channels
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        
        # Match checkpoint shape
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))
        
        # Optimized buffer (invisible to state_dict)
        self.register_buffer("filter_optimized", torch.zeros(channels, 1, self.kernel_size), persistent=False)
        self._is_prepared = False

    def prepare(self):
        with torch.no_grad():
            self.filter_optimized.copy_(self.filter.expand(self.channels, -1, -1))
        self._is_prepared = True

    def forward(self, x):
        if not self._is_prepared and not self.training:
            self.prepare()

        f = self.filter_optimized if not self.training else self.filter.expand(x.shape[1], -1, -1)
        return fast_upsample_forward(x, f, self.ratio, self.stride, self.pad, self.pad_left, self.pad_right)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = ratio
        # Pass the channel count down to avoid expand() in the lowpass forward
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
            channels=channels
        )

    def forward(self, x):
        return self.lowpass(x)
