import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2

        # Generate filter using your existing kaiser_sinc_filter1d function
        from __main__ import kaiser_sinc_filter1d 
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        
        # PRE-EXPAND: Instead of [1, 1, K], we store [C, 1, K]
        # This makes the forward pass a direct call with no expansions
        self.register_buffer("filter", filt.expand(channels, 1, -1))

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        # Groups is now pre-set by the filter shape
        return F.conv1d(x, self.filter, stride=self.stride, groups=self.filter.shape[0])


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        
        from __main__ import kaiser_sinc_filter1d
        filt = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        
        # Store pre-expanded filter
        self.register_buffer("filter", filt.expand(channels, 1, -1))

    def forward(self, x):
        # Call the JIT-optimized function
        return fast_upsample_forward(
            x, self.filter, self.ratio, self.stride, self.pad, self.pad_left, self.pad_right
        )


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
