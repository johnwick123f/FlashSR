import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

@torch.jit.script
def snake_fast(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    # Fused Snake: 1 trig call, zero divisions
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

@torch.jit.script
def fast_upsample_forward(x: Tensor, weight: Tensor, stride: int, pad_left: int, pad_right: int):
    # Ratio is now INSIDE the weight, so we skip the final multiplication
    x = F.conv_transpose1d(x, weight, stride=stride, padding=0, groups=x.shape[1])
    return x[..., pad_left:-pad_right]

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.ratio, self.stride, self.channels = ratio, ratio, channels
        self.full_k = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.target_k = 6
        
        self.pad_left = (self.target_k - self.stride) // 2
        self.pad_right = (self.target_k - self.stride + 1) // 2
        
        # Matches checkpoint [1, 1, 12]
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.full_k))
        # Pre-expanded and Pre-scaled buffer [C, 1, 6]
        self.register_buffer("f_opt", torch.zeros(channels, 1, self.target_k), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            start = (self.full_k - self.target_k) // 2
            # SLICE + EXPAND + SCALE (Folding Step 5)
            # We multiply by ratio HERE so we don't do it in forward()
            short_f = self.filter[:, :, start:start+self.target_k] * float(self.ratio)
            self.f_opt.copy_(short_f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        # Metadata optimization: If C is always the same, this is very fast
        C = x.shape[1]
        f = self.f_opt[:C] if not self.training else (self.filter[:, :, 3:9] * float(self.ratio)).expand(C, -1, -1)
        return fast_upsample_forward(x, f, self.stride, self.pad_left, self.pad_right)

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride, self.channels = stride, channels
        self.target_k = 6
        
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))
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
        f = self.f_opt[:C] if not self.training else self.filter[:, :, 3:9].expand(C, -1, -1)
        # padding=2 for 6-tap keeps alignment
        return F.conv1d(x, f, stride=self.stride, padding=2, groups=C)

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
            # Fused Scale + inv logic (Step 5 variant)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._prepared = True

    def forward(self, x):
        if not self._prepared and not self.training: self.prepare()
        if not self.training: return snake_fast(x, self.a_eff, self.inv_2b)
        # Training path (slow but auto-grad friendly)
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5/ratio, 0.6/ratio, stride=ratio, 
                                       kernel_size=int(6*ratio//2)*2 if kernel_size is None else kernel_size,
                                       channels=channels)
    def forward(self, x): return self.lowpass(x)
