import torch
import torch.nn as nn
from utils import get_device
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization import default_per_channel_weight_fake_quant, FakeQuantize
DEVICE = get_device()


class TorchWeightFakeQuantize(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        min_range = quantizer.min_range
        max_range = quantizer.max_range
        max_abs_range = torch.maximum(torch.abs(min_range), torch.abs(max_range))
        scales = max_abs_range / (2 ** (quantizer.n_bits - int(quantizer.sign)))
        zero_points = torch.zeros_like(scales)
        self.scales = scales.flatten()
        self.zero_points = zero_points.flatten()
        self.axis = quantizer.axis
        self.sign = quantizer.sign
        self.n_bits = quantizer.n_bits
    def forward(self, x):
        y = torch.fake_quantize_per_channel_affine(x,
                                                  scale=self.scales,
                                                  zero_point=self.zero_points,
                                                  axis=self.axis,
                                                  quant_min=-2**(self.n_bits-1) if self.sign else 0,
                                                  quant_max=2**(self.n_bits-1)-1 if self.sign else 2**self.n_bits-1)
        return y

class TorchActivationFakeQuantize(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        min_range = quantizer.min_range
        max_range = quantizer.max_range
        self.scale = float((max_range - min_range) / (2 ** quantizer.n_bits - 1))
        self.zero_point = int(torch.round(min_range / self.scale))
        self.zero_point = -self.zero_point if min_range < 0 else self.zero_point  # zp has to be positive, and a <=0, so we multiply by -1
        self.n_bits = quantizer.n_bits
    def forward(self, x):
        y = torch.fake_quantize_per_tensor_affine(x,
                                                 scale=self.scale,
                                                 zero_point=self.zero_point,
                                                 quant_min=0,
                                                 quant_max=2**self.n_bits-1)
        return y

def activation_fake_quantize():
    return FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                  quant_min=0,
                                  quant_max=255,
                                  dtype=torch.quint8,
                                  qscheme=torch.per_tensor_affine,
                                  reduce_range=False)()

def weight_fake_quantize():
    return default_per_channel_weight_fake_quant()


def round_ste(x):
    return (torch.round(x) - x).detach() + x


def floor_ste(x):
    return (torch.floor(x) - x).detach() + x


def clip_ste(x: torch.Tensor, min_val=-1.0, max_val=1.0):
    return (torch.clip(x, min=min_val, max=max_val) - x).detach() + x


def fix_range_to_include_zero(range_min: torch.Tensor, range_max: torch.Tensor, n_bits):
    min_positive = range_min > 0
    max_negative = range_max < 0
    mid_range = torch.logical_and(torch.logical_not(min_positive), torch.logical_not(max_negative))
    min_positive = min_positive.float()
    max_negative = max_negative.float()
    mid_range = mid_range.float()
    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * torch.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj
    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj


def quantize(x, min_range, max_range, n_bits, sign=True, sym=False):
    if sym:
        # Symmetric quantizer
        max_abs_range = torch.maximum(torch.abs(min_range), torch.abs(max_range))
        delta = max_abs_range / (2 ** (n_bits-int(sign)))
        X = round_ste(x / delta)
        Qmin = -2 ** (n_bits - 1) if sign else 0
        Qmax = 2 ** (n_bits - 1) - 1 if sign else 2**n_bits - 1
        y = delta * torch.clip(X, Qmin, Qmax)
    else:
        # Uniform quantizer
        delta = (max_range - min_range) / (2 ** n_bits - 1)
        zp = min_range
        X = round_ste((x - zp)/ delta)
        y = delta * torch.clip(X, 0, 2**n_bits - 1) + zp
    return y


class GradientActivationFakeQuantize(nn.Module):
    """
    Per tensor quanntization
    """
    def __init__(self, gradient_based, n_bits=8, sym=False):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.min_range = nn.Parameter(torch.Tensor([0]), requires_grad=gradient_based)
        self.max_range = nn.Parameter(torch.Tensor([0]), requires_grad=gradient_based)
        self.max_observations = 50
        self.observer_mode = True
        self.alpha = 0.9
        self.n_iter = 0
        self.sign = True

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode
        self.sign = self.min_range.sign().item()<0

    def forward(self, x):
        if self.observer_mode and self.n_iter < self.max_observations:
            self.n_iter += 1
            tilde_range_max, tilde_range_min = x.max(), x.min()
            self.min_range.data = self.alpha*self.min_range+(1-self.alpha)*tilde_range_min
            self.max_range.data = self.alpha*self.max_range+(1-self.alpha)*tilde_range_max
            self.sign = self.min_range.sign().item()<0
            return x

        y = quantize(x, self.min_range, self.max_range, self.n_bits, self.sign, self.sym)
        return y


class GradientWeightFakeQuantize(nn.Module):
    """
    Per channel quanntization
    """
    def __init__(self, gradient_based, weight_shape, n_bits=8, sym=True, ch_out_idx=0):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.axis = ch_out_idx
        self.x_dims = list(range(len(weight_shape)))
        self.x_dims.remove(ch_out_idx)
        zeros_shape = [1]*len(weight_shape)
        zeros_shape[ch_out_idx] = weight_shape[ch_out_idx]
        self.min_range = nn.Parameter(torch.zeros(zeros_shape, device=DEVICE), requires_grad=gradient_based)
        self.max_range = nn.Parameter(torch.zeros(zeros_shape, device=DEVICE), requires_grad=gradient_based)
        self.observer_mode = True
        self.sign = True

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode

    def forward(self, x):
        if self.observer_mode:
            self.max_range.data = torch.amax(x, dim=self.x_dims, keepdim=True)
            self.min_range.data = torch.amin(x, dim=self.x_dims, keepdim=True)
            self.observer_mode = False
            return x

        y = quantize(x, self.min_range, self.max_range, self.n_bits, self.sign, self.sym)
        return y


def get_activation_quantizer(gradient_based=True, n_bits=8):
    return GradientActivationFakeQuantize(gradient_based, n_bits=n_bits)


def get_weight_quantizer(gradient_based=True, weight_shape=(1,1,1), n_bits=8, ch_out_idx=0):
    return GradientWeightFakeQuantize(gradient_based, weight_shape, n_bits=n_bits, ch_out_idx=ch_out_idx)
