import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import _VF
from quantization.qat.qat_quant import get_weight_quantizer, get_activation_quantizer
import math

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.add(x1, x2)


class Sub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.sub(x1, x2)


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.mul(x1, x2)


class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.div(x1, x2)


class LayerQ(nn.Module):
    def __init__(self, gradient_based=True, weight_quant=False, act_quant=False, weight_shape=(1,1,1), ch_out_idx=0,
                act_n_bits=8, weight_n_bits=8, do_mac_op=False):
        super().__init__()
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.gradient_based = gradient_based
        self.activation_fake_quantize = get_activation_quantizer(gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.weight_fake_quantize = get_weight_quantizer(gradient_based, weight_shape, ch_out_idx=ch_out_idx, n_bits=weight_n_bits) if weight_quant else nn.Identity()
        self.do_mac_op = do_mac_op
        self.mac_op = 0


class AddQ(LayerQ):
    def __init__(self, add, gradient_based=True, act_quant=True, act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(add, Add):
            raise Exception(f'Quantizing wrong layer instead of Add got:{type(add)}')
        self.add = add

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        y = torch.add(x1, x2)
        return self.activation_fake_quantize(y)


class SubQ(LayerQ):
    def __init__(self, sub, gradient_based=True, act_quant=True, act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(sub, Sub):
            raise Exception(f'Quantizing wrong layer instead of Sub got:{type(sub)}')
        self.sub = sub

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        y = torch.sub(x1, x2)
        return self.activation_fake_quantize(y)


class MulQ(LayerQ):
    def __init__(self, mul, gradient_based=True, act_quant=True, act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(mul, Mul):
            raise Exception(f'Quantizing wrong layer instead of Mul got:{type(mul)}')
        self.mul = mul

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        y = torch.mul(x1, x2)
        self.calc_mac_op(x1.shape, x2.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x1_shape, x2_shape):
        if self.do_mac_op:
            mac1, mac2 = x1_shape.numel(), x2_shape.numel()
            self.mac_op = max(mac1,mac2)


class DivQ(LayerQ):
    def __init__(self, div, gradient_based=True, act_quant=True, act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(div, Div):
            raise Exception(f'Quantizing wrong layer instead of Div got:{type(div)}')
        self.div = div

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        y = torch.div(x1, x2)
        return self.activation_fake_quantize(y)


class Conv1dQ(LayerQ):
    def __init__(self, conv1d: nn.Conv1d, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based
                         , weight_quant=weight_quant,
                         act_quant=act_quant,
                         weight_shape=conv1d.weight.shape,
                         act_n_bits=act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(conv1d, nn.Conv1d):
            raise Exception(f'Quantizing wrong layer instead of Conv1d got:{type(conv1d)}')
        self.conv1d = conv1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(x,
                     weight=self.weight_fake_quantize(self.conv1d.weight),
                     bias=self.conv1d.bias,
                     stride=self.conv1d.stride,
                     padding=self.conv1d.padding,
                     dilation=self.conv1d.dilation,
                     groups=self.conv1d.groups)
        self.calc_mac_op(x.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Lo = math.floor((Li+2*self.conv1d.padding[0]-self.conv1d.dilation[0]*(self.conv1d.kernel_size[0]-1)-1)/self.conv1d.stride[0] + 1)
            self.mac_op = B*Ci*Lo*self.conv1d.kernel_size[0]


class Conv2dQ(LayerQ):
    def __init__(self, conv2d: nn.Conv2d, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=act_quant,
                         weight_shape=conv2d.weight.shape,
                         act_n_bits=act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(conv2d, nn.Conv2d):
            raise Exception(f'Quantizing wrong layer instead of Conv2d got:{type(conv2d)}')
        self.conv2d = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x,
                     weight=self.weight_fake_quantize(self.conv2d.weight),
                     bias=self.conv2d.bias,
                     stride=self.conv2d.stride,
                     padding=self.conv2d.padding,
                     dilation=self.conv2d.dilation,
                     groups=self.conv2d.groups)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Loh = math.floor((Li+2*self.conv2d.padding[0]-self.conv2d.dilation[0]*(self.conv2d.kernel_size[0]-1)-1)/self.conv2d.stride[0] + 1)
            Low = math.floor((Li+2*self.conv2d.padding[1]-self.conv2d.dilation[1]*(self.conv2d.kernel_size[1]-1)-1)/self.conv2d.stride[1] + 1)
            self.mac_op = B*Ci*(Loh*self.conv2d.kernel_size[0] + Low*self.conv2d.kernel_size[1])


class ConvTranspose1dQ(LayerQ):
    def __init__(self, convTr1d: nn.ConvTranspose1d, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=act_quant,
                         weight_shape=convTr1d.weight.shape,
                         ch_out_idx=1,
                         act_n_bits=act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(convTr1d, nn.ConvTranspose1d):
            raise Exception(f'Quantizing wrong layer instead of ConvTranspose1d got:{type(convTr1d)}')
        self.convTr1d = convTr1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose1d(x,
                                weight=self.weight_fake_quantize(self.convTr1d.weight),
                                bias=self.convTr1d.bias,
                                stride=self.convTr1d.stride,
                                padding=self.convTr1d.padding,
                                output_padding=self.convTr1d.output_padding,
                                dilation=self.convTr1d.dilation,
                                groups=self.convTr1d.groups)
        self.calc_mac_op(x.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Lo = (Li-1)*self.convTr1d.stride[0]-2*self.convTr1d.padding[0]+self.convTr1d.dilation[0]*(self.convTr1d.kernel_size[0]-1)+self.convTr1d.output_padding[0]+1
            self.mac_op = B*Ci*Lo*(self.convTr1d.kernel_size[0]//self.convTr1d.stride[0])


class GroupNormQ(LayerQ):
    def __init__(self, groupnorm: nn.GroupNorm, gradient_based=True, act_quant=True,
                  act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(groupnorm, nn.GroupNorm):
            raise Exception(f'Quantizing wrong layer instead of GroupNorm got:{type(groupnorm)}')
        self.groupnorm = groupnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.groupnorm(x)
        self.calc_mac_op(x.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            self.mac_op = x_shape.numel()


class LayerNormQ(LayerQ):
    def __init__(self, layernorm: nn.LayerNorm, gradient_based=True, act_quant=True,
                  act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(layernorm, nn.LayerNorm):
            raise Exception(f'Quantizing wrong layer instead of LayerNorm got:{type(layernorm)}')
        self.layernorm = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layernorm(x)
        self.calc_mac_op(x.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            self.mac_op = x_shape.numel()


class Conv1dNlQ(LayerQ):
    def __init__(self, conv1d: nn.Conv1d, nl, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=act_quant,
                         weight_shape=conv1d.weight.shape,
                         act_n_bits=act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(conv1d, nn.Conv1d):
            raise Exception(f'Quantizing wrong layer instead of Conv1d got:{type(conv1d)}')
        self.conv1d = conv1d
        self.nl = nl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(x,
                     weight=self.weight_fake_quantize(self.conv1d.weight),
                     bias=self.conv1d.bias,
                     stride=self.conv1d.stride,
                     padding=self.conv1d.padding,
                     dilation=self.conv1d.dilation,
                     groups=self.conv1d.groups)
        self.calc_mac_op(x.shape)
        z = self.nl(y)
        return self.activation_fake_quantize(z)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Lo = math.floor((Li+2*self.conv1d.padding[0]-self.conv1d.dilation[0]*(self.conv1d.kernel_size[0]-1)-1)/self.conv1d.stride[0] + 1)
            self.mac_op = B*Ci*Lo*self.conv1d.kernel_size[0]


class NlQ(LayerQ):
    def __init__(self, nl, gradient_based=True, act_quant=True, act_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        self.nl = nl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.nl(x)
        return self.activation_fake_quantize(y)


class LinearQ(LayerQ):
    def __init__(self, linear: nn.Linear, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based, weight_quant=weight_quant, act_quant=act_quant, weight_shape=linear.weight.shape,
                         act_n_bits=act_n_bits, weight_n_bits=weight_n_bits)
        if not isinstance(linear, nn.Linear):
            raise Exception(f'Quantizing wrong layer instead of Linear got:{type(linear)}')
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x,
                     weight=self.weight_fake_quantize(self.linear.weight),
                     bias=self.linear.bias)
        self.calc_mac_op(x.shape, self.linear.weight.shape)
        return self.activation_fake_quantize(y)

    def calc_mac_op(self, x_shape, w_shape):
        if self.do_mac_op:
            B, Li, Ci = x_shape
            Fi, Ci = w_shape
            self.mac_op = B*Li*Fi*Ci


class LSTMQ(LayerQ):
    def __init__(self, lstm: nn.LSTM, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(lstm, nn.LSTM):
            raise Exception(f'Quantizing wrong layer instead of LSTM got:{type(lstm)}')
        self.lstm = lstm
        self.num_directions = 2 if self.lstm.bidirectional else 1
        self.real_hidden_size = self.lstm.proj_size if self.lstm.proj_size > 0 else self.lstm.hidden_size
        self.weight_quantizers_dict = nn.ModuleDict([])
        for name, weights in zip(self.lstm._flat_weights_names, self.lstm._flat_weights):
            if name.startswith("weight"):
                self.weight_quantizers_dict.update({name: get_weight_quantizer(self.gradient_based, weights.shape, n_bits=weight_n_bits) if weight_quant else nn.Identity()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights parameters
        flat_weights, quant_id = [], 0
        for name, weights in zip(self.lstm._flat_weights_names, self.lstm._flat_weights):
            if name.startswith("weight"):
                weights = self.weight_quantizers_dict[name](self.lstm._flat_weights[self.lstm._flat_weights_names.index(name)])
            flat_weights.append(weights)

        max_batch_size = x.size(0) if self.lstm.batch_first else x.size(1)
        h_zeros = torch.zeros(self.lstm.num_layers * self.num_directions, max_batch_size, self.real_hidden_size, dtype=x.dtype, device=x.device)
        c_zeros = torch.zeros(self.lstm.num_layers * self.num_directions, max_batch_size, self.lstm.hidden_size, dtype=x.dtype, device=x.device)
        hx = (h_zeros, c_zeros)
        y = _VF.lstm(x, hx, flat_weights, self.lstm.bias, self.lstm.num_layers,
                     self.lstm.dropout, self.lstm.training, self.lstm.bidirectional, self.lstm.batch_first)
        self.calc_mac_op(x.shape)
        return [self.activation_fake_quantize(y[0])]

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Li, Ci = x_shape
            for name, weights in zip(self.lstm._flat_weights_names, self.lstm._flat_weights):
                if name.startswith("weight"):
                    weights = self.weight_quantizers_dict[name](self.lstm._flat_weights[self.lstm._flat_weights_names.index(name)])
                    Fw, Cw = weights.shape
                    if Cw==Ci:
                        self.mac_op += B * Li * Fw * Cw
                    else:
                        self.mac_op += B * self.real_hidden_size * Fw * Cw
                    self.mac_op += 3 * B * self.real_hidden_size


class LSTMQ_full_quantized(LayerQ):
    def __init__(self, lstm: nn.LSTM, gradient_based=True, weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(lstm, nn.LSTM):
            raise Exception(f'Quantizing wrong layer instead of LSTM got:{type(lstm)}')
        self.lstm = lstm
        self.num_directions = 2 if self.lstm.bidirectional else 1
        self.real_hidden_size = self.lstm.proj_size if self.lstm.proj_size > 0 else self.lstm.hidden_size
        self.weight_quantizers_dict = nn.ModuleDict([])
        for name, weights in zip(self.lstm._flat_weights_names, self.lstm._flat_weights):
            if name.startswith("weight"):
                self.weight_quantizers_dict.update({name: get_weight_quantizer(self.gradient_based, weights.shape, n_bits=weight_n_bits) if weight_quant else nn.Identity()})

        self.activation_fake_quantize_ih = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_hh = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_add0 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_add1 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_mul0 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_mul1 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_mul2 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_sig0 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_sig1 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_sig2 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_tanh0 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_tanh1 = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
        if self.lstm.bidirectional:
            self.activation_fake_quantize_ih_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_hh_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_add0_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_add1_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_mul0_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_mul1_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_mul2_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_sig0_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_sig1_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_sig2_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_tanh0_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()
            self.activation_fake_quantize_tanh1_r = get_activation_quantizer(self.gradient_based, n_bits=act_n_bits) if act_quant else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.size(0) if self.lstm.batch_first else x.size(1)
        if self.lstm.batch_first:
            x = x.permute(1,0,2)

        # x: shape (seq_len, batch_size, input_size)

        # Quantize weights
        Wih = self.weight_quantizers_dict['weight_ih_l0'](self.lstm._flat_weights[self.lstm._flat_weights_names.index('weight_ih_l0')])
        bih = self.lstm._flat_weights[self.lstm._flat_weights_names.index('bias_ih_l0')]
        Whh = self.weight_quantizers_dict['weight_hh_l0'](self.lstm._flat_weights[self.lstm._flat_weights_names.index('weight_hh_l0')])
        bhh = self.lstm._flat_weights[self.lstm._flat_weights_names.index('bias_hh_l0')]
        if self.lstm.bidirectional:
            Wih_r = self.weight_quantizers_dict['weight_ih_l0_reverse'](self.lstm._flat_weights[self.lstm._flat_weights_names.index('weight_ih_l0_reverse')])
            bih_r = self.lstm._flat_weights[self.lstm._flat_weights_names.index('bias_ih_l0_reverse')]
            Whh_r = self.weight_quantizers_dict['weight_hh_l0_reverse'](self.lstm._flat_weights[self.lstm._flat_weights_names.index('weight_hh_l0_reverse')])
            bhh_r = self.lstm._flat_weights[self.lstm._flat_weights_names.index('bias_hh_l0_reverse')]

        h_zeros = torch.zeros(self.lstm.num_layers * self.num_directions, batch_size, self.real_hidden_size, dtype=x.dtype, device=x.device)
        c_zeros = torch.zeros(self.lstm.num_layers * self.num_directions, batch_size, self.lstm.hidden_size, dtype=x.dtype, device=x.device)

        outputs_fw, outputs_bw = [], []

        hx_fw = h_zeros[:self.lstm.num_layers][0]
        cx_fw = c_zeros[:self.lstm.num_layers][0]
        hx_bw = h_zeros[self.lstm.num_layers:self.lstm.num_layers*self.num_directions][0]
        cx_bw = c_zeros[self.lstm.num_layers:self.lstm.num_layers*self.num_directions][0]

        for t in range(x.shape[0]):
            ih = self.activation_fake_quantize_ih(F.linear(x[t], Wih, bih))
            hh = self.activation_fake_quantize_hh(F.linear(hx_fw, Whh, bhh))
            gates_fw = self.activation_fake_quantize_add0(ih+hh) # gates_fw: shape (batch_size, 4 * hidden_size)
            gate_i_fw, gate_f_fw, gate_g_fw, gate_o_fw = gates_fw.chunk(4, 1)
            gate_i_fw = self.activation_fake_quantize_sig0(torch.sigmoid(gate_i_fw))
            gate_f_fw = self.activation_fake_quantize_sig1(torch.sigmoid(gate_f_fw))
            gate_g_fw = self.activation_fake_quantize_tanh0(torch.tanh(gate_g_fw))
            gate_o_fw = self.activation_fake_quantize_sig2(torch.sigmoid(gate_o_fw))
            cx_fw = self.activation_fake_quantize_add1(self.activation_fake_quantize_mul0(gate_f_fw*cx_fw)
                                                       +self.activation_fake_quantize_mul1(gate_i_fw*gate_g_fw))
            hx_fw = self.activation_fake_quantize_mul2(gate_o_fw*self.activation_fake_quantize_tanh1(torch.tanh(cx_fw)))
            outputs_fw.append(hx_fw.unsqueeze(0))

            if self.lstm.bidirectional:
                ih_r = self.activation_fake_quantize_ih_r(F.linear(x[-t-1], Wih_r, bih_r))
                hh_r = self.activation_fake_quantize_hh_r(F.linear(hx_bw, Whh_r, bhh_r))
                gates_bw = self.activation_fake_quantize_add0_r(ih_r+hh_r) # gates_bw: shape (batch_size, 4 * hidden_size)
                gate_i_bw, gate_f_bw, gate_g_bw, gate_o_bw = gates_bw.chunk(4, 1)
                gate_i_bw = self.activation_fake_quantize_sig0_r(torch.sigmoid(gate_i_bw))
                gate_f_bw = self.activation_fake_quantize_sig1_r(torch.sigmoid(gate_f_bw))
                gate_g_bw = self.activation_fake_quantize_tanh0_r(torch.tanh(gate_g_bw))
                gate_o_bw = self.activation_fake_quantize_sig2_r(torch.sigmoid(gate_o_bw))
                cx_bw = self.activation_fake_quantize_add1_r(self.activation_fake_quantize_mul0_r(gate_f_bw*cx_bw)
                                                           +self.activation_fake_quantize_mul1_r(gate_i_bw*gate_g_bw))
                hx_bw = self.activation_fake_quantize_mul2(gate_o_bw*self.activation_fake_quantize_tanh1_r(torch.tanh(cx_bw)))
                outputs_bw.append(hx_bw.unsqueeze(0))

        outputs_fw = torch.cat(outputs_fw, dim=0)
        if self.lstm.bidirectional:
            outputs_bw = torch.flip(torch.cat(outputs_bw, dim=0), [0])
            y = torch.cat([outputs_fw, outputs_bw], dim=-1)
        else:
            y = outputs_fw

        if self.lstm.batch_first:
            y = y.permute(1,0,2)

        return [self.activation_fake_quantize(y)]


class MultiheadAttentionQ(LayerQ):
    def __init__(self, mha: nn.MultiheadAttention, gradient_based=True,
                 weight_quant=True, act_quant=True,
                 act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        if not isinstance(mha, nn.MultiheadAttention):
            raise Exception(f'Quantizing wrong layer instead of MultiheadAttention got:{type(mha)}')
        self.mha = mha
        self.do, _ = mha.out_proj.weight.shape
        self.head_dim = self.mha.embed_dim // self.mha.num_heads
        self.activation_fake_quantize_in = get_activation_quantizer(self.gradient_based,
                                                                    n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_div = get_activation_quantizer(self.gradient_based,
                                                                         n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_attn = get_activation_quantizer(self.gradient_based,
                                                                      n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_softmax = get_activation_quantizer(self.gradient_based,
                                                                         n_bits=act_n_bits) if act_quant else nn.Identity()
        self.activation_fake_quantize_head = get_activation_quantizer(self.gradient_based,
                                                                      n_bits=act_n_bits) if act_quant else nn.Identity()
        self.weight_fake_quantize_in = get_weight_quantizer(self.gradient_based,
                                                            self.mha.in_proj_weight.shape,
                                                            n_bits=weight_n_bits) if weight_quant else nn.Identity()
        self.weight_fake_quantize_out = get_weight_quantizer(self.gradient_based,
                                                             self.mha.out_proj.weight.shape,
                                                             n_bits=weight_n_bits) if weight_quant else nn.Identity()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Note: Only support self attention
        # Quantize weights
        Wi = self.weight_fake_quantize_in(self.mha.in_proj_weight)
        Wo = self.weight_fake_quantize_out(self.mha.out_proj.weight)

        len_x, batch, d_x = query.shape
        # Input projection
        X = F.linear(query, Wi, self.mha.in_proj_bias)
        X = self.activation_fake_quantize_in(X)
        Q, K, V = X.chunk(3, dim=-1)
        # Reshape
        q = Q.reshape((len_x, batch*self.mha.num_heads, self.head_dim)).permute(1,0,2)
        k = K.reshape((len_x, batch*self.mha.num_heads, self.head_dim)).permute(1,0,2)
        v = V.reshape((len_x, batch*self.mha.num_heads, self.head_dim)).permute(1,0,2)
        # Matrix multiplication
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        q = self.activation_fake_quantize_div(q)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn - self.activation_fake_quantize_attn(attn)
        attn = torch.softmax(attn, dim=-1)
        attn - self.activation_fake_quantize_softmax(attn)
        heads = torch.bmm(attn, v)
        heads = self.activation_fake_quantize_head(heads)
        heads_reshape = heads.transpose(1,0).reshape((len_x*batch,self.mha.embed_dim))
        # Output projection
        y = F.linear(heads_reshape, Wo, self.mha.out_proj.bias)
        y = y.reshape((len_x, batch, self.do))
        self.calc_linear_mac_op(query.shape, Wi.shape, heads_reshape.shape, Wo.shape)
        self.calc_bmm_mac_op(q,k,v)
        return self.activation_fake_quantize(y),

    def forward_torch(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        y = F.multi_head_attention_forward(query, key, value,
                                           embed_dim_to_check=self.mha.embed_dim,
                                           num_heads=self.mha.num_heads,
                                           in_proj_weight=self.weight_fake_quantize_in(self.mha.in_proj_weight),
                                           in_proj_bias=self.mha.in_proj_bias,
                                           bias_k=self.mha.bias_k,
                                           bias_v=self.mha.bias_v,
                                           add_zero_attn=self.mha.add_zero_attn,
                                           dropout_p=self.mha.dropout,
                                           out_proj_weight=self.weight_fake_quantize_out(self.mha.out_proj.weight),
                                           out_proj_bias=self.mha.out_proj.bias)
        return self.activation_fake_quantize(y[0])

    def calc_linear_mac_op(self, xi_shape, wi_shape, xo_shape, wo_shape):
        if self.do_mac_op:
            B, L, C = xi_shape
            F, C = wi_shape
            self.mac_op = B*L*F*C
            BL, C = xo_shape
            F, C = wo_shape
            self.mac_op += BL*F*C

    def calc_bmm_mac_op(self, q, k, v):
        if self.do_mac_op:
            batch, q2, q3 = q.shape
            batch, k2, k3 = k.shape
            batch, v2, v3 = v.shape
            self.mac_op += batch*q2*k2*q3 # atten = q X k^T = [batch,q2,q3] X [batch,k3,k2] = [batch,q2,k2]
            self.mac_op += batch*q2*v3*k2 # = [batch,q2,k2] X [batch,v2,v3] = [batch, q2, v3]


class Conv1dEncoderQ(LayerQ):
    def __init__(self, encoder, n_splitter=1, gradient_based=True, weight_quant=True, act_quant=True,
                 in_quant=False,
                 act_n_bits=8, weight_n_bits=8, in_act_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=act_quant,
                         weight_shape=encoder[0].weight.shape,
                         act_n_bits=act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(encoder[0], nn.Conv1d):
            raise Exception(f'Quantizing wrong layer instead of Conv1d got:{type(encoder[0])}')
        self.in_quantizer = get_activation_quantizer(self.gradient_based,
                                                     n_bits=in_act_n_bits) if in_quant else nn.Identity()
        self.conv1d = encoder[0]
        self.nl = nn.Identity() if len(encoder) == 1 else encoder[1]
        if n_splitter >= 2:
            weight = self.conv1d.state_dict()['weight']
            in_channels = self.conv1d.in_channels
            self.conv1d = nn.Conv1d(in_channels=n_splitter * in_channels,
                                    out_channels=self.conv1d.out_channels,
                                    kernel_size=self.conv1d.kernel_size,
                                    stride=self.conv1d.stride,
                                    padding=0,
                                    bias=False)
            new_weight = weight.repeat(1, n_splitter, 1)
            for n_ch in range(1, n_splitter):
                for in_channel in range(in_channels):
                    weight_gaussian = torch.mean(weight[:, in_channel, :]) + torch.randn_like(
                        weight[:, in_channel, :]) * (torch.std(weight[:, in_channel, :]) ** n_ch)  # gaussian
                    new_weight[:, n_ch * in_channels + in_channel, :] = weight_gaussian
            self.conv1d.load_state_dict({'weight': new_weight})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_quantizer(x)
        y = F.conv1d(x,
                     weight=self.weight_fake_quantize(self.conv1d.weight),
                     bias=self.conv1d.bias,
                     stride=self.conv1d.stride,
                     padding=self.conv1d.padding,
                     dilation=self.conv1d.dilation,
                     groups=self.conv1d.groups)
        self.calc_mac_op(x.shape)
        z = self.nl(y)
        return self.activation_fake_quantize(z)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Lo = math.floor((Li+2*self.conv1d.padding[0]-self.conv1d.dilation[0]*(self.conv1d.kernel_size[0]-1)-1)/self.conv1d.stride[0] + 1)
            self.mac_op = B*Ci*Lo*self.conv1d.kernel_size[0]


class ResidualErrorBlock(LayerQ):
    def __init__(self, decoder, gradient_based, weight_quant, act_quant, act_n_bits=8, weight_n_bits=8):
        super().__init__(gradient_based=gradient_based, act_quant=act_quant, act_n_bits=act_n_bits)
        self.decoder_type = type(decoder)
        if self.decoder_type == nn.Linear:
            self.residual_encoder = nn.Linear(in_features=decoder.out_features,
                                             out_features=decoder.in_features,
                                             bias=False)
            self.decoder_bias = decoder.bias
        elif self.decoder_type == nn.ConvTranspose1d:
            self.residual_encoder = nn.Conv1d(in_channels=decoder.out_channels,
                                             out_channels=decoder.in_channels,
                                             kernel_size=decoder.kernel_size,
                                             stride=decoder.stride,
                                             padding=0,
                                             bias=False)
            self.decoder_bias = decoder.bias
            self.decoder_kernel = decoder.kernel_size
            self.decoder_stride = decoder.stride
            self.decoder_padding = decoder.padding
            self.decoder_output_padding = decoder.output_padding
            self.decoder_dilation = decoder.dilation
            self.decoder_groups = decoder.groups
        else:
            assert False, "Not supported residual block for type {}".format(self.decoder_type)

        self.weight_fake_quantize = get_weight_quantizer(gradient_based,
                                                         self.residual_encoder.weight.shape,
                                                         n_bits=weight_n_bits) if weight_quant else nn.Identity()

    def forward(self, Y, y_q, w_decoder):
        if self.decoder_type == nn.Linear:
            Y_q = F.linear(y_q,
                          weight=self.weight_fake_quantize(self.residual_encoder.weight),
                          bias=self.residual_encoder.bias)
            Y1 = self.activation_fake_quantize(Y - Y_q)
            y1 = F.linear(Y1,
                         weight=w_decoder,
                         bias=self.decoder_bias)
            self.calc_linear_mac_op(y_q.shape, self.residual_encoder.weight.shape, Y1.shape, w_decoder.shape)
        elif self.decoder_type == nn.ConvTranspose1d:
            Y_q = F.conv1d(y_q,
                          weight=self.weight_fake_quantize(self.residual_encoder.weight),
                          bias=self.residual_encoder.bias,
                          stride=self.residual_encoder.stride)
            Y1 = self.activation_fake_quantize(Y - Y_q)
            y1 = F.conv_transpose1d(Y1,
                                   weight=w_decoder,
                                   bias=self.decoder_bias,
                                   stride=self.decoder_stride,
                                   padding=self.decoder_padding,
                                   output_padding=self.decoder_output_padding,
                                   dilation=self.decoder_dilation,
                                   groups=self.decoder_groups)
            self.calc_conv_mac_op(y_q.shape, Y1.shape)
        else:
            assert False, "No support!"
        return y1

    def calc_conv_mac_op(self, y_q_shape, Y1_shape):
        if self.do_mac_op:
            B, Ci, Li = y_q_shape
            Lo = math.floor((Li+2*self.residual_encoder.padding[0]-self.residual_encoder.dilation[0]*(self.residual_encoder.kernel_size[0]-1)-1)/self.residual_encoder.stride[0] + 1)
            self.mac_op = B*Ci*Lo*self.residual_encoder.kernel_size[0]
            B, Ci, Li = Y1_shape
            Lo = (Li-1)*self.decoder_stride[0]-2*self.decoder_padding[0]+self.decoder_dilation[0]*(self.decoder_kernel[0]-1)+self.decoder_output_padding[0]+1
            self.mac_op += B*Ci*Lo*(self.decoder_kernel[0]//self.decoder_stride[0])

    def calc_linear_mac_op(self, y_q_shape, w_q_shape, Y1_shape, w_decoder_shape):
        if self.do_mac_op:
            B, S, Li, Ci = y_q_shape
            Ci, Fi = w_q_shape
            self.mac_op = B*S*Li*Fi*Ci
            B, S, Li, Ci = Y1_shape
            Ci, Fi = w_decoder_shape
            self.mac_op += B*S*Li*Fi*Ci


class LinearDecoderQ(LayerQ):
    def __init__(self, decoder, n_combiner=1, gradient_based=True,
                 weight_quant=True, weight_n_bits=8,
                 act_quant=True, act_n_bits=8,
                 out_quant=True, out_act_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=out_quant,
                         weight_shape=decoder[0].weight.shape,
                         act_n_bits=out_act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(decoder[0], nn.Linear):
            raise Exception(f'Quantizing wrong layer instead of Linear got:{type(decoder[0])}')
        self.linear = decoder[0]
        self.n_combiner = n_combiner
        if self.n_combiner >= 2:
            self.residual_error_block = ResidualErrorBlock(self.linear, gradient_based, weight_quant, act_quant,
                                                           act_n_bits=act_n_bits,
                                                           weight_n_bits=weight_n_bits)
            self.activation_fake_quantize_residual = get_activation_quantizer(gradient_based,n_bits=out_act_n_bits) if out_quant else nn.Identity()

    def forward(self, x: torch.Tensor):
        w_decoder = self.weight_fake_quantize(self.linear.weight)
        x0 = F.linear(x,
                     weight=w_decoder,
                     bias=self.linear.bias)
        self.calc_mac_op(x.shape, w_decoder.shape)
        y = self.activation_fake_quantize(x0)
        if self.n_combiner == 1:
            return y

        # ----------------------------
        # Residual Error blocks
        # ----------------------------
        outs = [y]
        for ch in range(1,self.n_combiner):
            x = self.residual_error_block(x, y, w_decoder)
            y = self.activation_fake_quantize_residual(x)
            outs.append(y)

        return torch.stack(outs)

    def calc_mac_op(self, x_shape, w_shape):
        if self.do_mac_op:
            B, S, Li, Ci = x_shape
            Fi, Ci = w_shape
            self.mac_op = B*S*Li*Fi*Ci


class ConvTr1dDecoderQ(LayerQ):
    def __init__(self, decoder, n_combiner=1, gradient_based=True,
                 weight_quant=True, weight_n_bits=8,
                 act_quant=True, act_n_bits=8,
                 out_quant=True, out_act_n_bits=8):
        super().__init__(gradient_based=gradient_based,
                         weight_quant=weight_quant,
                         act_quant=out_quant,
                         weight_shape=decoder[0].weight.shape,
                         ch_out_idx=1,
                         act_n_bits=out_act_n_bits,
                         weight_n_bits=weight_n_bits)
        if not isinstance(decoder[0], nn.ConvTranspose1d):
            raise Exception(f'Quantizing wrong layer instead of ConvTranspose1d got:{type(decoder[0])}')
        self.n_combiner = n_combiner
        self.convTr1d = decoder[0]

        if self.n_combiner >= 2:
            self.residual_error_block = ResidualErrorBlock(self.convTr1d, gradient_based,
                                                           weight_quant=weight_quant, act_quant=act_quant,
                                                           weight_n_bits=weight_n_bits, act_n_bits=act_n_bits)
            self.activation_fake_quantize_residual = get_activation_quantizer(gradient_based,
                                                                              n_bits=out_act_n_bits) if out_quant else nn.Identity()

    def forward(self, x: torch.Tensor):
        w_decoder = self.weight_fake_quantize(self.convTr1d.weight)
        x0 = F.conv_transpose1d(x,
                                weight=w_decoder,
                                bias=self.convTr1d.bias,
                                stride=self.convTr1d.stride,
                                padding=self.convTr1d.padding,
                                output_padding=self.convTr1d.output_padding,
                                dilation=self.convTr1d.dilation,
                                groups=self.convTr1d.groups)
        self.calc_mac_op(x.shape)
        y = self.activation_fake_quantize(x0)
        if self.n_combiner == 1:
            return y

        # ----------------------------
        # Residual Error blocks
        # ----------------------------
        outs = [y]
        for ch in range(1, self.n_combiner):
            x = self.residual_error_block(x, y, w_decoder)
            y = self.activation_fake_quantize_residual(x)
            outs.append(y)

        return torch.stack(outs)

    def calc_mac_op(self, x_shape):
        if self.do_mac_op:
            B, Ci, Li = x_shape
            Lo = (Li-1)*self.convTr1d.stride[0]-2*self.convTr1d.padding[0]+self.convTr1d.dilation[0]*(self.convTr1d.kernel_size[0]-1)+self.convTr1d.output_padding[0]+1
            self.mac_op = B*Ci*Lo*(self.convTr1d.kernel_size[0]//self.convTr1d.stride[0])