from typing import Tuple, Optional
import torch
import torch.nn as nn
from quantization.qat.qat_layers import Add, Mul
from quantization.qat.qat_utils import quantize_modules, replace_encoderq, replace_decoderq

EPS = 1e-8


class ConvBlock(nn.Module):
    """1D Convolutional block"""

    def __init__(
            self,
            io_channels: int,
            hidden_channels: int,
            kernel_size: int,
            padding: int,
            dilation: int = 1,
    ):
        super().__init__()

        self.shared_block = nn.Sequential(
            nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=EPS),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, groups=hidden_channels),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=EPS),
        )
        self.res_conv = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        self.skip_conv = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        self.add = Add()

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.shared_block(x)
        residual = self.res_conv(feature)
        skip_out = self.skip_conv(feature)
        feature = self.add(x, residual)
        return feature, skip_out


class MaskGenerator(nn.Module):
    """TCN (Temporal Convolution Network) Separation Module
    Generates masks for separation.
    """

    def __init__(
            self,
            input_dim: int,
            n_srcs: int,
            kernel_size: int,
            num_feats: int,
            num_hidden: int,
            num_layers: int,
            num_stacks: int,
            msk_activate: str,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_srcs = n_srcs

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=EPS),
            nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1))

        self.receptive_field = 0
        self.TCN = nn.ModuleList([])
        for s in range(num_stacks):
            for layer in range(num_layers):
                multi = 2 ** layer
                self.TCN.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                    )
                )

                self.receptive_field += kernel_size if s == 0 and layer == 0 else (kernel_size - 1) * multi

        self.adds = nn.ModuleList([Add() for i in range(len(self.TCN) - 1)])

        if msk_activate == "sigmoid":
            mask_activate_layer = nn.Sigmoid()
        elif msk_activate == "relu":
            mask_activate_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

        self.mask_net = torch.nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels=num_feats, out_channels=input_dim * n_srcs, kernel_size=1),
            mask_activate_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate separation mask"""
        batch_size = x.shape[0]
        feats = self.bottleneck(x)

        idx = 0
        feats, output = self.TCN[idx](feats)
        for layer in self.TCN[1:]:
            feats, skip = layer(feats)
            output = self.adds[idx](output, skip)
            idx += 1

        output = self.mask_net(output)

        return output.reshape(batch_size, self.n_srcs, self.input_dim, -1)


class ConvTasNetQ(nn.Module):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].
    Note: This implementation corresponds to the "non-causal" setting in the paper.
    """
    def __init__(
            self,
            n_spks: int = 1,
            n_splitter: int = 1,
            n_combiner: int = 1,
            # encoder/decoder parameters
            kernel_size: int = 32,
            stride: int = 16,
            n_filters: int = 512,
            # mask generator parameters
            mask_kernel_size: int = 3,
            bn_chan: int = 128,
            hid_chan: int = 512,
            n_blocks: int = 8,
            n_repeats: int = 3,
            mask_act: str = "relu",
    ):
        super().__init__()

        self.n_srcs = n_spks
        self.enc_num_feats = n_filters
        self.n_splitter = n_splitter
        self.n_combiner = n_combiner

        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.masker = MaskGenerator(
            input_dim=n_filters,
            n_srcs=n_spks,
            kernel_size=mask_kernel_size,
            num_feats=bn_chan,
            num_hidden=hid_chan,
            num_layers=n_blocks,
            num_stacks=n_repeats,
            msk_activate=mask_act,
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.mul = Mul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform source separation
        """
        # B: batch size
        # L: input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of speakers
        # E : enc_num_ch
        # D : dec_num_ch

        batch_size = x.shape[0]

        # ----------
        # Encoder
        # ----------
        feats = self.encoder(x)  # [B, E, L] -> [B, F, M]

        # ----------
        # Mask
        # ----------
        masked = self.mul(self.masker(feats), feats.unsqueeze(1))  # [B, S, F, M]
        masked_reshaped = torch.reshape(masked, (batch_size * self.n_srcs, self.enc_num_feats, -1))  # [B*S, F, M]

        # ----------
        # Decoder
        # ----------
        out_decoder = self.decoder(masked_reshaped)  # [B*S, D, L]
        out = out_decoder.reshape((self.n_combiner, batch_size, self.n_srcs, 1, -1)) # [D, B, S, 1, L]

        return out

    def load_pretrain(self, weights_path):
        model_state_dict = self.state_dict()
        model_state_dict_weights = torch.load(weights_path)
        model_state_dict_weights = model_state_dict_weights.get('state_dict', model_state_dict_weights)
        assert len(model_state_dict.keys()) == len(model_state_dict_weights.keys()), "Error: mismatch models weights. Please check if the model configurations match to model weights!"
        for new_key, key in zip(model_state_dict.keys(), model_state_dict_weights.keys()):
            model_state_dict[new_key] = model_state_dict_weights.get(key)
        self.load_state_dict(model_state_dict, strict=True)


    def quantize_model(self, gradient_based=True,
                       weight_quant=True, weight_n_bits=8,
                       act_quant=True, act_n_bits=8,
                       in_quant=False, in_act_n_bits=8,
                       out_quant=True, out_act_n_bits=8):

        for n, m in self.named_modules():
            if type(m) == ConvTasNetQ:
                replace_encoderq(m, ['encoder'], {'n_splitter': self.n_splitter,
                                                  'gradient_based': gradient_based,
                                                  'act_quant': act_quant,
                                                  'act_n_bits': act_n_bits,
                                                  'weight_quant': weight_quant,
                                                  'weight_n_bits': weight_n_bits,
                                                  'in_quant': in_quant,
                                                  'in_act_n_bits': in_act_n_bits})
                replace_decoderq(m, ['decoder'], {'n_combiner': self.n_combiner,
                                                  'gradient_based': gradient_based,
                                                  'act_quant': act_quant,
                                                  'act_n_bits': out_act_n_bits,
                                                  'out_quant': out_quant,
                                                  'out_act_n_bits': out_act_n_bits,
                                                  'weight_quant': weight_quant,
                                                  'weight_n_bits': weight_n_bits})
                quantize_modules(m, ['mul'], {'gradient_based': gradient_based,
                                              'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                              'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits})
            elif type(m) == ConvBlock:
                quantize_modules(m.shared_block, ['0', '1'], {'gradient_based': gradient_based,
                                                              'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                              'weight_quant': weight_quant,
                                                              'weight_n_bits': weight_n_bits})
                quantize_modules(m.shared_block, ['2'], {'gradient_based': gradient_based,
                                                         'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                         'weight_quant': weight_quant,
                                                         'weight_n_bits': weight_n_bits})
                quantize_modules(m.shared_block, ['3', '4'], {'gradient_based': gradient_based,
                                                              'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                              'weight_quant': weight_quant,
                                                              'weight_n_bits': weight_n_bits})
                quantize_modules(m.shared_block, ['5'], {'gradient_based': gradient_based,
                                                         'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                         'weight_quant': weight_quant,
                                                         'weight_n_bits': weight_n_bits})
                quantize_modules(m, ['res_conv'], {'gradient_based': gradient_based,
                                                   'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                   'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits})
                quantize_modules(m, ['skip_conv'], {'gradient_based': gradient_based,
                                                    'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                    'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits})
                quantize_modules(m, ['add'], {'gradient_based': gradient_based,
                                              'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                              'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits})
            elif type(m) == MaskGenerator:
                quantize_modules(m.bottleneck, ['0'], {'gradient_based': gradient_based,
                                                       'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                       'weight_quant': weight_quant,
                                                       'weight_n_bits': weight_n_bits})
                quantize_modules(m.bottleneck, ['1'], {'gradient_based': gradient_based,
                                                       'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                       'weight_quant': weight_quant,
                                                       'weight_n_bits': weight_n_bits})
                quantize_modules(m.mask_net, ['0'], {'gradient_based': gradient_based,
                                                     'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                     'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits})
                quantize_modules(m.mask_net, ['1', '2'], {'gradient_based': gradient_based,
                                                          'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                          'weight_quant': weight_quant,
                                                          'weight_n_bits': weight_n_bits})
                for i in range(len(m.adds)):
                    quantize_modules(m.adds, [str(i)], {'gradient_based': gradient_based,
                                                        'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                                        'weight_quant': weight_quant,
                                                        'weight_n_bits': weight_n_bits})





