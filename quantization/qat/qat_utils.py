import torch
import torch.nn as nn
import copy
from quantization.qat.qat_layers import Conv1dEncoderQ, ConvTr1dDecoderQ, LinearDecoderQ
from quantization.qat.qat_layers import Add, Sub, Mul, Div
from quantization.qat.qat_layers import Conv1dQ, Conv2dQ, ConvTranspose1dQ, Conv1dNlQ, NlQ, GroupNormQ, AddQ, SubQ, MulQ, DivQ
from quantization.qat.qat_layers import LSTMQ, MultiheadAttentionQ, LinearQ, LayerNormQ
from quantization.qat.qat_quant import TorchActivationFakeQuantize, TorchWeightFakeQuantize

def quant_encoderq(encoder, params_dict):
    return Conv1dEncoderQ(encoder,
                          n_splitter=params_dict.get('n_splitter'),
                          gradient_based=params_dict.get('gradient_based'),
                          act_quant=params_dict.get('act_quant'),
                          weight_quant=params_dict.get('weight_quant'),
                          in_quant=params_dict.get('in_quant'),
                          act_n_bits=params_dict.get('act_n_bits'),
                          weight_n_bits=params_dict.get('weight_n_bits'),
                          in_act_n_bits=params_dict.get('in_act_n_bits'))

def quant_convtranspose_decoderq(decoder, params_dict):
    return ConvTr1dDecoderQ(decoder,
                            n_combiner=params_dict.get('n_combiner'),
                            gradient_based=params_dict.get('gradient_based'),
                            act_quant=params_dict.get('act_quant'),
                            weight_quant=params_dict.get('weight_quant'),
                            weight_n_bits=params_dict.get('weight_n_bits'),
                            act_n_bits=params_dict.get('act_n_bits'),
                            out_quant=params_dict.get('out_quant'),
                            out_act_n_bits=params_dict.get('out_act_n_bits'))

def quant_linear_decoderq(decoder, params_dict):
    return LinearDecoderQ(decoder,
                        n_combiner=params_dict.get('n_combiner'),
                        gradient_based=params_dict.get('gradient_based'),
                        act_quant=params_dict.get('act_quant'),
                        weight_quant=params_dict.get('weight_quant'),
                        weight_n_bits=params_dict.get('weight_n_bits'),
                        act_n_bits=params_dict.get('act_n_bits'),
                        out_quant=params_dict.get('out_quant'),
                        out_act_n_bits=params_dict.get('out_act_n_bits'))


def quant_conv1d(conv1d, params_dict):
    return Conv1dQ(conv1d,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv2d(conv2d, params_dict):
    return Conv2dQ(conv2d,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_convtr1d(convTr1d, params_dict):
    return ConvTranspose1dQ(convTr1d,
                            gradient_based=params_dict.get('gradient_based'),
                            act_quant=params_dict.get('act_quant'),
                            weight_quant=params_dict.get('weight_quant'),
                            act_n_bits=params_dict.get('act_n_bits'),
                            weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv1d_nl(conv1d, nl, params_dict):
    return Conv1dNlQ(conv1d, nl,
                     gradient_based=params_dict.get('gradient_based'),
                     act_quant=params_dict.get('act_quant'),
                     weight_quant=params_dict.get('weight_quant'),
                     act_n_bits=params_dict.get('act_n_bits'),
                     weight_n_bits=params_dict.get('weight_n_bits'))

def quant_groupnorm(groupnorm, params_dict):
    return GroupNormQ(groupnorm,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      act_n_bits=params_dict.get('act_n_bits'))

def quant_layernorm(layernorm, params_dict):
    return LayerNormQ(layernorm,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      act_n_bits=params_dict.get('act_n_bits'))

def quant_nl(nl, params_dict):
    return NlQ(nl,
               gradient_based=params_dict.get('gradient_based'),
               act_quant=params_dict.get('act_quant'),
               act_n_bits=params_dict.get('act_n_bits'))

def quant_linear(linear, params_dict):
    return LinearQ(linear,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_mha(mha, params_dict):
    return MultiheadAttentionQ(mha,
                               gradient_based=params_dict.get('gradient_based'),
                               act_quant=params_dict.get('act_quant'),
                               weight_quant=params_dict.get('weight_quant'),
                               act_n_bits=params_dict.get('act_n_bits'),
                               weight_n_bits=params_dict.get('weight_n_bits'))

def quant_lstm(lstm, params_dict):
    return LSTMQ(lstm,
                 gradient_based=params_dict.get('gradient_based'),
                 act_quant=params_dict.get('act_quant'),
                 weight_quant=params_dict.get('weight_quant'),
                 act_n_bits=params_dict.get('act_n_bits'),
                 weight_n_bits=params_dict.get('weight_n_bits'))

def quant_add(add, params_dict):
    return AddQ(add,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))

def quant_sub(sub, params_dict):
    return SubQ(sub,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))


def quant_mul(mul, params_dict):
    return MulQ(mul,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))

def quant_div(div, params_dict):
    return DivQ(div,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))


def torch_weight_quantizer(quantizer):
    return TorchWeightFakeQuantize(quantizer)


def torch_activation_quantizer(quantizer):
    return TorchActivationFakeQuantize(quantizer)


def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def quantize_known_modules(mod_list, params_dict):
    types = tuple(type(m) for m in mod_list)
    if len(types)==1:
        types = types[0]
    quantize_method = OP_LIST_TO_QUANTIZE_METHOD.get(types, None)
    if quantize_method is None:
        raise NotImplementedError("Cannot quantize modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = quantize_method(*mod_list, params_dict)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod

def _quantize_modules(model, modules_to_quantize, params_dict, replacer_func=quantize_known_modules):
    mod_list = []
    for item in modules_to_quantize:
        mod_list.append(_get_module(model, item))

    # Quantize list of modules
    new_mod_list = replacer_func(mod_list, params_dict)

    # Replace original module list with quantize module list
    for i, item in enumerate(modules_to_quantize):
        _set_module(model, item, new_mod_list[i])

def quantize_modules(model,
                     modules_to_quantize,
                     params_dict={},
                     inplace=True,
                     replacer_func=quantize_known_modules):
    if not inplace:
        model = copy.deepcopy(model)
    # Handle case of modules_to_fuse being a list
    _quantize_modules(model, modules_to_quantize, params_dict, replacer_func)
    return model

def replace_encoderq(model, modules_to_replace, params_dict):
    mod_list = []
    for item in modules_to_replace:
        mod_list.append(_get_module(model, item))
    # Replace list of modules
    new_mod = quant_encoderq(mod_list, params_dict)
    # Replace original module list with quantize module list
    _set_module(model, modules_to_replace[0], new_mod)
    for i in range(1,len(modules_to_replace)):
        _set_module(model, modules_to_replace[i], nn.Identity())

def replace_decoderq(model, modules_to_replace, params_dict, module=nn.ConvTranspose1d):
    mod_list = []
    for item in modules_to_replace:
        mod_list.append(_get_module(model, item))
    # Replace list of modules
    if module==nn.ConvTranspose1d:
        new_mod = quant_convtranspose_decoderq(mod_list, params_dict)
    elif module==nn.Linear:
        new_mod = quant_linear_decoderq(mod_list, params_dict)
    else:
        assert False, "No support!"
    # Replace original module list with quantize module list
    _set_module(model, modules_to_replace[0], new_mod)
    for i in range(1,len(modules_to_replace)):
        _set_module(model, modules_to_replace[i], nn.Identity())

def replace_weight_quantizer(model, module_to_replace, module):
    # Create new module
    new_module = torch_weight_quantizer(module)
    # Replace original module list with new module
    _set_module(model, module_to_replace, new_module)


def replace_activation_quantizer(model, module_to_replace, module):
    # Create new module
    new_module = torch_activation_quantizer(module)
    # Replace original module list with new module
    _set_module(model, module_to_replace, new_module)


OP_LIST_TO_QUANTIZE_METHOD = {
    (nn.Conv1d, nn.PReLU): quant_conv1d_nl,
    (nn.Conv1d, nn.ReLU): quant_conv1d_nl,
    (nn.Conv1d, nn.Tanh): quant_conv1d_nl,
    (nn.Conv1d, nn.Sigmoid): quant_conv1d_nl,
    (nn.Conv1d): quant_conv1d,
    (nn.Conv2d): quant_conv2d,
    (nn.ConvTranspose1d): quant_convtr1d,
    (nn.GroupNorm): quant_groupnorm,
    (nn.LayerNorm): quant_layernorm,
    (nn.PReLU): quant_nl,
    (nn.ReLU): quant_nl,
    (nn.Sigmoid): quant_nl,
    (nn.Tanh): quant_nl,
    (nn.LSTM): quant_lstm,
    (nn.MultiheadAttention): quant_mha,
    (nn.Linear): quant_linear,
    (Add): quant_add,
    (Sub): quant_sub,
    (Mul): quant_mul,
    (Div): quant_div,
}
