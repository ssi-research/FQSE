import torch
from quantization.models.convtasnetq import ConvTasNetQ
from quantization.qat.qat_quant import GradientActivationFakeQuantize, GradientWeightFakeQuantize

def enable_observer(model, mode=False):
    with torch.no_grad():
        for _, m in model.named_modules():
            if isinstance(m, GradientWeightFakeQuantize) or isinstance(m, GradientActivationFakeQuantize):
                m.enable_observer(mode)

def create_model(model_cfg):
    name = model_cfg['name']
    if name == "ConvTasNet":
        model = ConvTasNetQ(n_spks=model_cfg.get('n_src',1),
                            n_splitter=model_cfg.get('n_splitter',1),
                            n_combiner=model_cfg.get('n_combiner',1),
                            kernel_size=model_cfg.get('kernel_size',32),
                            stride=model_cfg.get('stride',16))
    else:
        assert False, "Model {} is not supported!".format(name)

    return model

def load_model(model_cfg):
    model = create_model(model_cfg)
    quant_cfg = model_cfg['quantization']
    if quant_cfg.get('qat', False):
        gradient_based = quant_cfg.get('gradient_based', True)
        weight_quant, weight_n_bits = quant_cfg.get('weight_quant', True), quant_cfg.get('weight_n_bits', 8)
        act_quant, act_n_bits = quant_cfg.get('act_quant', True), quant_cfg.get('act_n_bits', 8)
        in_quant, in_act_n_bits = quant_cfg.get('in_quant', False), quant_cfg.get('in_act_n_bits', 8)
        out_quant, out_act_n_bits = quant_cfg.get('out_quant', False), quant_cfg.get('out_act_n_bits', 8)
        model.quantize_model(gradient_based=gradient_based,
                             weight_quant=weight_quant,
                             weight_n_bits=weight_n_bits,
                             act_quant=act_quant,
                             act_n_bits=act_n_bits,
                             in_quant=in_quant,
                             in_act_n_bits=in_act_n_bits,
                             out_quant=out_quant,
                             out_act_n_bits=out_act_n_bits)
    # Model weights
    model_path = model_cfg.get('model_path',None)
    if model_path:
        model.load_pretrain(model_path)
    enable_observer(model, False)
    return model