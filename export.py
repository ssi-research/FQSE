import hyperpyyaml
import argparse
import torch
import os
from quantization.models.load_model import load_model
from utils import set_seed
from process import preprocess
from quantization.qat.qat_quant import GradientActivationFakeQuantize, GradientWeightFakeQuantize
from quantization.qat.qat_utils import replace_weight_quantizer, replace_activation_quantizer


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--torchscript', action="store_true", help='Export to TorchScript')
    parser.add_argument('--onnx', action="store_true", help='Export to ONNX')
    args = parser.parse_args()
    return args


def replace_quantizers(model):
    with torch.no_grad():
        for m_name,m in model.named_modules():
            if isinstance(m, GradientWeightFakeQuantize):
                replace_weight_quantizer(model, m_name, m)
            elif isinstance(m, GradientActivationFakeQuantize):
                replace_activation_quantizer(model, m_name, m)


def export():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    # Read yml
    with open(args.yml_path) as f:
        conf = hyperpyyaml.load_hyperpyyaml(f)

    # ------------------------------------
    # Load model
    # ------------------------------------
    model_cfg = conf['model']
    model = load_model(model_cfg)
    model.to("cpu")
    model.eval()

    dataset_cfg, testing_cfg = conf['dataset'], conf['testing']
    frame_length = dataset_cfg["segment"]*dataset_cfg["sample_rate"]*dataset_cfg["resample"]
    work_dir = conf["work_dir"]

    # ------------------------------------
    # Replace quantizers
    # ------------------------------------
    set_seed(0)
    dummy_input = torch.randn(1, model_cfg["n_src"], frame_length)
    dummy_input = preprocess(dummy_input, n_splitter=model_cfg["n_splitter"])
    replace_quantizers(model) # Replace our quantizers with torch quantizers

    # ------------------------------------
    # Export
    # ------------------------------------
    if args.torchscript:
        torch_traced = torch.jit.trace(model, dummy_input)
        torch_script_model = torch.jit.script(torch_traced)
        torch.jit.save(torch_script_model, os.path.join(work_dir, "model_torchscript.pth"))
        print("Pytorch torch script model has been saved!")
    if args.onnx:
        torch.onnx.export(model, dummy_input, os.path.join(work_dir, "model.onnx"), opset_version=16,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print("ONNX Model has been saved!")


if __name__ == '__main__':
    export()















