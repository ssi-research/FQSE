import os
from utils import save_audio, plot_waveform, read_audio
from process import model_infer, normalize_audio
import argparse
import torch
import hyperpyyaml
from utils import get_device
from quantization.models.load_model import load_model
DEVICE = get_device()

def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--audio_path', '-a', type=str, required=True, help='Input audio path to separate')
    parser.add_argument('--use_cpu', action="store_true", help='Use cpu')
    parser.add_argument('--normalize', action="store_true", help='normalize input/output begore inference')
    parser.add_argument('--plot', action="store_true", help='Plot waveform figure')
    args = parser.parse_args()
    return args


def infer():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else 'cuda'
    # Read yml
    with open(args.yml_path) as f:
        conf = hyperpyyaml.load_hyperpyyaml(f)
    work_dir = conf['work_dir']

    # ------------------------------------
    # Load model
    # ------------------------------------
    model_cfg = conf['model']
    model = load_model(model_cfg)
    model.to(device)
    model.eval()

    # ------------------------------------
    # Run infer
    # ------------------------------------
    dataset_cfg, testing_cfg = conf['dataset'], conf['testing']

    # Read noisy and clean audios
    wav, fs = read_audio(args.audio_path, resample=dataset_cfg.get('resample',1))

    # normalize
    ref_mean, ref_std = 0, 1
    if args.normalize:
        ref_mean, ref_std = wav.mean(), wav.std()
        wav = (wav - ref_mean) / ref_std

    # Run model
    sep_wav = model_infer(model,
                           wav,
                           segment=testing_cfg.get('segment', None),
                           overlap=testing_cfg.get('overlap', 0.25),
                           device=device)

    # denormalize
    if args.normalize:
        sep_wav = sep_wav * ref_std + ref_mean

    # ------------------------- #
    # Save audios
    # ------------------------- #
    for src in range(model_cfg['n_src']):
        sep_wav_ch = sep_wav[src, ...]
        sep_wav_ch = normalize_audio(sep_wav_ch)
        save_audio(os.path.join(work_dir,"output"+str(src)+".wav"), sep_wav_ch, sample_rate=fs)
        print("output" + str(src) + ".wav has been saved!")

    if args.plot:
        fig = plot_waveform(sep_wav[0], sample_rate=fs)
        fig.savefig(os.path.join(work_dir,model_cfg["name"]+".png"))

if __name__ == '__main__':
    infer()















