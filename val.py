import glob
from utils import read_audio, read_audios
import hyperpyyaml
from tqdm import tqdm
from process import model_infer, metric_evaluation
import argparse
import torch
import numpy as np
import os
from utils import get_device
from quantization.models.load_model import load_model
DEVICE = get_device()


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--use_cpu', action="store_true", help='Use cpu')
    args = parser.parse_args()
    return args


def read_librimix(folder, n_spks=1, noisy=False):
    assert 1<=n_spks<=3, "Error: Up to 3 sources to seperate!"
    if n_spks==1:
        mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_single', '*')))
        clean_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        assert len(mix_audio_files) == len(clean_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean_audio_files]
    elif n_spks==2:
        if not noisy:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_clean', '*')))
        else:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_both', '*')))
        clean1_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        clean2_audio_files = sorted(glob.glob(os.path.join(folder, 's2', '*')))
        assert len(mix_audio_files) == len(clean1_audio_files) == len(clean2_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean1_audio_files, clean2_audio_files]
    elif n_spks==3:
        if not noisy:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_clean', '*')))
        else:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_both', '*')))
        clean1_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        clean2_audio_files = sorted(glob.glob(os.path.join(folder, 's2', '*')))
        clean3_audio_files = sorted(glob.glob(os.path.join(folder, 's3', '*')))
        assert len(mix_audio_files) == len(clean1_audio_files) == len(clean2_audio_files) == len(clean3_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean1_audio_files, clean2_audio_files, clean3_audio_files]


def val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device):
    # ------------------------------------
    # Read dataset
    # ------------------------------------
    n_src = model_cfg.get('n_src', 1)
    mix_audio_files, clean_audio_files_list = read_librimix(testing_cfg['test_dir'], n_src, dataset_cfg['noisy'])
    dataset_size = len(mix_audio_files)

    # ------------------------------------
    # Run validation
    # ------------------------------------
    sisnrs, sdrs, stois = np.zeros(dataset_size), np.zeros(dataset_size), np.zeros(dataset_size)
    sisnrs_imp = np.zeros(dataset_size)
    torch.no_grad().__enter__()
    for i in tqdm(range(dataset_size)):
        # Read noisy and clean audios
        mix_wav, fs = read_audio(mix_audio_files[i], resample=dataset_cfg.get('resample',1))
        clean_wavs, _ = read_audios(clean_audio_files_list, i, resample=dataset_cfg.get('resample',1))
        # Run model
        wavs = model_infer(model,
                           mix_wav,
                           segment=testing_cfg.get('segment', None),
                           overlap=testing_cfg.get('overlap', 0.25),
                           n_splitter_bits=model_cfg.get('n_splitter_bits',8),
                           n_combiner_bits=model_cfg.get('n_combiner_bits',8),
                           device=device,
                           target=clean_wavs)
        # Metric evaluation
        sisnrs[i], sdrs[i], stois[i] = metric_evaluation(wavs, clean_wavs, sample_rate=fs)
        sisnr_bl, sdr_bl, stoi_bl = metric_evaluation(clean_wavs.squeeze(1), torch.stack([mix_wav]*n_src), sample_rate=fs)
        sisnrs_imp[i] = sisnrs[i] - sisnr_bl # SI-SNR improvement
        if i % 500 == 0 and i > 0:
            print("SI-SNR={:0.4f},SI-SNR-imp={:0.4f},SDR={:0.4f},STOI={:0.4f}".format(np.mean(sisnrs[:i]),np.mean(sisnrs_imp[:i]),np.mean(sdrs[:i]),np.mean(stois[:i])))

    # ------------------------- #
    # Result
    # ------------------------- #
    # Average by number of samples
    avg_sisnr, avg_sisnr_imp, avg_sdr, avg_stoi = np.mean(sisnrs), np.mean(sisnrs_imp), np.mean(sdrs), np.mean(stois)
    return avg_sisnr, avg_sisnr_imp, avg_sdr, avg_stoi


def val():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else 'cuda'
    # Read yml
    with open(args.yml_path) as f:
        conf = hyperpyyaml.load_hyperpyyaml(f)

    # ------------------------------------
    # Load model
    # ------------------------------------
    model_cfg = conf['model']
    model = load_model(model_cfg)
    model.to(device)
    model.eval()

    dataset_cfg, testing_cfg = conf['dataset'], conf['testing']
    if dataset_cfg['name'] == "librimix":
        sisnr, sisnr_imp, sdr, stoi = val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device)
        print("SI-SNR={:0.4f},SI-SNR-imp={:0.4f},SDR={:0.4f},STOI={:0.4f}".format(sisnr, sisnr_imp, sdr, stoi))
    else:
        assert False, "Dataset {} is not supported!".format(dataset_cfg['name'])


if __name__ == '__main__':
    val()