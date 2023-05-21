import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_log(folder, results_dict, writeline=False):
    with open(os.path.join(folder,"results.txt"), 'a') as f:
        for key in results_dict.keys():
            f.write(key+': '+str(results_dict[key])+', ')
        if writeline:
            f.writelines("\n")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_audio(audio_path, resample=1):
    waveform, fs = torchaudio.load(audio_path)
    if resample!=1:
        waveform = torchaudio.transforms.Resample(fs, fs*resample)(waveform)
        fs = int(fs*resample)
    return waveform, fs

def read_audios(audio_paths_list, index, resample=1):
    waveforms = []
    for audio_paths in audio_paths_list:
        waveform, fs = read_audio(audio_paths[index])
        if resample != 1:
            waveform = torchaudio.transforms.Resample(fs, fs * resample)(waveform)
            fs = int(fs * resample)
        waveforms.append(waveform)
    return torch.stack(waveforms), fs

def save_audio(path, waveform, sample_rate, bits_per_sample=16):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate=sample_rate, bits_per_sample=bits_per_sample)

def plot_waveform(waveform, sample_rate, title="waveform"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        axes[c].set_xlabel('Time[sec]')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    return figure

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        axes[c].grid(True)
        axes[c].set_xlabel('Time[sec]')
        axes[c].set_ylabel('Freq[Hz]')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    return figure

def plot_frequency(waveform, sample_rate, title="Frequency"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].psd(waveform[c], Fs=sample_rate)
        axes[c].grid(True)
        axes[c].set_xlabel('Freq[Hz]')
        axes[c].set_ylabel('Amp')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    return figure
