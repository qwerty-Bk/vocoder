from prettytable import PrettyTable
import torch
from scipy.io import wavfile

import config
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def load_waveform(filename):
    sr, wav = read(filename)
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    return wav


def log_audio(wav, prefix):
    wav = wav.squeeze()
    wav = wav * MAX_WAV_VALUE
    wav = wav.cpu().detach().numpy().astype('int16')
    tmp_path = config.output_dir + '/' + prefix + ".wav"
    wavfile.write(tmp_path, 22050, wav)
