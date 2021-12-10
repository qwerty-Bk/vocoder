import config
from dataloader.dataloader import get_dataloader
from model.generator import Generator
from dataloader.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from train import validation

import torch
from torch import nn
from scipy.io import wavfile
from torch.utils.data import DataLoader
import os
from scipy.io.wavfile import write

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def log_audio(wav, prefix):
    tmp_path = prefix + ".wav"
    print(wav.shape)
    wavfile.write(tmp_path, 22050, wav.cpu().detach().numpy())


if __name__ == '__main__':
    model = Generator(config.num_mels).to(device)
    model.load_state_dict(torch.load("generator_9", map_location=device))

    files = os.listdir(config.input_wavs_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    model.eval()
    model.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(files):
            wav, sr = load_wav(os.path.join(config.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = mel_spectrogram(wav.unsqueeze(0))
            y_g_hat = model(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(config.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, config.sr, audio)
            print(output_file)
