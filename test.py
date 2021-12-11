import torch
import os

import config
from model.generator import Generator
from dataloader.meldataset import mel_spectrogram
from utils.utils import load_waveform, log_audio


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    model = Generator(config.num_mels).to(device)
    model.load_state_dict(torch.load("generator_999", map_location=device))

    files = os.listdir(config.input_wavs_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    model.eval()
    model.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(files):
            if i == 9:
                break
            wav = load_waveform(os.path.join(config.input_wavs_dir, filename))
            x = mel_spectrogram(wav.unsqueeze(0))
            predicted = model(x)
            log_audio(predicted, filename[:-4] + '_gen.wav')
