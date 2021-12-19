import torch
import os
import librosa

import config
from model.generator import Generator
from utils.utils import log_audio, load_waveform
from dataloader.melspec import get_featurizer


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    model = Generator(config.num_mels).to(device)
    model.load_state_dict(torch.load("final_generator", map_location=device))

    files = os.listdir(config.test_wavs_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    model.eval()
    model.remove_weight_norm()
    featurizer = get_featurizer()
    with torch.no_grad():
        for i, filename in enumerate(files):
            y, s = librosa.load(os.path.join(config.test_wavs_dir, filename), sr=config.sr)

            wav = torch.Tensor(y).unsqueeze(0)
            x = featurizer(wav)
            predicted = model(x)
            log_audio(predicted, filename[:-4] + '_gen.wav')
