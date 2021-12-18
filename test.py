import torch
import os

import config
from model.generator import Generator
from utils.utils import log_audio


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    model = Generator(config.num_mels).to(device)
    model.load_state_dict(torch.load("final_generator", map_location=device))

    files = os.listdir(config.test_mels_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    model.eval()
    model.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(files):
            x = torch.load(os.path.join(config.test_mels_dir, filename), map_location=device)
            predicted = model(x)
            log_audio(predicted, filename[:-7] + '_gen.wav')
