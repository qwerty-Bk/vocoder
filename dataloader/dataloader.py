from dataloader.collator import LJSpeechCollator
from dataloader.ljspeech import LJSpeechDataset
from torch.utils.data import DataLoader
import config

import os
from pathlib import Path


def get_dataset_filelist():
    with open(config.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(config.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(config.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(config.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    # valid_path = 'LJSpeech-1.1/valid'
    # if not Path(valid_path).exists():
    #     os.makedirs(valid_path)
    #     for file in validation_files:
    #         os.rename(file, valid_path + '/' + file[-14:])
    return training_files, validation_files


def get_dataloader(dataset=LJSpeechDataset, path='.', batch_size=64, collate_fn=LJSpeechCollator,
                   limit=-1):
    ds = dataset(path)
    if limit != -1:
        ds = list(ds)[:limit * batch_size]
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn())
