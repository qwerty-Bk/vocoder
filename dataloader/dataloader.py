from dataloader.collator import LJSpeechCollator
from dataloader.ljspeech import LJSpeechDataset
from torch.utils.data import DataLoader


def get_dataloader(dataset=LJSpeechDataset, path='.', batch_size=64, collate_fn=LJSpeechCollator,
                   limit=-1, mode='train'):
    ds = dataset(path)
    if limit != -1:
        ds = list(ds)[:limit * batch_size]
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn())
