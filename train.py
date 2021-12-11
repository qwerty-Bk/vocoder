import numpy as np
import torch
import wandb
import os
from scipy.io import wavfile
from random import randint
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import itertools
from torch.utils.data import DataLoader

from model.generator import Generator
from model.mp_disc import MPDiscriminator
from model.ms_disc import MSDiscriminator
from loss.loss import GeneratorLoss, DiscriminatorLoss, FeatureLoss, MelLoss
from utils.utils import count_parameters, load_waveform, log_audio
import config
from dataloader.meldataset import MelDataset, get_dataset_filelist, mel_spectrogram, MAX_WAV_VALUE


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def log_audio(wav, prefix):
    tmp_path = "tmp.wav"
    wavfile.write(tmp_path, config.sr, wav.cpu().detach().numpy())
    wandb.log({prefix + "audio": wandb.Audio(tmp_path, sample_rate=config.sr)})
    os.remove(tmp_path)


def validation_log(vocoder, dataset, mode):
    vocoder.eval()
    with torch.no_grad():
        wav_i = randint(0, len(dataset) - 1)
        _, _, filename, _ = dataset[wav_i]
        wav = load_waveform(filename)
        x = mel_spectrogram(wav.unsqueeze(0))
        predicted = vocoder(x)
        log_audio(predicted, mode)
    vocoder.train()


if __name__ == '__main__':

    # mel, waveform, fine_name, mel_loss = trainset[0]
    generator = Generator(80).to(device)
    ms_disc = MSDiscriminator().to(device)
    mp_disc = MPDiscriminator().to(device)

    gen_table, gen_count = count_parameters(generator)
    mpd_table, mpd_count = count_parameters(mp_disc)
    msd_table, msd_count = count_parameters(ms_disc)
    print("Generator:", gen_count)
    print("MSDiscrim:", msd_count)
    print("MPDiscrim:", mpd_count)

    # featurizer = get_featurizer().to(device)

    training_filelist, validation_filelist = get_dataset_filelist()

    train_dataset = MelDataset(training_filelist, n_cache_reuse=0, shuffle=True,
                               device=device, base_mels_path=config.input_mels_dir)  # fine-tuning
    valid_dataset = MelDataset(validation_filelist, n_cache_reuse=0, shuffle=True,
                               device=device, base_mels_path=config.input_mels_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, pin_memory=True)

    gen_loss_fn = GeneratorLoss()
    disc_loss_fn = DiscriminatorLoss()
    feat_loss_fn = FeatureLoss()
    mel_loss_fn = MelLoss()

    # val_dataloader = get_dataloader(batch_size=1, mode='valid')

    gen_opt = AdamW(generator.parameters(),
                    config.lr,
                    betas=(0.8, 0.99),
                    eps=1e-9)
    disc_opt = AdamW(itertools.chain(ms_disc.parameters(), mp_disc.parameters()),
                     config.lr,
                     betas=(0.8, 0.99),
                     eps=1e-9)

    gen_sched = ExponentialLR(gen_opt, gamma=config.lr_decay, last_epoch=config.last_epoch)
    disc_sched = ExponentialLR(disc_opt, gamma=config.lr_decay, last_epoch=config.last_epoch)

    generator.train()
    ms_disc.train()
    mp_disc.train()

    wandb.init(project='dla4')

    best_loss = np.inf

    for epoch in range(config.epochs):
        msd_gen_running_loss, mpd_gen_running_loss = 0, 0
        msd_feature_running_loss, mpd_feature_running_loss = 0, 0
        mpd_running_loss, msd_running_loss = 0, 0
        mel_running_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader)):
            gen_opt.zero_grad()
            disc_opt.zero_grad()

            # batch.to(device)
            # mels = featurizer(batch.waveform)
            mels, waveform, _, _ = batch
            waveform = waveform.to(device).unsqueeze(1)
            mels = mels.to(device)

            # print('mels', mels.shape)
            # print('batch.waveform', waveform.shape)
            # print('_mels', _mels.shape)

            gen_output = generator(mels)
            # print('gen_output', gen_output.shape)
            gen_mel = mel_spectrogram(gen_output.squeeze(1))
            # print('gen_mel', gen_mel.shape)

            # Discriminators
            disc_opt.zero_grad()

            # MPD
            mpd_real_output, mpd_pred_output, _, _ = mp_disc(waveform, gen_output.detach())
            mpd_loss, mpd_loss_real, mpd_loss_pred = disc_loss_fn(mpd_real_output, mpd_pred_output)

            # MSD
            msd_real_output, msd_pred_output, _, _ = ms_disc(waveform, gen_output.detach())
            msd_loss, msd_loss_real, msd_loss_pred = disc_loss_fn(msd_real_output, msd_pred_output)

            disc_loss = mpd_loss + msd_loss
            disc_loss.backward()
            disc_opt.step()

            # Generator
            gen_opt.zero_grad()

            mel_loss = mel_loss_fn(mels, gen_mel)

            _, mpd_pred_output, mpd_real_features, mpd_pred_features = mp_disc(waveform, gen_output)
            _, msd_pred_output, msd_real_features, msd_pred_features = ms_disc(waveform, gen_output)

            mpd_feature_loss = feat_loss_fn(mpd_real_features, mpd_pred_features)
            msd_feature_loss = feat_loss_fn(msd_real_features, msd_pred_features)
            mpd_gen_loss, mpd_gen_losses = gen_loss_fn(mpd_pred_output)
            msd_gen_loss, msd_gen_losses = gen_loss_fn(msd_pred_output)

            loss = msd_gen_loss + mpd_gen_loss + config.lambda_feat * (msd_feature_loss + mpd_feature_loss) + \
                                                                                        config.lambda_mel * mel_loss
            loss.backward()

            gen_opt.step()

            # Clipping Gradient Norms
            clip_grad_norm_(generator.parameters(), config.clip_grad)
            clip_grad_norm_(ms_disc.parameters(), config.clip_grad)
            clip_grad_norm_(mp_disc.parameters(), config.clip_grad)

            msd_gen_running_loss += msd_gen_loss.item()
            mpd_gen_running_loss += mpd_gen_loss.item()
            msd_feature_running_loss += msd_feature_loss.item()
            mpd_feature_running_loss += mpd_feature_loss.item()
            mel_running_loss += mel_loss.item()
            mpd_running_loss += mpd_loss.item()
            msd_running_loss += msd_loss.item()

            if (i + 1) % config.log_every == 0:
                msd_gen_running_loss /= config.log_every
                mpd_gen_running_loss /= config.log_every
                msd_feature_running_loss /= config.log_every
                mpd_feature_running_loss /= config.log_every
                mel_running_loss /= config.log_every
                mpd_running_loss /= config.log_every
                msd_running_loss /= config.log_every
                wandb.log({'msd_gen_loss': msd_gen_running_loss, 'mpd_gen_loss': mpd_gen_running_loss,
                           'msd_feature_loss': msd_feature_running_loss, 'mpd_feature_loss': mpd_feature_running_loss,
                           'mel_loss': mel_running_loss, 'mpd_loss': mpd_running_loss, 'msd_loss': msd_running_loss})

                msd_gen_running_loss = 0
                mpd_gen_running_loss = 0
                msd_feature_running_loss = 0
                mpd_feature_running_loss = 0
                mel_running_loss = 0
                mpd_running_loss = 0
                msd_running_loss = 0

                validation_log(generator, train_dataset, 'train')
                validation_log(generator, valid_dataset, 'valid')

        disc_sched.step()
        gen_sched.step()
        wandb.log({'disc_lr': disc_sched.get_last_lr()[0], 'gen_lr': gen_sched.get_last_lr()[0]})
        wandb.log({"epoch": epoch})
        if (epoch + 1) % config.save_model == 0:
            torch.save(generator.state_dict(), "last_generator")
