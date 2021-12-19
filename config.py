# general
sr = 22050

# generator
hu = 512
in_conv_kernel = 7
out_conv_kernel = 7
kr = [3, 7, 11]
ku = [16, 16, 4, 4]
su = [8, 8, 2, 2]
dr = [[1, 1], [3, 1], [5, 1]]
leaky_relu = 0.1

# mpd
p = [2, 3, 5, 7, 11]
mpd_kernel = (5, 1)
mpd_stride = [(3, 1)] * 4 + [1]
mpd_out_kernel = (3, 1)
mpd_in_channels = [1, 32, 128, 512, 1024]
mpd_out_channels = mpd_in_channels[1:] + [1024]
mpd_blocks = 5

# msd
msd_in_channels = [1, 128, 128, 256, 512, 1024, 1024]  # didn't find it in the paper, so I took these numbers from the official repo
msd_out_channels = msd_in_channels[1:] + [1024]
msd_kernels = [15] + [41] * 5 + [5]
msd_strides = [1, 2, 2, 4, 4, 1, 1]
msd_groups = [1, 4] + [16] * 4 + [1]
msd_out_kernel = 3
msd_pool_kernel = 4
msd_pool_stride = 2

# dataset
segment_size = 8192
n_fft = 1024
num_mels = 80
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000
fmax_for_loss = None
input_mels_dir = 'ft_dataset'
input_training_file = 'LJSpeech-1.1/training.txt'  # LJSpeech-1.1/one_batch.txt
input_validation_file = 'LJSpeech-1.1/validation.txt'
test_wavs_dir = 'test'
test_mels_dir = 'test_mels'
input_wavs_dir = 'LJSpeech-1.1/wavs'
input_valid_dir = 'LJSpeech-1.1/valid'
output_dir = 'predicted'

# training
batch_size = 8
epochs = 313
lr = 2e-4
limit = -1
lambda_mel = 45
lambda_feat = 2
clip_grad = 10
log_every = 50
lr_decay = 0.999
last_epoch = -1  # continue training
save_model = 1  # save dict every save_model epochs
