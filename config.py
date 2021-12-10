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
msd_in_channels = [1, 128, 128, 256, 512, 1024, 1024]  # тут я не уловила че надо поэтому стоят числа из офиц репо
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
input_training_file = 'LJSpeech-1.1/one_batch.txt'  # LJSpeech-1.1/training.txt
input_validation_file = 'LJSpeech-1.1/validation.txt'
input_wavs_dir = 'LJSpeech-1.1/wavs'
output_dir = 'predicted'

# training
batch_size = 8
epochs = 10
lr = 2e-4
limit = 1
lambda_mel = 45
lambda_feat = 2
clip_grad = 10
log_every = 1
lr_decay = 0.999
last_epoch = -1  # дообучение
save_model = 10  # save dict every save_model epochs
