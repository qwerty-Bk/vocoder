import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import config


class SubMSD(nn.Module):
    def __init__(self, spectral):
        super().__init__()
        self.blocks_n = len(config.msd_kernels)
        if spectral:
            norm = spectral_norm
        else:
            norm = weight_norm
        self.convs = nn.ModuleList()
        for i in range(self.blocks_n):
            self.convs.append(
                nn.Sequential(
                    norm(nn.Conv1d(config.msd_in_channels[i], config.msd_out_channels[i], config.msd_kernels[i],
                                   config.msd_strides[i], config.msd_kernels[i] // 2, 1, config.msd_groups[i])),
                    nn.LeakyReLU(config.leaky_relu)
                )
            )

        self.out_conv = norm(nn.Conv1d(config.msd_out_channels[-1], 1, config.msd_out_kernel,
                                       padding=config.msd_out_kernel))

    def forward(self, input):
        output = input
        features = []
        for i in range(self.blocks_n):
            output = self.convs[i](output)
            features.append(output)
        output = self.out_conv(output)
        features.append(output)
        output = torch.flatten(output, 1, -1)

        return output, features


class MSDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList()
        self.discs.append(SubMSD(True))
        for i in range(2):
            self.discs.append(
                nn.Sequential(
                    nn.AvgPool1d(config.msd_pool_kernel, config.msd_pool_stride,
                                 padding=config.msd_pool_kernel // 2),
                    SubMSD(False),
                )
            )

    def forward(self, real_input, gen_input):
        real_outputs, real_features = [], []
        gen_outputs, gen_features = [], []
        for i in range(3):
            real_output, real_feature = self.discs[i](real_input)
            gen_output, gen_feature = self.discs[i](gen_input)
            real_outputs.append(real_output)
            gen_outputs.append(gen_output)
            real_features.append(real_feature)
            gen_features.append(gen_feature)

        return real_outputs, gen_outputs, real_features, gen_features
