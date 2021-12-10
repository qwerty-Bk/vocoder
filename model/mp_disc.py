import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import config
from utils.utils import get_padding


class SubMPD(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.convs = nn.ModuleList()
        for i in range(config.mpd_blocks):
            self.convs.append(
                nn.Sequential(
                    weight_norm(nn.Conv2d(
                        config.mpd_in_channels[i], config.mpd_out_channels[i], config.mpd_kernel, config.mpd_stride[i],
                        padding=(get_padding(config.mpd_kernel[0], 1), 0)
                    )),
                    nn.LeakyReLU(config.leaky_relu)
                )
            )

        self.convs.append(weight_norm(nn.Conv2d(config.mpd_out_channels[-1], 1, config.mpd_out_kernel)))

    def forward(self, input):
        pad = self.p - (input.shape[-1] % self.p)
        output = F.pad(input, (0, pad), "reflect")
        output = output.view(input.shape[0], input.shape[1], -1, self.p)

        features = []
        for module in self.convs:
            output = module(output)
            features.append(output)
        return torch.flatten(features[-1], 1, -1), features


class MPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList()
        self.blocks_n = len(config.p)
        for i, p in enumerate(config.p):
            self.discs.append(SubMPD(p))

    def forward(self, real_input, gen_input):
        real_outputs, real_features = [], []
        gen_outputs, gen_features = [], []
        for i in range(self.blocks_n):
            real_output, real_feature = self.discs[i](real_input)
            gen_output, gen_feature = self.discs[i](gen_input)
            real_outputs.append(real_output)
            gen_outputs.append(gen_output)
            real_features.append(real_feature)
            gen_features.append(gen_feature)

        return real_outputs, gen_outputs, real_features, gen_features
