from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
import config
from utils.utils import get_padding


class ResnetBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_sizes=config.kr, dilations=config.dr):
        super().__init__()
        self.convs = nn.ModuleList()
        self.block_n = len(dilations)
        for i in range(self.block_n):
            conv_i = []
            for j in range(len(dilations[i])):
                conv_i.append(nn.LeakyReLU(config.leaky_relu))
                conv_i.append(weight_norm(nn.Conv1d(
                    hidden_channels, hidden_channels, kernel_sizes[i], dilation=dilations[i][j],
                    padding=get_padding(kernel_sizes[i], dilations[i][j])
                )))
            self.convs.append(nn.Sequential(*conv_i))

    def remove_weight_norm(self):
        for i in range(self.block_n):
            remove_weight_norm(self.convs[i][-1])

    def forward(self, input):
        output = input
        # print('\t\tresnet, input', output.shape)
        for i in range(self.block_n):
            output = output + self.convs[i](output)
            # print('\t\tresnet, block', i + 1, output.shape)
        return output


class MRFBlock(nn.Module):
    def __init__(self, channels, block_n=3):
        super().__init__()
        self.res = nn.ModuleList()
        self.block_n = block_n
        for i in range(block_n):
            self.res.append(ResnetBlock(channels))

    def remove_weight_norm(self):
        for i in range(self.block_n):
            self.res[i].remove_weight_norm()

    def forward(self, input):
        # print('\tmrf, input', input.shape)
        output = self.res[0](input)
        # print('\tmrf, block 1', output.shape)
        for i in range(1, self.block_n):
            output = output + self.res[i](input)
            # print('\tmrf, block', i + 1, output.shape)
        output /= self.block_n
        return output


class Generator(nn.Module):
    def __init__(self, in_channels, kernel_sizes=config.ku, strides=config.su):
        super().__init__()
        self.in_conv = weight_norm(nn.Conv1d(in_channels, config.hu, config.in_conv_kernel,
                                             padding=get_padding(config.in_conv_kernel, 1)))

        blocks = []
        self.block_n = len(kernel_sizes)
        for i in range(len(kernel_sizes)):
            blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(config.leaky_relu),
                    nn.utils.weight_norm(
                        nn.ConvTranspose1d(config.hu // (2 ** i), config.hu // (2 ** (i + 1)),
                                           kernel_sizes[i], strides[i], strides[i] // 2)
                    ),
                    MRFBlock(config.hu // (2 ** (i + 1)))
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Sequential(
            nn.LeakyReLU(config.leaky_relu),
            weight_norm(nn.Conv1d(config.hu // (2 ** len(kernel_sizes)), 1, config.out_conv_kernel,
                                  padding=get_padding(config.out_conv_kernel, 1))),
            nn.Tanh()
        )

    def remove_weight_norm(self):
        remove_weight_norm(self.in_conv)
        for i in range(self.block_n):
            remove_weight_norm(self.blocks[i][1])
            self.blocks[i][-1].remove_weight_norm()
        remove_weight_norm(self.out_conv[1])

    def forward(self, input):
        # print('generator, input', input.shape)
        output = self.in_conv(input)
        # print('generator, in_conv', output.shape)
        output = self.blocks(output)
        # print('generator, blocks', output.shape)
        output = self.out_conv(output)
        # print('generator, out_conv', output.shape)

        return output
