import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_output):
        gen_losses = []
        for d in disc_output:
            gen_losses.append(
                torch.mean((1 - d) ** 2)
            )
        return sum(gen_losses), gen_losses


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_real_output, disc_gen_output):
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_output, disc_gen_output):
            r_losses.append(
                torch.mean((1 - dr) ** 2)
            )
            g_losses.append(
                torch.mean(dg ** 2)
            )
        return sum(r_losses) + sum(g_losses), r_losses, g_losses


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_features, gen_features):
        loss = 0
        for dr, dg in zip(real_features, gen_features):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, real_mel, gen_mel):
        return self.loss_fn(real_mel, gen_mel)


if __name__ == '__main__':
    pass
    # TODO: vectorize
