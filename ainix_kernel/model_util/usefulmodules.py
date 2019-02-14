import torch

from torch import nn


class Conv2dSame(nn.Module):
    # From code on https://github.com/pytorch/pytorch/issues/3867
    # and @kylemcdonald
    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 tokens_before_channels = False):
        super().__init__()
        assert kernel_size % 2 == 1
        pad_size = kernel_size // 2
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    bias=bias, padding=pad_size)
        self.tokens_before_channels = tokens_before_channels

    def get_weight(self):
        return self.conv_layer.weight

    def set_weight(self, value):
        self.conv_layer.weight = value

    def forward(self, x):
        if self.tokens_before_channels:
            x = x.transpose(1, 2)  # B x T x C -> B x C x T
        x = self.conv_layer(x)
        if self.tokens_before_channels:
            x = x.transpose(1, 2)  # B x C x T -> B x T x C
        return x
