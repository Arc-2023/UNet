import torch
from torch import nn


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, base_width=64):
        super(ConvEncoderBlock, self).__init__()
        self.mid_channels = int(out_channels * (base_width / 64))
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, stride=1, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            # DownScale
            nn.Conv2d(self.mid_channels, self.mid_channels, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.Conv2d(self.mid_channels, out_channels, stride=1, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        _x = self.down_layer(x)
        x = self.conv(x)
        # print(x.shape)
        # print(_x.shape)
        return _x + x
