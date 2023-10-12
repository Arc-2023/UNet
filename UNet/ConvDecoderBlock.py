import torch
from torch import nn


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecoderBlock, self).__init__()
        self.up = nn.Upsample(mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, jump_img=None):
        x = self.up(x)
        if jump_img:
            x = torch.cat([x, jump_img], dim=1)
        x = self.conv(x)
        return x
