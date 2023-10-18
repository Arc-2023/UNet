import torch
from torch import nn
from rich.console import Console

console = Console()


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, jump_img):
        x = self.up(x)
        x = torch.cat([x, jump_img], dim=1)
        # console.log(x.shape)
        x = self.conv(x)
        return x
