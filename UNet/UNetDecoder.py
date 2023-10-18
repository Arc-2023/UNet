import torch
from torch import nn
from MyTrU.UNet.ConvEncoderBlock import ConvEncoderBlock
from MyTrU.UNet.ConvDecoderBlock import *
from MyTrU.Transformer.VIT import *


class UNetdecoder(nn.Module):
    def __init__(self, final_channels, class_num):
        super(UNetdecoder, self).__init__()
        self.conv1 = ConvDecoderBlock(in_channels=final_channels * 16, out_channels=final_channels * 4)
        self.conv2 = ConvDecoderBlock(in_channels=final_channels * 8, out_channels=final_channels * 2)
        self.conv3 = ConvDecoderBlock(in_channels=final_channels * 4, out_channels=final_channels)

        self.conv4 = nn.Conv2d(in_channels=final_channels, out_channels=class_num,
                               kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x4, x3, x2, x1):
        x = self.conv1(x4, x3)
        x = self.conv2(x, x2)
        x = self.conv3(x, x1)
        x = self.conv4(x)
        x = self.up(x)
        return x
