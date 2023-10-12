import torch
from torch import nn
from ConvEncoderBlock import ConvEncoderBlock
from ConvDecoderBlock import *
from MyTrU.Transformer.VIT import *


class UNetEncoder(nn.Module):
    def __init__(self, img_y, img_x, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout, head_num):
        super(UNetEncoder, self).__init__()
        self.img_x = img_x
        self.img_y = img_y
        self.patch_d = patch_d

        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels, final_out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(final_out_channels),
            nn.GELU()
        )

        self.enc_conv1 = ConvEncoderBlock(final_out_channels, final_out_channels * 2)
        self.enc_conv2 = ConvEncoderBlock(final_out_channels * 2, final_out_channels * 4)
        self.enc_conv3 = ConvEncoderBlock(final_out_channels * 2, final_out_channels * 8)

        self.vit = VIT(block_size, mlp_d, head_num, final_out_channels * 8, dropout, final_out_channels * 8, img_x,
                       img_y, patch_d)
        self.enc_conv_out = nn.Sequential(
            nn.Conv2d(final_out_channels * 8, final_out_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_out_channels * 4)
        )

    def forward(self, x):
        x1 = self.enc_conv0(x)
        x2 = self.enc_conv1(x1)
        x3 = self.enc_conv2(x2)

        x4 = self.enc_conv3(x3)
        x4 = self.vit(x4)
        x4 = rearrange(x4, 'b (x y) c->b c x y', x=self.img_x // self.patch_d)
        x4 = self.enc_conv_out(x4)
        return x4, x3, x2, x1
