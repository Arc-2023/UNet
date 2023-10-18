import torch
from torch import nn
from MyTrU.UNet.ConvEncoderBlock import ConvEncoderBlock
from MyTrU.UNet.ConvDecoderBlock import *
from MyTrU.Transformer.VIT import *
from rich.console import Console

console = Console()


class Unetencoder(nn.Module):
    def __init__(self, img_h, img_w, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout, head_num):
        super(Unetencoder, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.patch_d = patch_d

        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels, final_out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(final_out_channels),
            nn.GELU()
        )

        self.enc_conv1 = ConvEncoderBlock(final_out_channels, final_out_channels * 2)
        self.enc_conv2 = ConvEncoderBlock(final_out_channels * 2, final_out_channels * 4)
        self.enc_conv3 = ConvEncoderBlock(final_out_channels * 4, final_out_channels * 8)

        self.vit = VIT(block_size, mlp_d, head_num, final_out_channels * 8, dropout, final_out_channels * 8, img_w // 8,
                       img_h // 8, patch_d)
        self.enc_conv_out = nn.Sequential(
            nn.Conv2d(final_out_channels * 8, final_out_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_out_channels * 8)
        )

    def forward(self, x):
        # b c h w
        x = self.enc_conv0(x)
        # b o_c h w
        x1 = self.enc_conv1(x)
        # b o_c*2 h/2 w/2
        x2 = self.enc_conv2(x1)
        # b o_c*4 h/4 w/4
        x3 = self.enc_conv3(x2)
        # b o_c*8 h/8 w/8
        x4 = self.vit(x3)
        # b o_c*8 h/16 w/16
        x4 = rearrange(x4, 'b (x y) c->b c x y', x=(self.img_w // 8) // self.patch_d)
        x4 = self.enc_conv_out(x4)
        return x4, x3, x2, x1
