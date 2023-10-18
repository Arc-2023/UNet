import torch
from torch import nn
from MyTrU.UNet.UnetEncoder import Unetencoder
from MyTrU.UNet.UNetDecoder import UNetdecoder
from torchinfo import summary


class TransUNet(nn.Module):
    def __init__(self, img_h, img_w, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout, head_num,
                 classes):
        super(TransUNet, self).__init__()
        self.encoder = Unetencoder(img_h, img_w, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout,
                                   head_num)
        self.decoder = UNetdecoder(final_out_channels, classes)

    def forward(self, x):
        # 1/16 1/8 1/4 1/2
        x4, x3, x2, x1 = self.encoder(x)
        # 1
        x = self.decoder(x4, x3, x2, x1)
        return x
