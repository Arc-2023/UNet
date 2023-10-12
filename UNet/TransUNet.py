import torch
from torch import nn
from UNetEncoder import UNetEncoder
from UNetDecoder import UNetDecoder


class TransUNet(nn.Module):
    def __init__(self, img_y, img_x, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout, head_num,
                 classes):
        super(TransUNet, self).__init__()
        self.encoder = UNetEncoder(img_y, img_x, patch_d, in_channels, final_out_channels, block_size, mlp_d, dropout,
                                   head_num)
        self.decoder = UNetDecoder(final_out_channels, classes)

    def forward(self, x):
        # 1/16 1/8 1/4 1/2
        x4, x3, x2, x1 = self.encoder(x)
        # 1
        x = self.decoder(x4, x3, x2, x1)
        return x
