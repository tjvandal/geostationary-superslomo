# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down0 = down(32, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.up0 = up(256, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)

        #self.down3 = down(256, 512)
        #self.down4 = down(512, 512)
        #self.up0 = up(1024, 512)
        #self.up1 = up(1024, 256)
        #self.up2 = up(512, 128)
        #self.up3 = up(256, 64)
        #self.up4 = up(128, 32)
        #self.up5 = up(64, 16)
        self.outc = outconv(16, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        x = self.outc(x)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        #x = self.up4(x, x1)
        #x = self.up5(x, x0)
        #x = self.outc(x)
        return x
