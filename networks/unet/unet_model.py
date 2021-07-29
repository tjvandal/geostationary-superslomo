# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNetSmall(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNetSmall, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down0 = down(32, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
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
        return x

class UNetMedium(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNetMedium, self).__init__()
        self.inc = inconv(n_channels, 64, kernel_size=7)
        self.down0 = down(64, 128, kernel_size=5)
        self.down1 = down(128, 256, kernel_size=5)
        self.down2 = down(256, 512)
        self.down3 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.outc = outconv(32, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.outc(x)
        return x

class UNetMultiscale(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNetMultiscale, self).__init__()
        self.inc = inconv(n_channels, 64, kernel_size=[3,5,7])
        self.down0 = down(64, 128, kernel_size=[3,5,])
        self.down1 = down(128, 256, kernel_size=[3,5,])
        self.down2 = down(256, 512, kernel_size=3)
        self.down3 = down(512, 512, kernel_size=3)
        self.up1 = up(1024, 256, kernel_size=3)
        self.up2 = up(512, 128, kernel_size=3)
        self.up3 = up(256, 64, kernel_size=3)
        self.up4 = up(128, 32, kernel_size=3)
        self.outc = outconv(32, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.outc(x)
        return x

class UNetMultiscaleV2(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNetMultiscaleV2, self).__init__()
        self.inc = inconv(n_channels, 64*4, kernel_size=[3,5,7,9])
        self.down0 = down(64*4, 128*3, kernel_size=[3,5,7,])
        self.down1 = down(128*3, 256*2, kernel_size=[3,5,])
        self.down2 = down(256*2, 512*2, kernel_size=[3,5,])
        self.down3 = down(512*2, 512, kernel_size=[3,])
        self.up1 = up(512+512*2, 256, kernel_size=3)
        self.up2 = up(256 + 256*2, 128, kernel_size=3)
        self.up3 = up(128 + 128*3, 64, kernel_size=3)
        self.up4 = up(64 + 64*4, 32, kernel_size=3)
        self.outc = outconv(32, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.outc(x)
        return x

