# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(ConvLReLU, self).__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                                  nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DoubleConv, self).__init__()
        padding = (kernel_size - 1)/2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvMultiscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        '''
        kernel_size: list of ints
        '''
        super(ConvMultiscale, self).__init__()
        self.convs = []
        n_per = out_ch // len(kernel_size)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, k in enumerate(kernel_size):
            if (i+1) == len(kernel_size):
                out_ch_k  = out_ch - n_per*i
            else:
                out_ch_k = n_per
            #self.convs.append(ConvLReLU(in_ch, out_ch_k, kernel_size=k).to(device))
            self.convs.append(ConvLReLU(in_ch, out_ch_k, kernel_size=k))
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        y = []
        for conv in self.convs:
            y.append(conv(x))
        y = torch.cat(y, 1)
        return y

class DoubleConvMultiscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(DoubleConvMultiscale, self).__init__()
        self.conv = nn.Sequential(
            ConvMultiscale(in_ch, out_ch, kernel_size),
            ConvMultiscale(out_ch, out_ch, kernel_size)
            )

    def forward(self, x):
        return self.conv(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, conv=None):
        super(inconv, self).__init__()
        if conv:
            self.conv = conv
        elif isinstance(kernel_size, list):
            self.conv = DoubleConvMultiscale(in_ch, out_ch, kernel_size)
        else:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, conv=None):
        super(down, self).__init__()
        self.pool = nn.AvgPool2d(2)

        if conv is not None:
            self.conv = conv
        elif isinstance(kernel_size, list):
            self.conv = DoubleConvMultiscale(in_ch, out_ch, kernel_size=kernel_size)
        else:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bilinear=False):
        super(up, self).__init__()

        # Bilinear makes propogating flows easier 
        # if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #else:
        #    self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if isinstance(kernel_size, list):
            self.conv = DoubleConvMultiscale(in_ch, out_ch, kernel_size)
        else:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
