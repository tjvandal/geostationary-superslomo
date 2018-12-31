import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision
import unet

class FlowWarper(torch.nn.Module):
    def __init__(self, gpu=True):
        super(FlowWarper, self).__init__()
        self.gpu = gpu
    # end

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1)
            if self.gpu:
                self.tensorGrid = self.tensorGrid.cuda()
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
    # end

class _FlowWarper(nn.Module):
    def __init__(self, w, h, gpu=True):
        super(FlowWarper, self).__init__()
        x = np.arange(0,w)
        y = np.arange(0,h)
        gx, gy = np.meshgrid(x,y)
        self.w = w
        self.h = h
        self.grid_x = torch.autograd.Variable(torch.Tensor(gx), requires_grad=False)#.cuda()
        self.grid_y = torch.autograd.Variable(torch.Tensor(gy), requires_grad=False)#.cuda()
        if gpu:
            self.grid_x = self.grid_x.cuda()
            self.grid_y = self.grid_y.cuda()

    def forward(self, img, uv):
        u = uv[:,0,:,:]
        v = uv[:,1,:,:]
        X = self.grid_x.unsqueeze(0).expand_as(u) + u
        Y = self.grid_y.unsqueeze(0).expand_as(v) + v
        X = 2*(X/self.w - 0.5)
        Y = 2*(Y/self.h - 0.5)
        grid_tf = torch.stack((X,Y), dim=3)
        img_tf = torch.nn.functional.grid_sample(img, grid_tf)
        return img_tf

class InterpNet(nn.Module):
    def __init__(self, in_channels, out_channels, gpu=True):
        super(InterpNet, self).__init__()
        self.warper = FlowWarper(gpu=gpu)
        self.unet = unet.UNet(in_channels, out_channels)
        if gpu:
            self.warper = self.warper.cuda()
            self.unet = self.unet.cuda()

    def forward(self, I0, I1, F0, F1, t):
        assert F0.shape[1] == 2
        assert F1.shape[1] == 2

        f_t0 = -(1-t) * t * F0 + t**2 * F1
        f_t1 = (1-t)**2 * F0 - t*(1-t) * F1
        g0 = self.warper(I0, f_t0)
        g1 = self.warper(I1, f_t1)

        x_flow = torch.cat([I0, I1, f_t0, f_t1, g0, g1], 1)

        # V_t_0, V_t_1: 1 channel each
        # delta_f_t0, delta_f_t1: 2 channels each
        # I0, I1: 3 channels each
        interp = self.unet(x_flow)
        V_t0 = torch.unsqueeze(interp[:,0], 1)
        V_t1 = torch.unsqueeze(interp[:,1], 1)
        delta_f_t0 = interp[:, 2:4]
        delta_f_t1 = interp[:, 4:6]

        normalization = (1-t) * V_t0 + t * V_t1
        g_0_ft_0 = self.warper(I0, f_t0 + delta_f_t0)
        g_1_ft_1 = self.warper(I1, f_t1 + delta_f_t1)
        I_t = (1-t) * V_t0 * g_0_ft_0 + t * V_t1 * g_1_ft_1
        I_t /= normalization
        return I_t
