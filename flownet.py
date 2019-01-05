import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision
import unet

class FlowWarper(nn.Module):
    def __init__(self, device):
        super(FlowWarper, self).__init__()
        self.device = device
    # end

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(self.device)
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

        out = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        return out
    # end

class SloMoFlowNet(nn.Module):
    '''
    The Model:
        I0 and I1 are model inputs
        Flow computation predicts forward and backward optical flows
    '''
    def __init__(self, n_channels=6):
        super(SloMoFlowNet, self).__init__()
        self.n_channels = n_channels
        self.flow_model  = unet.UNet(self.n_channels*2, 4)

    def forward(self, x0, x1):
        x = torch.cat([x0, x1], dim=1)
        f = self.flow_model(x)
        return f

class SloMoInterpNet(nn.Module):
    def __init__(self, n_channels, device):
        super(SloMoInterpNet, self).__init__()
        self.device = device
        self.warper = FlowWarper(device)
        self.unet = unet.UNet(n_channels*4 + 4, n_channels)

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

        return I_t, g0, g1
