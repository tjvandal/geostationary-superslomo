import numpy as np

import torch
import torch.nn as nn
from . import unet
from .warper import BackwardWarp

class FlowNet(nn.Module):
    '''
    The Model:
        I0 and I1 are model inputs
        Flow computation predicts forward and backward optical flows
    '''
    def __init__(self, n_channels=5, model=unet.UNetSmall,
                 multivariate=False):
        super(FlowNet, self).__init__()
        self.n_channels = n_channels
        if not multivariate:
            self.out_channels = 2
        else:
            self.out_channels = self.n_channels*2
        self.flow_model  = model(self.n_channels*2, self.out_channels)

    def forward(self, x0, x1):
        x = torch.cat([x0, x1], dim=1)
        f = self.flow_model(x)
        return f

class SloMoFlowNet(FlowNet):
    def __init__(self, n_channels=5, model=unet.UNetMedium,
                 multivariate=False):
        super(SloMoFlowNet, self).__init__(n_channels=n_channels, multivariate=multivariate,
                                           model=model)
        self.n_channels = n_channels
        if not multivariate:
            self.out_channels = 4
        else:
            self.out_channels = self.n_channels*4
        self.flow_model  = model(self.n_channels*2, self.out_channels)

class SloMoInterpNet(nn.Module):
    def __init__(self, n_channels, model=unet.UNetMedium,
                 multivariate=False, ignore_visible=False):
        super(SloMoInterpNet, self).__init__()
        #self.device = device
        self.warper = BackwardWarp()
        self.n_channels = n_channels
        self.multivariate = multivariate
        self.ignore_visible = ignore_visible
        if multivariate:
            in_channels = n_channels*4 + n_channels*4
            out_channels = n_channels + 4*n_channels
        else:
            in_channels = n_channels * 4 + 4
            out_channels = 5
        self.model = model(in_channels, out_channels)

    def forward(self, I0, I1, F01, F10, t):
        assert F01.shape[1] == 2
        assert F10.shape[1] == 2
        f_t0 = -(1-t) * t * F01 + t**2 * F10
        f_t1 = (1-t)**2 * F01 - t*(1-t) * F10

        g1 = self.warper(I0, F01)
        g0 = self.warper(I1, F10)

        #next_index = g0.device.index + 1
        if torch.cuda.is_available():
            g0 = g0.cuda()
            g1 = g1.cuda()
        x_flow = torch.cat([I0, I1, f_t0, f_t1, g0, g1], 1)

        # V_t_0, V_t_1: 1 channel each
        # delta_f_t0, delta_f_t1: 2 channels each
        # I0, I1: 3 channels each

        interp = self.model(x_flow)
        delta_f_t0 = interp[:, 1:3]
        delta_f_t1 = interp[:, 3:5]
        V = interp[:,:1]

        if self.ignore_visible:
            V_t0 = torch.ones_like(V) * 0.5
            V_t1 = V_t0
        else:
            V_t0 = torch.sigmoid(V) #+ 1e-6
            V_t1 = 1 - V_t0

        normalization = (1-t) * V_t0 + t * V_t1
        g_0_ft_0 = self.warper(I0, f_t0 + delta_f_t0)
        g_1_ft_1 = self.warper(I1, f_t1 + delta_f_t1)

        I_t = (1-t) * V_t0 * g_0_ft_0 + t * V_t1 * g_1_ft_1
        I_t /= normalization

        output = dict(I_t=I_t, g0=g0, g1=g1, V_t0=V_t0, V_t1=V_t1,
                      delta_f_t0=delta_f_t0, delta_f_t1=delta_f_t1,
                      f_t0=f_t0, f_t1=f_t1, g_t0=g_0_ft_0, g_t1=g_1_ft_1)
        return output

class SuperSlomo(nn.Module):
    def __init__(self, n_channels=1, model=unet.UNetMedium, occlusion=True):
        super(SuperSlomo, self).__init__()
        self.n_channels = n_channels
        self.nn_model = model
        self.occlusion = occlusion
        self.net1 = SloMoFlowNet(self.n_channels, self.nn_model)
        self.net2 = SloMoInterpNet(self.n_channels, self.nn_model,
                                   ignore_visible=(not self.occlusion))
        self.warper = BackwardWarp()

    def forward(self, X1, X2, t):
        flows_1 = self.net1(X1, X2)
        flows_1_01 = flows_1[:,:2]
        flows_1_10 = flows_1[:,2:]
        interp = self.net2(X1, X2,flows_1_01,flows_1_10,t)
        interp['f_01'] = flows_1_01
        interp['f_10'] = flows_1_10
        return interp

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flows):
        loss_smooth = torch.mean(torch.abs(flows[:,:,:,:-1] - flows[:,:,:,1:])) +\
                      torch.mean(torch.abs(flows[:,:,:-1,:] - flows[:,:,1:,:]))
        return loss_smooth
