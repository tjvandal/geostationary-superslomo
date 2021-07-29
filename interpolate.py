import torch
import torch.nn as nn
import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append('/nobackupp10/tvandal/pyPyrTools/pyPyrTools/')
#import phase_based
from networks import slomo, unet
from tools import utils
#import warper

class Interpolate(object):
    def __init__(self, model_checkpoint, nn_model=unet.UNetMedium,
                 occlusion=True, multigpu=True, patch_size=256,
                 stride=100, trim=0):
        self.model_checkpoint = model_checkpoint
        self.net = slomo.SuperSlomo(n_channels=1, model=nn_model, occlusion=occlusion)
        self.patch_size = patch_size
        self.stride = stride
        if multigpu:
            self.net = nn.DataParallel(self.net)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        ## load checkpoint
        checkpoint = torch.load(self.model_checkpoint)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.trim = trim

    def predict(self, X1, X2, t):
        '''
        X1 and X2 should be of size (C, H, W)
        t: float
        '''
        patch_size = self.patch_size
        stride = self.stride
        trim = self.trim

        mask = ~np.isfinite(X1 * X2)
        mean = np.nanmean(X1)
        std = np.nanmean(X2)
        X1[mask] = mean
        X2[mask] = mean
        xf = X1.flatten()

        Y = np.expand_dims(np.zeros_like(X1), 0)
        counter = np.expand_dims(np.zeros_like(X1), 0)

        X1 = torch.from_numpy(X1).float().to(self.device)
        X2 = torch.from_numpy(X2).float().to(self.device)
        t = torch.ones((1,1,1,1)).float().to(self.device) * t
        if len(X1.shape) == 3:
            X1 = X1.unsqueeze(0)
            X2 = X2.unsqueeze(0)

        # normalize X1, X2
        X1_norm = X1 #(X1 - mean) / std
        X2_norm = X2 #(X2 - mean) / std
        H, W = X1.shape[-2:]
        for ih in np.arange(0, H, stride):
            for iw in np.arange(0, W, stride):
                if (iw + patch_size) > W:
                    iw = W - patch_size
                if (ih + patch_size) > H:
                    ih = H - patch_size
                x1 = X1_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]
                x2 = X2_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]
                y_ = self.net(x1, x2, t)['I_t'].detach().cpu().numpy()
                if trim > 0:
                    y_ = y_[:,:,trim:-trim,trim:-trim]
                Y[:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += y_
                counter[:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += 1.

        output = Y / counter
        It = output #* std + mean
        return It[0,0]

    def get_flow_info(self, X1, X2, t, trim=0):
        '''
        X1 and X2 should be of size (C, H, W)
        t: float
        '''
        C, H, W = X1.shape
        patch_size = self.patch_size
        stride = self.stride

        mask = ~np.isfinite(X1 * X2)
        mean = np.nanmean(X1)
        std = np.nanmean(X1)
        X1[mask] = mean
        X2[mask] = mean
        xf = X1.flatten()

        counter = np.zeros((1,C,H,W), dtype=float)

        X1 = torch.from_numpy(X1).float().to(self.device)
        X2 = torch.from_numpy(X2).float().to(self.device)
        t = torch.ones((1,1,1,1)).float().to(self.device) * t
        if len(X1.shape) == 3:
            X1 = X1.unsqueeze(0)
            X2 = X2.unsqueeze(0)

        res = dict()
        # normalize X1, X2
        X1_norm = X1 #(X1 - mean) / std
        X2_norm = X2 #(X2 - mean) / std
        for ih in np.arange(0, H, stride):
            for iw in np.arange(0, W, stride):
                if (iw + patch_size) > W:
                    iw = W - patch_size
                if (ih + patch_size) > H:
                    ih = H - patch_size
                x1 = X1_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]
                x2 = X2_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]
                
                curr_res = self.net(x1, x2, t)
                for key, item in curr_res.items():
                    item = item.detach().cpu().numpy()
                    if key not in res:
                        res[key] = np.zeros((1,item.shape[1], H, W), dtype=np.float32)
                    if trim != 0:
                        item = item[:,:,trim:-trim,trim:-trim]
                    res[key][:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += item
                counter[:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += 1.
                
                #output = dict(I_t=I_t, g0=g0, g1=g1, V_t0=V_t0, V_t1=V_t1,
                #delta_f_t0=delta_f_t0, delta_f_t1=delta_f_t1,
                #f_t0=f_t0, f_t1=f_t1, g_t0=g_0_ft_0, g_t1=g_1_ft_1) 

        for key in res:
            res[key] /= counter

        #for key in ['I_t', 'g0', 'g1', 'g_t0', 'g_t1']:
        #    res[key] = res[key] * std + mean
            
        return res

class Linear(object):
    def __init__(self):
        pass

    def predict(self, X1, X2, t):
        return (1.-t) * X1[0] + t * X2[0]

class PhaseBased(object):
    def __init__(self):
        pass

    def predict(self, X1, X2, t):
        print("X1", X1.shape)
        phased_base.interpolate_frame(X1, X2)
        return

## NOT FINISHED ###
class Flownet(object):
    def __init__(self, model_checkpoint, nn_model=unet.UNetMedium, patch_size=256, stride=100, trim=0, multigpu=False):
        self.model_checkpoint = model_checkpoint
        self.net = slomo.FlowNet(n_channels=1, model=nn_model)
        self.patch_size = patch_size
        self.stride = stride
        if multigpu:
            self.net = nn.DataParallel(self.net)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)

        ## load checkpoint
        checkpoint = torch.load(self.model_checkpoint)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.trim = trim
        self.warp = warper.FlowWarper()

    def predict(self, X1, X2, t):
        '''
        X1 and X2 should be of size (C, H, W)
        t: float
        '''
        patch_size = self.patch_size
        stride = self.stride
        trim = self.trim

        mask = ~np.isfinite(X1 * X2)
        mean = np.nanmean(X1)
        std = np.nanmean(X2)
        X1[mask] = mean
        X2[mask] = mean
        xf = X1.flatten()

        Y = np.expand_dims(np.zeros_like(X1), 0)
        counter = np.expand_dims(np.zeros_like(X1), 0)

        X1 = torch.from_numpy(X1).float().to(self.device)
        X2 = torch.from_numpy(X2).float().to(self.device)
        t = torch.ones((1,1,1,1)).float().to(self.device) * t
        if len(X1.shape) == 3:
            X1 = X1.unsqueeze(0)
            X2 = X2.unsqueeze(0)

        # normalize X1, X2
        X1_norm = X1 #(X1 - mean) / std
        X2_norm = X2 #(X2 - mean) / std
        H, W = X1.shape[-2:]
        for ih in np.arange(0, H, stride):
            for iw in np.arange(0, W, stride):
                if (iw + patch_size) > W:
                    iw = W - patch_size
                if (ih + patch_size) > H:
                    ih = H - patch_size
                x1 = X1_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]
                x2 = X2_norm[:,:,ih:ih+patch_size,iw:iw+patch_size]

                F = self.net(x1, x2)#.detach().cpu().numpy()
                F_01 = F[:,:self.n_channels]
                F_10 = F[:,self.n_channels:]

                ## OCCLUSION AND INTERPOLATION NEEDS TO GO HERE

                if trim > 0:
                    y_ = y_[:,:,trim:-trim,trim:-trim]
                Y[:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += y_
                counter[:,:,ih+trim:ih+patch_size-trim,iw+trim:iw+patch_size-trim] += 1.

        output = Y / counter
        It = output #* std + mean
        return It[0,0]
