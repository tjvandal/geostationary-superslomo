import os
import xarray as xr
import utils
import torch
import torchvision
import flownet as fl
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_models(n_channels, model_path, multivariate=False):
    
    if multivariate:
        model_filename = os.path.join(model_path, 'checkpoint.flownet.mv.pth.tar')
        flownet = fl.SloMoFlowNetMV(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNetMV(n_channels)#.cuda()
    else:
        model_filename = os.path.join(model_path, 'checkpoint.flownet.pth.tar')
        flownet = fl.SloMoFlowNet(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNet(n_channels)#.cuda()
        
    warper = fl.FlowWarper()
    
    flownet = flownet.to(device)
    interpnet = interpnet.to(device)
    warper = warper.to(device)

    def load_checkpoint(flownet, interpnet):
        epoch = 0
        if os.path.isfile(model_filename):
            print("loading checkpoint %s" % model_filename)
            checkpoint = torch.load(model_filename)
            flownet.load_state_dict(checkpoint['flownet_state_dict'])
            interpnet.load_state_dict(checkpoint['interpnet_state_dict'])
            epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(model_filename, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(model_filename))
        return flownet, interpnet

    flownet.train()
    interpnet.train()
    flownet, interpnet = load_checkpoint(flownet, interpnet)
    return flownet, interpnet, warper

def block_predictions_to_dataarray(predictions, block):
    block_predictions = np.concatenate(predictions, 0)
    block_predictions[block_predictions < 0] = 0
    block_predictions[block_predictions > 1] = 1

    N_pred = block_predictions.shape[0]
    T = np.arange(0,N_pred)
    da = xr.DataArray(block_predictions,#[:,:,shave:-shave,shave:-shave],
              coords=[T, block.band.values,
                      block.y.values,#[shave:-shave], 
                      block.x.values,],#][shave:-shave]],
              dims=['t', 'band', 'y', 'x'])

    return da

def inference_5min_block(block, flownet, interpnet, warper, multivariate, T=4):
    block_vals = torch.from_numpy(block.values)
    block_vals[np.isnan(block_vals)] = 0
    n_channels = block_vals.shape[1]
    N = block_vals.shape[0]
    idxs = np.arange(0, N)
    preds = []
    for idx0, idx1 in zip(idxs[:-1], idxs[1:]):
        idx1 = min(N-1, idx1)
        I0 = torch.unsqueeze(block_vals[idx0], 0).to(device)
        I1 = torch.unsqueeze(block_vals[idx1], 0).to(device)

        f = flownet(I0, I1)
        n_channels = I0.shape[1]
        
        if multivariate:
            f_01 = f[:,:2*n_channels]
            f_10 = f[:,2*n_channels:]
        else:
            f_01 = f[:,:2]
            f_10 = f[:,2:]

        predicted_frames = []
        for j in range(1,T+1):
            t = 1. * j / (T+1)
            I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(I0, I1, f_01, f_10, t)
            predicted_frames.append(I_t.cpu().detach().numpy())
        
        preds += [I0.cpu().numpy()] + predicted_frames
    del f_01, f_10, f, I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1
    torch.cuda.empty_cache()
    return block_predictions_to_dataarray(preds, block)

def merge_and_average_dataarrays(dataarrays):
    ds = xr.merge([xr.Dataset({k: d}) for k, d in enumerate(dataarrays)])
    das = []
    for b in range(0,len(dataarrays)):
        das.append(ds[b])

    return xr.concat(das).mean('concat_dims', skipna=True)

