import sys
import os

import xarray as xr
from .utils import blocks
import torch
import torch.nn as nn
import torchvision
from slomo import flownet as fl

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def merge_and_average_dataarrays(dataarrays):
    ds = xr.merge([xr.Dataset({k: d}) for k, d in enumerate(dataarrays)])
    das = []
    for b in range(0,len(dataarrays)):
        das.append(ds[b])

    return xr.concat(das).mean('concat_dims', skipna=True)

# Split 3D numpy array into patches with overlap
def split_array(arr, tile_size=128, overlap=16):
    '''
    Split a 3D numpy array into patches for inference
    (Channels, Height, Width)
    Args:
        tile_size: width and height of patches to return
        overlap: number of pixels to overlap between patches
    Returns:
        dict(patches, upper_left): patches and indices of original array
    '''
    arr = arr[np.newaxis]
    width, height = arr.shape[2:4]
    arrs = dict(patches=[], upper_left=[])
    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            i = min(i, width - tile_size)
            j = min(j, height - tile_size)
            arrs['patches'].append(arr[:,:,i:i+tile_size,j:j+tile_size])
            arrs['upper_left'].append([[i,j]])
    arrs['patches'] = np.concatenate(arrs['patches'])
    arrs['upper_left'] = np.concatenate(arrs['upper_left'])
    return arrs['patches'], arrs['upper_left']

def reassemble_array(arr, upper_left_indices, height, width):
    assert len(arr.shape) == 4
    N_patches, channels, pheight, pwidth = arr.shape
    y_sum = np.zeros((arr.shape[1], height, width))
    y_counter = np.zeros((arr.shape[1], height, width))

    for n, (i,j)  in enumerate(upper_left_indices):
        y_counter[:, i:i+pheight, j:j+pwidth] += 1
        y_sum[:, i:i+pheight, j:j+pwidth] += arr[n]

    y = np.where(y_counter!=0,y_sum/y_counter,0)
    return y

## Load and return flownet, interpnet, and warper
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

    flownet = nn.DataParallel(flownet)
    interpnet = nn.DataParallel(interpnet)
    warper = nn.DataParallel(warper)

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

## Perform interpolation for a single timepoint t

def single_inference(X0, X1, t, flownet, interpnet,
                     multivariate):
    X0_arr_torch = torch.from_numpy(X0)
    X1_arr_torch = torch.from_numpy(X1)

    # nans to 0
    X0_arr_torch[np.isnan(X0_arr_torch)] = 0.
    X1_arr_torch[np.isnan(X1_arr_torch)] = 0.

    if len(X0_arr_torch.shape) == 3:
        X0_arr_torch = torch.unsqueeze(X0_arr_torch, 0)
        X1_arr_torch = torch.unsqueeze(X1_arr_torch, 0)

    X0_arr_torch = X0_arr_torch.to(device)
    X1_arr_torch = X1_arr_torch.to(device)

    f = flownet(X0_arr_torch, X1_arr_torch)
    n_channels = X0_arr_torch.shape[1]

    if multivariate:
        f_01 = f[:,:2*n_channels]
        f_10 = f[:,2*n_channels:]
    else:
        f_01 = f[:,:2]
        f_10 = f[:,2:]

    I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(X0_arr_torch, X1_arr_torch, f_01, f_10, t)
    out = {'f_01': f_01, 'f_10': f_10, 'I_t': I_t,
           'V_t0': V_t0, 'V_t1': V_t1, 'delta_f_t0': delta_f_t0,
           'delta_f_t1': delta_f_t0}
    for key in out:
        out[key] = out[key].cpu().detach().numpy()

    #torch.cuda.empty_cache()
    return out

## Split interpolation for a single timepoint t
def single_inference_split(X0, X1, t, flownet, interpnet,
                           multivariate, block_size=128, overlap=16):
    X0_split, upper_left_idxs = split_array(X0, block_size, overlap)
    X1_split, _ = split_array(X1, block_size, overlap)

    # perform inference on patches
    split_res = single_inference(X0_split, X1_split, t, flownet,
                                 interpnet, multivariate)

    # reassemble into arrays
    res = dict()
    for key in split_res:
        res[key] = reassemble_array(split_res[key], upper_left_idxs,
                                    X0.shape[1], X0.shape[2])
    return res


def _inference(X0, X1, flownet, interpnet, warper,
               multivariate, T=4, block_size=None):
    '''
    Given two consecutive frames, interpolation T between them 
        using flownet and interpnet
    Returns:
        Interpolated Frames
    '''

    X0_arr_torch = torch.from_numpy(X0.values)
    X1_arr_torch = torch.from_numpy(X1.values)

    # nans to 0
    X0_arr_torch[np.isnan(X0_arr_torch)] = 0.
    X1_arr_torch[np.isnan(X1_arr_torch)] = 0.

    X0_arr_torch = torch.unsqueeze(X0_arr_torch, 0).to(device)
    X1_arr_torch = torch.unsqueeze(X1_arr_torch, 0).to(device)

    f = flownet(X0_arr_torch, X1_arr_torch)
    n_channels = X0_arr_torch.shape[1]

    if multivariate:
        f_01 = f[:,:2*n_channels]
        f_10 = f[:,2*n_channels:]
    else:
        f_01 = f[:,:2]
        f_10 = f[:,2:]

    predicted_frames = []

    for j in range(1,T+1):
        t = 1. * j / (T+1)
        I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(X0_arr_torch, X1_arr_torch, f_01, f_10, t)
        predicted_frames.append(I_t.cpu().detach().numpy())

    torch.cuda.empty_cache()
    return predicted_frames

## Perform interpolation over multiple time steps
def inference(X0, X1, flownet, interpnet, warper,
              multivariate, T=4, block_size=352):
    '''
    Given two consecutive frames, interpolation T between them 
        using flownet and interpnet
    Returns:
        Interpolated Frames
    '''
    # Get a list of dataarrays chunked
    X0_blocks = blocks(X0, width=block_size)
    X1_blocks = blocks(X1, width=block_size)

    interpolated_blocks = []
    for x0, x1 in zip(X0_blocks, X1_blocks):
        predicted_frames = _inference(x0, x1, flownet,
                                      interpnet, warper,
                                      multivariate, T)
        predicted_frames = [x0.values[np.newaxis]] + predicted_frames + [x1.values[np.newaxis]]
        interpolated_blocks += [block_predictions_to_dataarray(predicted_frames, x0)]
    return merge_and_average_dataarrays(interpolated_blocks)
