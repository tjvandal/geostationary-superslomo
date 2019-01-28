import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torchvision
from flownet import FlowWarper
import goes16

model_dir = './saved-models/lr1e-3-randshift'
flow_net_model_dir = model_dir + '/flownet.torch'
interp_net_dir = model_dir + '/interpnet.torch'
prediction_dir = model_dir + '/images/'
lowres_dir = model_dir + '/lr-images/'

flow_net = torch.load(flow_net_model_dir).cuda()
flow_net.eval()

interp_net = torch.load(interp_net_dir).cuda()
interp_net.eval()

if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
if not os.path.exists(lowres_dir):
    os.makedirs(lowres_dir)


dir_data = '/raid/tj/GOES16/'
dataset = goes16.NOAAGOESC(data_dir=dir_data)
daydata = dataset.read_day(2017, 120)
dayblocks = dataset.blocks(daydata, width=352)
NBlocks = len(dayblocks)
N, H, W, C = dayblocks[0].shape

# dayblocks (N, NBlocks, height, width, channels)
# given 5 minute data, we want to fill in 2 of every 3 examples

warper = FlowWarper().cuda()

block_num = 11
frame_counter= 0
block = dayblocks[block_num].transpose('time', 'channel', 'y', 'x')

observed_frames = np.arange(0, N, 3)
print('observed frames', observed_frames)

predicted_das = []

for idx, frame0 in enumerate(observed_frames):
    frame1 = min(frame0+3, N-1)
    print('Frame: %i to %i' % (frame0, frame1))
    # check if there are any frames to fill
    if frame1 < frame0 + 1:
        break

    I0 = torch.from_numpy(block.isel(time=[frame0,]).values).cuda()
    I1 = torch.from_numpy(block.isel(time=[frame1,]).values).cuda()
    IT = torch.from_numpy(block.isel(time=range(frame0+1, frame1)).values).cuda()

    x = torch.cat([I0, I1], dim=1)
    f = flow_net(x)

    f_10 = f[:,:2]
    f_01 = f[:,2:]

    T = IT.shape[0]
    predicted_frames = np.ones((T, C, H, W)) * -9999.
    for i in range(1,T+1):
        t = 1. * i / (T+1)
        I_t = interp_net(I0, I1, f_10, f_01, t)
        #predicted_frames.append(I_t)
        predicted_frames[i-1] = I_t[0].cpu().detach().numpy()

    tfilled = block.isel(time=range(frame0+1, frame1)).time.values
    predicted_frames = xr.DataArray(predicted_frames,
                                    coords=[tfilled, block.channel, block.y, block.x], 
                                    dims=['time', 'channel', 'y', 'x'])
    predicted_das.append(predicted_frames)

    # lets save these predictions to images
    #sample_image_file = os.path.join(prediction_dir, '%04i.jpg')
    #print("Saving Predictions")
    #for jj, image in enumerate([I0] + predicted_frames):
    #    join_images = torch.cat([I0, image], dim=3)
    #    torchvision.utils.save_image((join_images), sample_image_file  % frame_counter,
    #                                 normalize=True)
    #    frame_counter += 1

predictions = xr.concat(predicted_das, 'time')
save_file = os.path.join(prediction_dir, 'block-%i.nc' % block_num)
xr.Dataset(dict(Rad=predictions)).to_netcdf(save_file)



