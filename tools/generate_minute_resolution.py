import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from flownet import FlowWarper
import goes16s3

model = '5Min-3Channels'
flow_net_model_dir = './saved-models/%s/flownet.torch' % model
interp_net_dir = './saved-models/%s/interpnet.torch' % model
prediction_dir = './saved-models/%s/images-minute/' % model
lowres_dir = './saved-models/%s/lr-images-minute/' % model

flow_net = torch.load(flow_net_model_dir)
flow_net.eval()

interp_net = torch.load(interp_net_dir)
interp_net.eval()

if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
if not os.path.exists(lowres_dir):
    os.makedirs(lowres_dir)

sys.exit()
dir_data = './data/'
dataset = goes16.GOESDataset(noaagoes_data_dir=dir_data)


warper = FlowWarper(512,512)

block_num = 75
frame_counter= 0

for i, block in enumerate(dataset.iterate(shuffle=False)): # N,12or13,512,512,3
    if i == 0:
        continue

    print("Number of blocks", block.shape[0])

    block = torch.from_numpy(block[block_num])
    block = block.permute(0,3,1,2) # 12,3,512,512

    B1 = block[:7]
    B2 = block[6:]

    # 2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 2, 7

    for b in range(block.shape[0] - 1):
        print('Frame: %i' % frame_counter)
        I0 = torch.unsqueeze(block[b], 0)
        I1 = torch.unsqueeze(block[b+1], 0)

        x = torch.cat([I0, I1], dim=1)
        f = flow_net(x)

        f_10 = f[:,:2]
        f_01 = f[:,2:]

        T = 4
        predicted_frames = []
        for i in range(1,T+1):
            t = 1. * i / (T+1)
            I_t = interp_net(I0, I1, f_10, f_01, t)
            predicted_frames.append(I_t)
            print(str(i) + "\tI_t", I_t.shape)

        # lets save these predictions to images!
        sample_image_file = os.path.join(prediction_dir, '%04i.jpg')
        obs_image_file = os.path.join(lowres_dir, '%04i.jpg')
        torchvision.utils.save_image((I0), obs_image_file  % frame_counter, normalize=True)

        for jj, image in enumerate([I0] + predicted_frames):
            join_images = torch.cat([I0, image], dim=3)
            torchvision.utils.save_image((join_images), sample_image_file  % frame_counter,
                                         normalize=True)
            frame_counter += 1

