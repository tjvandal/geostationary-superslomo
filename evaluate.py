import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from flownet import FlowWarper
import goes16

flow_net_model_dir = './saved-models/model1/flownet.torch'
interp_net_dir = './saved-models/model1/interpnet.torch'
prediction_dir = './saved-models/model1/images/'
lowres_dir = './saved-models/model1/lr-images/'

flow_net = torch.load(flow_net_model_dir).cuda()
flow_net.eval()

interp_net = torch.load(interp_net_dir).cuda()
interp_net.eval()

if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
if not os.path.exists(lowres_dir):
    os.makedirs(lowres_dir)


dir_data = '/raid/tj/GOES16/'
dataset = goes16.NOAAGOES(data_dir=dir_data)

warper = FlowWarper().cuda()

block_num = 75
frame_counter= 0

for i, block in enumerate(dataset.iterate(shuffle=False)): # N,12or13,512,512,3
    if i == 0:
        continue

    print("Number of blocks", block.shape[0])

    block = torch.from_numpy(block[block_num]).cuda()
    block = block.permute(0,3,1,2) # 12,3,512,512

    B1 = block[:4]
    B2 = block[3:7]
    B3 = block[6:10]
    B4 = block[9:]

    2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 2, 7

    for B in [B1, B2, B3, B4]:
        print('Frame: %i' % frame_counter)
        I0 = torch.unsqueeze(B[0], 0)
        I1 = torch.unsqueeze(B[-1], 0)
        IT = B[1:-1]

        x = torch.cat([I0, I1], dim=1)
        f = flow_net(x)

        f_10 = f[:,:2]
        f_01 = f[:,2:]

        T = IT.shape[0]
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

