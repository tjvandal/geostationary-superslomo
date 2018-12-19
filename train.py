import sys
import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torchvision
import unet
import goes16

from flownet import FlowWarper, InterpNet
import eval_utils

def train_net(flow_net,
              interp_net,
              model_path='./saved-models/default/',
              epochs=5,
              batch_size=1,
              val_percent=0.05,
              gpu=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow_net = flow_net.cuda()
    interp_net = interp_net.cuda()


    if not os.path.exists(model_path):
        os.makedirs(model_path)

    sample_image_file = 'samples/%03i-%03i.jpg'
    if not os.path.exists("samples"):
        os.makedirs('samples')

    data_params = {'batch_size': 10, 'shuffle': True,
                   'num_workers': 4}

    dir_data = '/raid/tj/GOES16/'
    training_set = goes16.GOESDataset(example_dir='/raid/tj/GOES16/pytorch-training/')
    training_generator = data.DataLoader(training_set, **data_params)

    optimizer = torch.optim.Adam(list(flow_net.parameters()) + list(interp_net.parameters()), lr=1e-4)

    recon_l2_loss = nn.MSELoss()

    flow_net.train()
    interp_net.train()

    warper = FlowWarper().cuda()
    step = 0

    statsfile = open(os.path.join(model_path, 'loss.txt'), 'w')

    for epoch in range(epochs):
        for I0, I1, IT in training_generator:
            # this is a hack, should be ignored in data writing
            if not np.all(np.isfinite(I0)):
                print("I0 is not all finite")
                continue
            if not np.all(np.isfinite(I1)):
                print("I1 is not all finite")
                continue
            if not np.all(np.isfinite(IT)):
                print("IT is not all finite")
                continue

            I0, I1, IT = I0.to(device), I1.to(device), IT.to(device)
            T = IT.shape[1]

            x = torch.cat([I0, I1], dim=1)
            f = flow_net(x)

            f_10 = f[:,:2]
            f_01 = f[:,2:]

            loss_vector = []
            perceptual_loss_collector = []
            warping_loss_collector = []

            image_collector = []
            for i in range(1,T+1):
                t = 1. * i / (T+1)
                I_t = interp_net(I0, I1, f_10, f_01, t)

                f_t0 = -(1-t) * t * f_01 + t**2 * f_10
                f_t1 = (1-t)**2 * f_01 - t*(1-t)*f_10
                image_collector.append(I_t)

                # reconstruction loss
                loss_recon = recon_l2_loss(I_t, IT[:, i-1])
                loss_vector.append(loss_recon)

                # perceptual loss can not currently be applied because classification are not defined

                # warping loss
                loss_warp_i= recon_l2_loss(I_t, warper(I0, f_t0)) + recon_l2_loss(I_t, warper(I1, f_t1))
                warping_loss_collector.append(loss_warp_i)


            loss_reconstruction = sum(loss_vector)  / T

            loss_warp = recon_l2_loss(I0, warper(I1, f_01)) + recon_l2_loss(I1, warper(I0, f_10))
            loss_warp += sum(warping_loss_collector)/T

            loss_smooth_1_0 = torch.mean(torch.abs(f_10[:,:,:,:-1] - f_10[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_10[:,:,:-1,:] - f_10[:,:,1:,:]))
            loss_smooth_0_1 = torch.mean(torch.abs(f_01[:,:,:,:-1] - f_01[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_01[:,:,:-1,:] - f_01[:,:,1:,:]))
            loss_smooth = loss_smooth_0_1 + loss_smooth_1_0

            loss = loss_reconstruction + 0.4 * loss_warp #+ 0.5 * loss_smooth

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()



            for jj, image in enumerate([I0] + image_collector + [I1]):
                torchvision.utils.save_image((image), sample_image_file  % (step+1, jj), normalize=True)

            if step % 1 == 0:
                losses = [loss_reconstruction.item(), loss_warp.item(), loss_smooth.item(), loss.item()]
                out = "(Epoch: %i, Iteration %i) Reconstruction Loss=%2.6f\tWarping Loss=%2.6f\tSmoothing Loss=%2.6f"\
                      "\tTotal Loss=%2.6f" % (epoch, step, losses[0], losses[1], losses[2], losses[3])
                print(out)
                if np.isnan(losses[0]):
                    for k, l in enumerate([l.item() for l in loss_vector]):
                        print("I0", np.histogram(I0.cpu().flatten()))
                        print("I1", np.histogram(I1.cpu().flatten()))

                losses = [str(l) for l in losses]
                statsfile.write('%s\n' % ','.join(losses))
                #psnrs = [eval_utils.psnr(image_collector[i], IT[i]) for i in range(len(image_collector))]
                #print(psnrs) 
                torch.save(flow_net, os.path.join(model_path, 'flownet.torch'))
                torch.save(interp_net, os.path.join(model_path, 'interpnet.torch'))

            step += 1

if __name__ == "__main__":
    train_net(unet.UNet(6, 4), InterpNet(16, 6))
