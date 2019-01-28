import sys
import os
import time
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torchvision
import unet
import goes16s3

from flownet import FlowWarper, SloMoFlowNet, SloMoInterpNet
import eval_utils

torch.manual_seed(0)

def train_net(n_channels=3,
              model_path='./saved-models/default/',
              epochs=500,
              batch_size=1,
              val_percent=0.05,
              gpu=False,
              lr=1e-2):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flownet = SloMoFlowNet(n_channels)#.cuda()
    interpnet = SloMoInterpNet(n_channels)#.cuda()
    warper = FlowWarper()#.cuda()

    if torch.cuda.device_count() > 0:
        print("Let's use %i GPUs!" % torch.cuda.device_count())
        flownet = nn.DataParallel(flownet)
        interpnet = nn.DataParallel(interpnet)
        warper = nn.DataParallel(warper)

    flownet = flownet.to(device)
    interpnet = interpnet.to(device)
    warper = warper.to(device)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    sample_image_file = 'samples/%03i-%03i.jpg'
    if not os.path.exists("samples"):
        os.makedirs('samples')

    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 1}

    training_set = goes16s3.GOESDatasetS3(s3_base_path='slomo-band1-5min/')
    training_generator = data.DataLoader(training_set, **data_params)

    # define optimizer
    optimizer = torch.optim.Adam(list(flownet.parameters()) + list(interpnet.parameters()),
                                 lr=lr)
    recon_l2_loss = nn.MSELoss()

    flownet.train()
    interpnet.train()

    step = 0
    first_n_T = None
    statsfile = open(os.path.join(model_path, 'loss.txt'), 'w')
    print("Begin Training")
    for epoch in range(epochs):
        for I0, I1, IT in training_generator:
            t0 = time.time()
            if I0.shape[1] != n_channels: continue
            I0, I1, IT = I0.to(device), I1.to(device), IT.to(device)
            T = IT.shape[1]
            f = flownet(I0, I1)
            # x, y optical flows
            f_10 = f[:,:2]
            f_01 = f[:,2:]

            # collect loss data and predictions
            loss_vector = []
            warping_loss_collector = []
            image_collector = []
            g0_flows, g1_flows = [], []
            for i in range(1,T+1):
                t = 1. * i / (T+1)

                # Input Channels: predicted image and warped without derivatives
                I_t, g0, g1 = interpnet(I0, I1, f_10, f_01, t)
                image_collector.append(I_t)

                # reconstruction loss
                loss_recon = recon_l2_loss(I_t, IT[:, i-1])
                loss_vector.append(loss_recon)
                if (loss_recon.item() > 10) and (step > 5):
                    break

                # perceptual loss can not currently be applied because classification are not defined

                # warping loss
                loss_warp_i= recon_l2_loss(I_t, g0) \
                            + recon_l2_loss(I_t, g1)

                warping_loss_collector.append(loss_warp_i)

            # compute the reconstruction loss
            loss_reconstruction = sum(loss_vector)  / T

            # compute the total warping loss
            loss_warp = recon_l2_loss(I0, warper(I1, f_01)) + recon_l2_loss(I1, warper(I0, f_10))
            loss_warp += sum(warping_loss_collector)/T

            # compute the smoothness loss
            loss_smooth_1_0 = torch.mean(torch.abs(f_10[:,:,:,:-1] - f_10[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_10[:,:,:-1,:] - f_10[:,:,1:,:]))
            loss_smooth_0_1 = torch.mean(torch.abs(f_01[:,:,:,:-1] - f_01[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_01[:,:,:-1,:] - f_01[:,:,1:,:]))
            loss_smooth = loss_smooth_0_1 + loss_smooth_1_0

            # take a weighted sum of the losses
            loss = loss_reconstruction + 0.5 * loss_warp + 0.5 * loss_smooth

            # compute the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if step % 1 == 0:
                #for jj, image in enumerate([I0] + image_collector + [I1]):
                    #torchvision.utils.save_image((image[:, [2,0,1]]), 
                    #            sample_image_file  % (step+1, jj), normalize=True)

                losses = [loss_reconstruction.item(), loss_warp.item(), loss_smooth.item(), loss.item()]
                if (losses[3] > 10) and (step > 5):
                    IT_hat_np = torch.cat(image_collector, 1).cpu().detach().numpy()
                    I_t_np = I_t.cpu().detach().numpy()
                    I0_np = I0.cpu().numpy()
                    I1_np = I1.cpu().numpy()
                    IT_np = IT.cpu().numpy()
                    g0_np = g0.cpu().detach().numpy()
                    g1_np = g1.cpu().detach().numpy()
                    print("g0_np", g0_np.shape)

                    print("Losses", losses)
                    print("I0", np.histogram(I0.cpu().flatten()))
                    print("I1", np.histogram(I1.cpu().flatten()))
                    print("IT", np.histogram(IT.cpu().flatten()))
                    print("I_t-Pred", np.histogram(I_t_np.flatten()))

                    import matplotlib.pyplot as plt
                    fig, axs = plt.subplots(3,3)
                    axs = np.ravel(axs)
                    #I0 (N,C,H,W)
                    print(I0_np[0].shape)
                    axs[0].imshow(np.squeeze(np.transpose(I0_np[0], (1,2,0))))
                    axs[1].imshow(np.squeeze(np.transpose(I1_np[0], (1,2,0))))
                    axs[2].imshow(np.squeeze(np.transpose((I1_np[0]-I0_np[0]), (1,2,0))))
                    axs[3].imshow(g0_np[0,0])
                    axs[4].imshow(g0_np[0,1])
                    axs[5].imshow(g1_np[0,0])
                    axs[6].imshow(g1_np[0,1])
                    print(I_t_np[0].shape)
                    axs[7].imshow(np.squeeze(np.transpose(I_t_np[0], (1,2,0))))
                    #axs[8].imshow(np.abs(I_t_np[0]).max(axis=0) > 10)
                    axs[8].imshow(np.squeeze(np.transpose(I_t_np[0] - IT_np[0,i-1], (1,2,0))))
                    plt.show()
                    return

                out = "(Epoch: %i, Iteration %i, Iteration time: %1.4f) Reconstruction Loss=%2.6f\tWarping Loss=%2.6f"\
                      "\tSmooth Loss=%2.6f\tTotal Loss=%2.6f" % (epoch, step, time.time()-t0, losses[0], losses[1],
                                              losses[2], losses[3])
                print(out)
                if np.isnan(losses[0]):
                    for k, l in enumerate([l.item() for l in loss_vector]):
                        print("I0", np.histogram(I0.cpu().flatten()))
                        print("I1", np.histogram(I1.cpu().flatten()))

                losses = [str(l) for l in losses]
                statsfile.write('%s\n' % ','.join(losses))
                #psnrs = [eval_utils.psnr(image_collector[i], IT[i]) for i in range(len(image_collector))]
                #print(psnrs) 

            if step % 100 == 0:
                torch.save(flownet, os.path.join(model_path, 'flownet.torch'))
                torch.save(interpnet, os.path.join(model_path, 'interpnet.torch'))

            step += 1

if __name__ == "__main__":
    train_net(model_path='saved-models/5min-samples/', lr=1e-3, epochs=5, batch_size=1, n_channels=1)
