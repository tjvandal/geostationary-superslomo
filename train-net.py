import sys
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torchvision
import unet
import goes16s3

# from flownet import FlowWarper, SloMoFlowNetMV, SloMoInterpNetMV
import flownet as fl
import eval_utils

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--multivariate", dest='multivariate', action='store_true')
parser.add_argument("--channel", default=None, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.set_defaults(multivariate=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

EPOCHS = args.epochs
LEARNING_RATE = 1e-4
BATCH_SIZE = 5


torch.manual_seed(0)

def train_net(n_channels=3,
              model_path='./saved-models/default/',
              example_directory='/raid/tj/GOES/SloMo-5min-band123',
              epochs=500,
              batch_size=1,
              lr=1e-4,
              multivariate=True):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if multivariate:
        flownet_filename = os.path.join(model_path, 'checkpoint.flownet.mv.pth.tar')
        interpnet_filename= os.path.join(model_path, 'checkpoint.interpnet.mv.pth.tar')
        flownet = fl.SloMoFlowNetMV(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNetMV(n_channels)#.cuda()
    else:
        flownet_filename = os.path.join(model_path, 'checkpoint.flownet.pth.tar')
        interpnet_filename= os.path.join(model_path, 'checkpoint.interpnet.pth.tar')
        flownet = fl.SloMoFlowNet(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNet(n_channels)#.cuda()

    warper = fl.FlowWarper()#.cuda()

    flownet = flownet.to(device)
    interpnet = interpnet.to(device)
    warper = warper.to(device)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 4}

    training_set = goes16s3.GOESDataset(example_directory=example_directory)
    training_generator = data.DataLoader(training_set, **data_params)

    # define optimizer
    optimizer = torch.optim.Adam(list(flownet.parameters()) + list(interpnet.parameters()),
                                 lr=lr)
    recon_l2_loss = nn.MSELoss()


    def load_checkpoint(flownet, interpnet, optimizer, filename):
        start_epoch = 0
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            flownet.load_state_dict(checkpoint['flownet_state_dict'])
            interpnet.load_state_dict(checkpoint['interpnet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        return flownet, interpnet, optimizer, start_epoch

    flownet.train()
    interpnet.train()
    flownet, interpnet, optimizer, start_epoch = load_checkpoint(flownet, interpnet, optimizer,
                                                      filename=flownet_filename)

    step = 0
    statsfile = open(os.path.join(model_path, 'loss.txt'), 'w')
    print("Begin Training at epoch {}".format(start_epoch))
    for epoch in range(start_epoch, EPOCHS):
        for I0, I1, IT in training_generator:
            t0 = time.time()
            if I0.shape[1] != n_channels: print('N channels dont match with array shape'); continue

            I0, I1, IT = I0.to(device), I1.to(device), IT.to(device)
            T = IT.shape[1]
            f = flownet(I0, I1)  # optical flows per channel
            # x, y optical flows
            if multivariate:
                f_01 = f[:,:2*n_channels]
                f_10 = f[:,2*n_channels:]
            else:
                f_01 = f[:,:2]
                f_10 = f[:,2:]

            # collect loss data and predictions
            loss_vector = []
            perceptual_loss_collector = []
            warping_loss_collector = []
            image_collector = []
            for i in range(1,T+1):
                t = 1. * i / (T+1)

                # Input Channels: predicted image and warped without derivatives
                I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(I0, I1, f_01, f_10, t)
                image_collector.append(I_t)

                # reconstruction loss
                loss_recon = recon_l2_loss(I_t, IT[:, i-1])
                loss_vector.append(loss_recon)

                # perceptual loss can not currently be applied because classification are not defined

                # warping loss
                loss_warp_i= recon_l2_loss(I_t, g0) \
                            + recon_l2_loss(I_t, g1)

                warping_loss_collector.append(loss_warp_i)

            # compute the reconstruction loss
            loss_reconstruction = sum(loss_vector)  / T

            # compute the total warping loss
            I_hat_0, I_hat_1 = [], []
            if multivariate:
                I_0_warp, I_1_warp = [], []
                for c in range(n_channels):
                    I_0_warp.append(warper(I1[:,c].unsqueeze(1), f_10[:,c*2:(c+1)*2]))
                    I_1_warp.append(warper(I0[:,c].unsqueeze(1), f_01[:,c*2:(c+1)*2]))
                I_0_warp = torch.cat(I_0_warp,  1)
                I_1_warp = torch.cat(I_1_warp,  1)
            else:
                I_0_warp = warper(I1, f_10)
                I_1_warp = warper(I0, f_01)


            loss_warp = recon_l2_loss(I0, I_0_warp) +\
                        recon_l2_loss(I1, I_1_warp)
            loss_warp += sum(warping_loss_collector)/T

            # compute the smoothness loss
            loss_smooth_1_0 = torch.mean(torch.abs(f_10[:,:,:,:-1] - f_10[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_10[:,:,:-1,:] - f_10[:,:,1:,:]))
            loss_smooth_0_1 = torch.mean(torch.abs(f_01[:,:,:,:-1] - f_01[:,:,:,1:])) +\
                              torch.mean(torch.abs(f_01[:,:,:-1,:] - f_01[:,:,1:,:]))
            loss_smooth = loss_smooth_0_1 + loss_smooth_1_0

            # take a weighted sum of the losses
            loss = 0.8 * loss_reconstruction + 0.4 * loss_warp + 1. * loss_smooth

            # compute the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            IT_hat = torch.cat(image_collector, 1)
            if step % 10 == 0:
                #for jj, image in enumerate([I0] + image_collector + [I1]):
                    #torchvision.utils.save_image((image[:, [2,0,1]]), 
                    #            sample_image_file  % (step+1, jj), normalize=True)

                losses = [loss_reconstruction.item(), loss_warp.item(), loss.item()]
                if losses[2] > 1e6:
                    print("Losses", losses)
                    print("I0", np.histogram(I0.cpu().flatten()))
                    print("I1", np.histogram(I1.cpu().flatten()))
                    print("IT", np.histogram(IT.cpu().flatten()))
                    print("IT-Pred", np.histogram(IT_hat.cpu().detach().numpy().flatten()))
                    return

                out = "(Epoch: %i, Iteration %i, Iteration time: %1.2f) Reconstruction Loss=%2.6f\tWarping Loss=%2.6f"\
                      "\tTotal Loss=%2.6f" % (epoch+1, step, time.time()-t0, losses[0], losses[1], losses[2])
                if np.isnan(losses[-1]):
                    print("nan loss")
                    train_net(n_channels=n_channels,
                              model_path=model_path,
                              example_directory=example_directory,
                              epochs=epochs,
                              batch_size=batch_size,
                              lr=lr,
                              multivariate=multivariate)


                losses = [str(l) for l in losses]
                statsfile.write('%s\n' % ','.join(losses))
                #psnrs = [eval_utils.psnr(image_collector[i], IT[i]) for i in range(len(image_collector))]
                #print(psnrs) 
                print(out)

            step += 1

        state = {'epoch': epoch, 'flownet_state_dict': flownet.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'interpnet_state_dict': interpnet.state_dict()}

        torch.save(state, flownet_filename)

def run_experiments(multivariate):
    example_directory = '/raid/tj/GOES/5Min-%iChannels-Train'
    model_directory = 'saved-models/5Min-%iChannels'
    for c in [1,3,5,8]:
        if (args.channel is not None) and (args.channel != c):
            continue

        if multivariate and (c == 1):
            continue

        if multivariate:
            print("Training MV Model with %i Channels" % c)
        else:
            print("Training Model with %i Channels" % c)

        data = example_directory % c
        train_net(model_path=model_directory % c,
                  lr=LEARNING_RATE,
                  batch_size=BATCH_SIZE,
                  n_channels=c,
                  example_directory=data,
                  epochs=EPOCHS,
                  multivariate=multivariate)

if __name__ == "__main__":
    run_experiments(args.multivariate)
