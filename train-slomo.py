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
from tensorboardX import SummaryWriter

# from flownet import FlowWarper, SloMoFlowNetMV, SloMoInterpNetMV
import slomo.flownet2 as fl
from slomo import unet
from data import goes16s3
import tools.eval_utils

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--multivariate", dest='multivariate', action='store_true')
parser.add_argument("--channel", default=None, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.set_defaults(multivariate=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

EPOCHS = args.epochs
LEARNING_RATE = 1e-4
BATCH_SIZE = 20


torch.manual_seed(0)


def train_net(n_channels=3,
              model_path='./saved-models/default/',
              example_directory='/raid/tj/GOES/SloMo-5min-band123',
              epochs=20,
              batch_size=1,
              lr=1e-4,
              multivariate=True,
              lambda_w=0.0,
              lambda_s=0.0):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if multivariate:
        flownet_filename = os.path.join(model_path, 'checkpoint.flownet.mv.pth.tar')
        flownet = fl.SloMoFlowNetMV(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNetMV(n_channels)#.cuda()
    else:
        flownet_filename = os.path.join(model_path, 'checkpoint.flownet.pth.tar')
        flownet = fl.SloMoFlowNet(n_channels)#.cuda()
        interpnet = fl.SloMoInterpNet(n_channels)#.cuda()

    warper = fl.FlowWarper()#.cuda()

    flownet = flownet.to(device)
    interpnet = interpnet.to(device)
    warper = warper.to(device)

    if not os.path.exists(model_path):
        os.makedirs(model_path)


    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 8, 'pin_memory': True}

    dataset = goes16s3.GOESDataset(example_directory=example_directory,
                                        n_upsample=9,
                                        n_overlap=3)
    train_size = int(len(dataset)*0.9)
    val_size = len(dataset) - train_size
    training_set, val_set= torch.utils.data.random_split(dataset, [train_size, val_size])
    training_generator = data.DataLoader(training_set, **data_params)
    val_generator = data.DataLoader(val_set, **data_params)
    data_loaders = {"train": training_generator, "val": val_generator}
    data_lengths = {"train": train_size, "val": val_size}

    # define optimizer
    optimizer = torch.optim.Adam(list(flownet.parameters()) + list(interpnet.parameters()),
                                 lr=lr)
    recon_l1_loss = nn.L1Loss()
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

    step = int(start_epoch * data_lengths['train'] / BATCH_SIZE)
    tfwriter = SummaryWriter(os.path.join(model_path, 'tfsummary'))
    print("Begin Training at epoch {}".format(start_epoch))
    best_validation_loss = 1e10
    for epoch in range(start_epoch+1, EPOCHS+1):
        print("\nEpoch {}/{}".format(epoch, EPOCHS))
        print("-"*10)
        t0 = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                flownet.train(True)
                interpnet.train(True)
            else:
                flownet.train(False)
                interpnet.train(False)

            running_loss = 0.0
            for I0, I1, IT in data_loaders[phase]:
                if I0.shape[1] != n_channels: print('N channels dont match with array shape'); continue

                I0, I1, IT = I0.to(device), I1.to(device), IT.to(device, non_blocking=True)

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
                warping_loss_collector = []
                image_collector = []

                T = IT.shape[1]
                for i in range(1,T+1):
                    t = 1. * i / (T+1)

                    # Input Channels: predicted image and warped without derivatives
                    I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(I0, I1, f_01, f_10, t)
                    image_collector.append(I_t)

                    # reconstruction loss
                    loss_recon = recon_l1_loss(I_t, IT[:, i-1])
                    loss_vector.append(loss_recon)

                    # warping loss
                    loss_warp_i= recon_l1_loss(I_t, g0) \
                                + recon_l1_loss(I_t, g1)

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


                loss_warp = recon_l1_loss(I0, I_0_warp) +\
                            recon_l1_loss(I1, I_1_warp)
                loss_warp += sum(warping_loss_collector)/T

                # compute the smoothness loss
                loss_smooth_1_0 = torch.mean(torch.abs(f_10[:,:,:,:-1] - f_10[:,:,:,1:])) +\
                                  torch.mean(torch.abs(f_10[:,:,:-1,:] - f_10[:,:,1:,:]))
                loss_smooth_0_1 = torch.mean(torch.abs(f_01[:,:,:,:-1] - f_01[:,:,:,1:])) +\
                                  torch.mean(torch.abs(f_01[:,:,:-1,:] - f_01[:,:,1:,:]))
                loss_smooth = loss_smooth_0_1 + loss_smooth_1_0

                # take a weighted sum of the losses
                loss = loss_reconstruction + lambda_w * loss_warp + lambda_s * loss_smooth

                # compute the gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if step % 100 == 0:
                    losses = [loss_reconstruction.item(), loss_warp.item(), loss.item()]
                    tfwriter.add_scalar('train/losses/recon', loss_reconstruction, step)
                    tfwriter.add_scalar('train/losses/warp', loss_warp, step)
                    tfwriter.add_scalar('train/losses/smooth', loss_smooth, step)
                    tfwriter.add_scalar('train/total_loss', loss, step)
                    '''
                    if np.isnan(losses[-1]):
                        print("nan loss")
                        train_net(n_channels=n_channels,
                                  model_path=model_path,
                                  example_directory=example_directory,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  lr=lr,
                                  multivariate=multivariate)
                    '''

                step += 1

            state = {'epoch': epoch, 'flownet_state_dict': flownet.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'interpnet_state_dict': interpnet.state_dict()}

            epoch_loss = running_loss / data_lengths[phase]
            torch.save(state, flownet_filename)
            if (phase == 'val') and (epoch_loss < best_validation_loss):
                filename = os.path.join(model_path, 'best.flownet.pth.tar')
                torch.save(state, filename)
                best_validation_loss = epoch_loss

            t = (time.time() - t0)/data_lengths[phase]
            example_per_second = 1./t
            print('[{}] Loss: {:.6f}, Examples per second: {:6f}'.format(phase, epoch_loss,
                                                                         example_per_second))



def run_experiments(multivariate):
    example_directory = '/raid/tj/GOES/SloMo/9Min-%iChannels-Train'
    model_directory = 'saved-models/9Min-%iChannels-LambdaW_%1.2f-LambdaS_%1.2f-Batch' + str(BATCH_SIZE) 
    #lambda_ws = [0.01, 0.1, 0.5, 1.0]
    #lambda_ws = [0.01, 0.1, 0.5, 1.0]
    lambda_ws = [0.1,]
    lambda_ss = [0.1,]
    if multivariate:
        model_directory += '_MV2'

    for c in [3,8]:
        for w in lambda_ws:
            for s in lambda_ss:
                if (args.channel is not None) and (args.channel != c):
                    continue

                if multivariate and (c == 1):
                    continue

                data = example_directory % c
                train_net(model_path=model_directory % (c, w, s),
                          lr=LEARNING_RATE,
                          batch_size=BATCH_SIZE,
                          n_channels=c,
                          example_directory=data,
                          epochs=EPOCHS,
                          multivariate=multivariate,
                          lambda_w=w,
                          lambda_s=s)

if __name__ == "__main__":
    run_experiments(args.multivariate)
