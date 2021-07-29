import sys
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from data import dataloader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import unet, slomo
from networks.warper import BackwardWarp

def train_net(dataset=None,
              example_directory='training-data/Interp-ABI-L1b-RadM-20min/',
              model_path='.tmp/default/',
              epochs=100,
              batch_size=1,
              max_iterations=1000000,
              lr=1e-4,
              lambda_w=1.,
              lambda_s=1.,
              model_name='unet-medium',
              occlusion=True,
              progress=10):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # select inner neural network 
    if model_name == 'unet-small':
        nn_model = unet.UNetSmall
    elif model_name == 'unet-medium':
        nn_model = unet.UNetMedium
    elif model_name == 'unet-multiscale':
        nn_model = unet.UNetMultiscale
    elif model_name == 'unet-multiscalev2':
        nn_model = unet.UNetMultiscaleV2

    # Load model
    flownet_filename = os.path.join(model_path, 'checkpoint.flownet.pth.tar')
    net = slomo.SuperSlomo(n_channels=1, model=nn_model, occlusion=occlusion)
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)
    net = net.to(device)
    warper = BackwardWarp()

    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # Load dataset
    if dataset is None:
        dataset = dataloader.InterpLoader(example_directory)
    train_size = int(len(dataset)*0.9)
    val_size = len(dataset) - train_size
    print("train_size: {}, val size: {}".format(train_size, val_size))

    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 20, 'pin_memory': True}
    training_set, val_set= torch.utils.data.random_split(dataset, [train_size, val_size])
    training_generator = data.DataLoader(training_set, **data_params)
    val_generator = data.DataLoader(val_set, **data_params)
    data_loaders = {"train": training_generator, "val": val_generator}
    data_lengths = {"train": train_size, "val": val_size}

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    # define losses
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    smooth_loss = slomo.SmoothnessLoss()

    def load_checkpoint(net, optimizer, filename):
        start_epoch = 0
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        return net, optimizer, start_epoch

    net.train()
    net, optimizer, start_epoch = load_checkpoint(net, optimizer, flownet_filename)

    step = int(start_epoch * data_lengths['train'] / batch_size)
    tfwriter = SummaryWriter(os.path.join(model_path, 'tfsummary'))
    stats_file = os.path.join(model_path, 'training_curve.csv')

    def write_stats(step,time,rmse,loss):
        with open(stats_file, 'a') as fopen:
            fopen.write('%i,%f,%f,%f\n' % (step, time, rmse, loss))

    print("Begin Training at epoch {}".format(start_epoch))
    best_validation_loss = 1e10

    sample0 = next(iter(data_loaders['train']))[0]
    for epoch in range(start_epoch+1, epochs+1):
        if step < max_iterations:
            print("Epoch {}/{}".format(epoch, epochs))
            print("-"*10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train(True)
                else:
                    net.train(False)

                running_loss = 0.0
                t0 = time.time()
                for batch_idx, (sample, t_sample) in enumerate(data_loaders[phase]):
                    sample = sample.unsqueeze(2) ## add a channel index
                    t_sample = t_sample.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device).float()
                    sample = sample.to(device, dtype=torch.float)

                    I0 = sample[:,0]
                    label = sample[:,1]
                    I1 = sample[:,2]

                    output = net(I0, I1, t_sample)
                    ##### Compute Losses ##### 
                    # Reconstruction Loss
                    loss_reconstruction = l1_loss(output['I_t'], label)

                    # Warping loss
                    loss_warp = l1_loss(I1, output['g1']) +\
                                l1_loss(I0, output['g0'])
                    I_t1_warp = warper(I1, output['f_t1'])
                    I_t0_warp = warper(I0, output['f_t0'])
                    loss_warp += l1_loss(label, I_t0_warp) +\
                                 l1_loss(label, I_t1_warp)

                    # Smoothness loss
                    loss_smooth = smooth_loss(output['f_01']) + smooth_loss(output['f_10'])

                    # Total loss 
                    loss = loss_reconstruction + lambda_w * loss_warp + lambda_s * loss_smooth

                    ###### Optimization ######

                    # compute the gradient
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    def rmse():
                        rmse = torch.sum((label - output['I_t'])**2, (2,3))**0.5
                        return torch.mean(rmse)

                    running_loss += loss_reconstruction.item()
                    if batch_idx % progress == 0:
                        if phase == 'val':
                            write_stats(step, time.time() - t0, rmse().item(), loss.item())

                        def scale_image(x):
                            xmn = torch.min(x)
                            xmx = torch.max(x)
                            return (x - xmn) / (xmx - xmn)

                        tfwriter.add_scalar('losses/recon', loss_reconstruction, step)
                        tfwriter.add_scalar('losses/warp', loss_warp, step)
                        tfwriter.add_scalar('losses/smooth', loss_smooth, step)
                        tfwriter.add_scalar('losses/total', loss, step)
                        tfwriter.add_histogram('sample', sample, step)
                        tfwriter.add_image('images/I0', scale_image(I0[0]), step)
                        tfwriter.add_image('images/I1', scale_image(I1[0]), step)
                        tfwriter.add_image('images/It', scale_image(output['I_t'][0]), step)
                        tfwriter.add_image('images/flows_01', scale_image(output['f_01'][0]), step)
                        tfwriter.add_image('images/flows_t0', scale_image(output['f_t0'][0] +
                                                                   output['delta_f_t0'][0]), step)
                        tfwriter.add_histogram('flows_t0', output['f_t0'] + output['delta_f_t0'], step)
                        examples_per_second = batch_idx * batch_size / (time.time() - t0)
                        if phase == 'train':
                            ssize = train_size
                        else:
                            ssize = val_size
                        print('[{}] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tExamples/Second: {:.0f}'.
                                    format(phase, epoch, batch_idx * batch_size,
                                           ssize, 100 * batch_size * batch_idx / ssize,
                                           loss.item(), examples_per_second))
                    if phase == 'train':
                        step += 1

                state = {'epoch': epoch, 'state_dict': net.state_dict(),
                         'optimizer': optimizer.state_dict()}

                epoch_loss = running_loss * batch_size / data_lengths[phase]

                torch.save(state, flownet_filename)
                if (phase == 'val') and (epoch_loss < best_validation_loss):
                    filename = os.path.join(model_path, 'best.flownet.pth.tar')
                    torch.save(state, filename)
                    best_validation_loss = epoch_loss

                t = (time.time() - t0)/data_lengths[phase]
                example_per_second = 1./t
                print('[{}] Loss: {:.6f}, Examples per second: {:6f}'.format(phase, epoch_loss,
                                                                             example_per_second))
    state = {'step': epoch, 'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict()}

    epoch_loss = running_loss * batch_size / data_lengths[phase]

    torch.save(state, flownet_filename)
    return best_validation_loss

def manual_experiment(args):
    s = args.lambda_s
    w = args.lambda_w

    mu, sd = dataloader.get_band_stats(args.band)
    dataset = dataloader.InterpLoader(args.data, patch_size=256, mean=mu, std=sd)

    train_net(model_path=args.model_directory,
              lr=args.learning_rate,
              batch_size=args.batch_size,
              dataset=dataset,
              epochs=args.epochs,
              lambda_w=args.lambda_w,
              lambda_s=args.lambda_s,
              model_name=args.model_name)
 
if __name__ == "__main__":
    # Feb 1 2020, Band 1 hyper-parameter search
    # {'w': 0.6490224421024322, 's': 0.22545622639358046, 'batch_size': 128}

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lambda_s", default=0.23, type=float)
    parser.add_argument("--lambda_w", default=0.65, type=float)
    parser.add_argument("--model_directory", default="models_weights/default/Channel-01", type=str)
    parser.add_argument("--data",
                        default="data/training-data/Interp-ABI-L1b-RadM-15min-264x264/Channel-01", type=str)
    parser.add_argument("--band", default=1, type=int)
    parser.add_argument("--model_name", default="unet-medium", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    BATCH_SIZE = args.batch_size

    torch.manual_seed(0)

    manual_experiment(args)
