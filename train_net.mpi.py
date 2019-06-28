'''
1 GPU per network to train -- This is the fastest way to train a large number of networks
'''

import os, sys
from mpi4py import MPI

import train_net

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

N_GPUS = 8

example_directory = '/nobackupp10/tvandal/GOES-SloMo/data/training/9Min-{}Channels-Train-pt'
model_directory = 'saved-models/v1/9Min-{}Channels-{}'

best_params = {"lr": 0.001, "w": 0.01, "s": 1.54, "batch_size": 128}
epochs = 50

param_list = [(c, m) for c in [3, 8, 16] for m in [True, False]]

# set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % N_GPUS)

for i, p in enumerate(param_list):
    if i % N_GPUS == rank:
        if p[1]:
            mdir = model_directory.format(p[0], 'MV')
        else:
            mdir = model_directory.format(p[0], 'SV')

        print("Rank {}, Size {}, Process {}, Channels {}, Multivariate {}".format(rank, size, name, p[0], p[1]))
        train_net.train_net(model_path=mdir,
                        lr=best_params['lr'],
                        batch_size=best_params['batch_size'],
                        n_channels=p[0],
                        example_directory = example_directory.format(p[0]),
                        epochs=epochs,
                        multivariate=p[1],
                        lambda_w=best_params['w'],
                        lambda_s=best_params['s'])
