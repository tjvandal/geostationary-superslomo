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

model_name = 'unet-medium'
version = '1.4.1-{}'.format(model_name)
example_directory = '/nobackupp10/tvandal/GOES-SloMo/data/training/9Min-{}Channels-Train-pt'
model_directory = 'saved-models/{}'.format(version) + '/9Min-{}Channels-{}'

# training parameters
N_GPUS = 4
epochs = 100

# these parameters are selected with consensus from bayesian optimization runs
best_params = {(3, False): {"lr": 0.0001, "w": 0.01, "s": 1.3, "batch_size": 128},
               (3, True): {"lr": 0.0001, "w": 0.01, "s": 0.03, "batch_size": 128},
               (8, False): {"lr": 0.0001, "w": 0.01, "s": 0.30, "batch_size": 128},
               (8, True): {"lr": 0.0001, "w": 0.01, "s": 0.01, "batch_size": 128}}

#param_list = [(c, m) for c in [3, 8] for m in [True, False]]
param_list = [(8,  True)]

# set GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in range(0, N_GPUS)])

for i, p in enumerate(param_list):
    if i % size == rank:
        if p[1]:
            mdir = model_directory.format(p[0], 'MV')
        else:
            mdir = model_directory.format(p[0], 'SV')

        print("Rank {}, Size {}, Process {}, Channels {}, Multivariate {}".format(rank, size, name, p[0], p[1]))
        train_net.train_net(model_path=mdir,
                        lr=best_params[p]['lr'],
                        batch_size=best_params[p]['batch_size'],
                        n_channels=p[0],
                        example_directory = example_directory.format(p[0]),
                        epochs=epochs,
                        multivariate=p[1],
                        lambda_w=best_params[p]['w'],
                        lambda_s=best_params[p]['s'],
                        model_name=model_name)
