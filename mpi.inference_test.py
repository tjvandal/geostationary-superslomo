from mpi4py import MPI
import os, sys

import numpy as np

import inference_test


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

# restrict GPU access
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)

# Set variables
n_channels = 8
n_minutes = 15
year = 2019
days = np.arange(1, 180, 5)

## preform inference for each day
#inference_day(year, days[0], n_minutes=n_minutes, n_channels=n_channels)
for i, d in enumerate(days):
    if i % size == rank:
        print("Rank: {}, Processor: {}, Day: {}".format(rank, name, d))
        inference_test.inference_day(year, d, n_minutes=n_minutes, n_channels=n_channels)
