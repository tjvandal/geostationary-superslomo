from mpi4py import MPI
import os, sys
import numpy as np

import numpy as np

import inference_test

import datetime as dt 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

# restrict GPU access
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)

# Set variables
#n_channels = [3, 8]
n_channels = [8,]

n_minutes = 15
year = 2019
days = np.arange(1, 180, 5)
data_directory = '/nobackupp10/tvandal/data/goes16'

#year = 2017
#days = [dt.datetime(year, 9, 8).timetuple().tm_yday]

pairs = [(c, d) for c in n_channels for d in days]
## preform inference for each day
#inference_day(year, days[0], n_minutes=n_minutes, n_channels=n_channels)
for i, (c, d) in enumerate(pairs):
    if i % size == rank:
        inference_dir = 'data/v1.4.1-inference/%iChannel-%iminute' % (c, n_minutes)
        print("Rank: {}, Processor: {}, Day: {}".format(rank, name, d))
        inference_test.inference_day(year, d, inference_dir, n_minutes=n_minutes, n_channels=c)
