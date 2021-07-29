import os, sys
from mpi4py import MPI

import numpy as np
import dask

from training_data import training_dataset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

bands = range(1,17)
product = 'ABI-L1b-RadM'
zarr_path = f'training-data/15min-264x264-Large'
data_directory = '/nex/datapool/geonex/public/GOES16/NOAA-L1B/'

year = 2017
days = np.arange(1, 365, 5)
hours = np.arange(0, 24, 2)

groups = [(d, h, b) for b in bands for h in hours for d in days]

if rank == 0:
    print(f"number of groups: {len(groups)}")

for i, (d, h, b) in enumerate(groups):
    if i % size == rank:
        path = os.path.join(zarr_path, 'Channel-%02i' % b)
        training_dataset(path, b, [year], [d], [h],
                         product=product, n_times=15, cpu_count=1,
                         data_directory=data_directory,
                         patch_size=264)

