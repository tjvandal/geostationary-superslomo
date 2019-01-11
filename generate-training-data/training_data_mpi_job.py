#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI
import goes16s3
import boto3
import psutil

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
group = comm.Get_group()

exclude_ranks = [] #i for i in range(size) if i % 2 != 0]
newgroup = group.Excl(exclude_ranks)
newcomm = comm.Create(newgroup)


BUCKET_NAME = 'nex-goes-slowmo'

noaas3 = goes16s3.NOAAGOESS3()
year_day_pairs = noaas3.year_day_pairs()
#year_day_pairs = year_day_pairs[60:]

goespytorch = goes16s3.GOESDatasetS3(s3_base_path='training-bands-312')

bands = [3, 1, 2] #range(1,17)
mode = 'train'

if rank == 0:
    pairs_to_process = []
    for year, day in year_day_pairs:
        if not goespytorch.in_s3(year, day, mode):
            pairs_to_process.append([year, day])
else:
    pairs_to_process = None

pairs_to_process = comm.bcast(pairs_to_process, root=0)

if newcomm == MPI.COMM_NULL:
    print("skipping rank", rank)
    pass
else:
    for i, (year, day) in enumerate(pairs_to_process):
        if i % newcomm.Get_size() == newcomm.Get_rank(): 
            [os.remove(os.path.join('/tmp', f)) for f in os.listdir('/tmp') if (f[-3:] == '.nc') 
                                                                            and ('%4%03' in f)]

            print("rank", rank, "year", year, "day", day, "group rank", newcomm.Get_rank(),
                   "processor", MPI.Get_processor_name())
            try:
                goespytorch.write_example_blocks_to_s3(year, day, channels=bands)
            except ValueError:
                print("ValueError: year", year, "day", day)
                #break

