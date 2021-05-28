import os, sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

import numpy as np
import pandas as pd
import comparison

N_GPUS = 4

os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % N_GPUS)

TO_FILE = 'comparison_statistics_all.pkl'

TMP_DIR = '.tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

N = 200
T = 10
year = 2019
trange = np.linspace(0.1, 0.9, 9)
bands = range(1,17)
#bands = [13,]
pairs = comparison.random_dayhours(N, year)
jobs = []
root = 0

stats = []
procs = [(t, p[0], p[1], b) for p in pairs for b in bands for t in trange]
for i, (t, day, hour, band) in enumerate(procs):
    if i % size == rank:
        try:
            res = comparison.example_statistics(year, day, hour, band, T=T, t=t)
            stats.append(res)
        except:
            print(f"ERROR - Band: {band}, t={t}, day={day}, hour={hour}")

stats = pd.concat(stats)
stats.to_pickle(os.path.join(TMP_DIR, f'stats_{rank}.pkl'))
send_stats = comm.gather(stats, root=root)

if rank == root:
    gather_stats = pd.concat(send_stats)
    gather_stats.to_pickle(TO_FILE)
    gather_stats.to_csv(TO_FILE + '.csv', index=False)
    print(gather_stats.head())
else:
    gather_stats = None

