from mpi4py import MPI

import os, sys
import argparse

import bayesopt_training

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

N_GPUS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4) #','.join([str(v) for v in range(0, N_GPUS)])

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--total_trials", default=50, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--experiment_name", default='bayesopt-mpi-default', type=str)
parser.set_defaults(multivariate=False)
args = parser.parse_args()

exps =[(c, m) for c in [3, 8] for m in [True, False]]

for i, ex in enumerate(exps):
    if i % size == rank:
        print("Rank: {}, Name: {}, Experiment: {}".format(rank, name, ex))
        if ex[1]:
            exp_name = os.path.join(args.experiment_name, '{}channels-MV'.format(ex[0]))
        else:
            exp_name = os.path.join(args.experiment_name, '{}channels-SV'.format(ex[0]))

        bayesopt_training.hyperparameter_optimization(ex[0], ex[1],
                                                      exp_name,
                                                      args.batch_size,
                                                      epochs=args.epochs,
                                                      total_trials=args.total_trials)
