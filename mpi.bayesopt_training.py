from mpi4py import MPI

import os, sys
import argparse

import bayesopt_training

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


N_GPUS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in range(0, N_GPUS)])


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--total_trials", default=50, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--experiment_name", default='bayesopt-default', type=str)
parser.set_defaults(multivariate=False)
args = parser.parse_args()

exps =[(c, m) for c in [3, 8] for m in [True, False]]

for i, ex in enumerate(exps):
    if i % size == rank:
        print("Rank: {}, Name: {}, Experimernt: {}".format(rank, name, ex))
        bayesopt_training.hyperparameter_optimization(ex[0],ex[1],
                                                      args.experiment_name,
                                                      args.batch_size,
                                                      epochs=args.epochs,
                                                      total_trials=args.total_trials)

