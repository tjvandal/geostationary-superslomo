import os, sys
from mpi4py import MPI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_interpolation as train
import dataloader

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

N_GPUS = 4

os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % N_GPUS)
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

bands = range(1,17)
#bands = [1,]
lr = 0.0001
iterations = 50000
epochs = 1000

for i, b in enumerate(bands):
    if i % size == rank:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(file_dir),
                        '.tmp/training-data/Interp-ABI-L1b-RadM-15min-264x264/Channel-%02i' % b)
        model_path = os.path.join(os.path.dirname(file_dir),
                        '.tmp/models/slomo-ind/v0.5/Channel-%02i' % b)

        mu, sd = dataloader.get_band_stats(b)
        dataset = dataloader.InterpLoader(data_path, patch_size=256, mean=mu, std=sd)

        print("-------------Training Model--------------")
        print("Data: {}".format(data_path))
        print("Model: {}".format(model_path))

        train.train_net(dataset=dataset,
                        lr=lr,
                        batch_size=32,
                        model_path=model_path,
                        epochs=epochs,
                        max_iterations=iterations,
                        lambda_w=0.65,
                        lambda_s=0.23,
                        model_name='unet-medium',
                        progress=20)
