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
#bands = range(1,2)
#bands = [1,]
lr = 0.0001
iterations = 20000
epochs = 1000


models = [('unet-medium', 'interp-ind'),
          ('unet-multiscale', 'interp-ind-ms'),
          #('unet-multiscalev2', 'interp-ind-ms-large'),
         ]
pairs = [(b, m[0], m[1]) for b in bands for m in models]

file_dir = os.path.dirname(os.path.abspath(__file__))
base_model_path = os.path.join(os.path.dirname(file_dir), 'models/V2')

for i, (b, m, n) in enumerate(pairs):
    if i % size == rank:
        data_path = os.path.join(os.path.dirname(file_dir),
                        'training-data/15min-264x264-Large/Channel-%02i' % b)

        model_path = os.path.join(base_model_path, n, 'Channel-%02i' % b)
        mu, sd = dataloader.get_band_stats(b)
        dataset = dataloader.InterpLoader(data_path, patch_size=256, mean=mu, std=sd)

        print("-------------Training Model--------------")
        print("Data: {}".format(data_path))
        print("Model: {}".format(model_path))
        train.train_net(dataset=dataset,
                        lr=lr,
                        batch_size=16, #32,
                        model_path=model_path,
                        epochs=epochs,
                        max_iterations=iterations,
                        lambda_w=0.2,#V1 0.65
                        lambda_s=0.05,#V1 0.23
                        model_name=m,
                        progress=20)
