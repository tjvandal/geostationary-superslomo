import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_interpolation as train
import dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' #str(rank % N_GPUS)

lr = 0.0001
max_iterations = 20000
epochs = 1000

file_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(file_dir),
                '.tmp/training-data/15min-264x264-Large/Channel-%02i')
model_path = os.path.join(os.path.dirname(file_dir),
                '.tmp/models/V2/global')

bands = range(1,17)
#bands = range(1,2)
data_paths = [data_path % b for b in bands]
dataset = dataloader.InterpLoaderMultitask(data_paths, bands, patch_size=256)

print("-------------Training Model--------------")
print("Data: {}".format(data_path))
print("Model: {}".format(model_path))

train.train_net(dataset=dataset,
                lr=lr,
                batch_size=32,
                epochs=epochs,
                model_path=model_path,
                max_iterations=max_iterations,
                model_name='unet-medium',
                progress=20,
                lambda_w=0.2,#V1 0.65
                lambda_s=0.06#V1 0.23
                )
