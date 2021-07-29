import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

import argparse
import torch
import numpy as np
import json

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_interpolation as train
import dataloader

def hyperparameter_optimization(experiment_path,
                                batch_size,
                                total_trials,
                                model_name,
                                band=1,
                                epochs=5):

    def train_evaluate(parameterization):
        w = parameterization['w']
        s = parameterization['s']
        lr = 1e-4 #parameterization['lr']

        file_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = '/nobackupp10/tvandal/nex-ai-opticalflow/geo/'\
                    '.tmp/training-data/Interp-ABI-L1b-RadM-15min-264x264-Large/'\
                     'Channel-%02i' % band

        model_path = os.path.join(experiment_path, 'LambdaW_%1.2f-LambdaS_%1.2f' % (w, s))
        mu, sd = dataloader.get_band_stats(band)
        dataset = dataloader.InterpLoader(data_path, patch_size=256, mean=mu, std=sd)
        
        print('Parameterization:', parameterization)
        loss = train.train_net(model_path=model_path,
                          dataset=dataset,
                          epochs=epochs,
                          batch_size=batch_size,
                          max_iterations=20000,
                          lr=lr,
                          lambda_w=w,
                          lambda_s=s,
                          model_name=model_name,
                          occlusion=True,
                          progress=20)
        if not np.isfinite(loss):
            loss = 1e6

        return dict(loss=(-min([loss, 1.]), 0.0))

    best_parameters, values, experiment, model = optimize(
            parameters=[
                        #{"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
                        {"name": "w", "type": "range", "bounds": [1e-2, 1.0]},
                        {"name": "s", "type": "range", "bounds": [1e-2, 1.0]},
                    ],
            evaluation_function=train_evaluate,
            objective_name='loss',
            total_trials=total_trials,
    )

    best_parameters['batch_size'] = batch_size

    print("---------Best Parameters----------")
    print(best_parameters)

    json_params = json.dumps(best_parameters)
    with open(os.path.join(experiment_path, 'best_parameters.json'), 'w') as f:
        f.write(json_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--total_trials", default=20, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--band", default=1, type=int)
    parser.add_argument("--model_name", default='unet-medium', type=str)
    parser.set_defaults(multivariate=False)
    args = parser.parse_args()

    experiment_path = '/nobackupp10/tvandal/nex-ai-opticalflow/geo/'\
                    '.tmp/models/parameter-search-v3/%02i/' % args.band

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(1)
    hyperparameter_optimization(experiment_path,
                                args.batch_size,
                                args.total_trials,
                                args.model_name,
                                band=args.band,
                                epochs=args.epochs)
