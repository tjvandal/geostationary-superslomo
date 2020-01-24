import argparse
import torch
import numpy as np
import os
import json

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate

import train_net

def hyperparameter_optimization(n_channels,
                                multivariate,
                                experiment_name,
                                batch_size,
                                epochs,
                                total_trials,
                                model_name):

    def train_evaluate(parameterization):
        w = parameterization['w']
        s = parameterization['s']
        lr = 1e-4 #parameterization['lr']
        c = n_channels

        if multivariate:
            exp_name = '9min-%iChannels-MV' % c
        else:
            exp_name = '9min-%iChannels-SV' % c
        exp_name = exp_name + '/LambdaW_%1.2f-LambdaS_%1.2f' % (w, s)

        example_directory = '/nobackupp10/tvandal/GOES-SloMo/data/training/9Min-%iChannels-Train-pt'
        model_directory = os.path.join('saved-models/', experiment_name, exp_name)

        print('Parameterization:', parameterization)
        loss = train_net.train_net(model_path=model_directory,
                          lr=lr,
                          batch_size=batch_size,
                          n_channels=c,
                          example_directory=example_directory % c,
                          epochs=epochs,
                          multivariate=multivariate,
                          lambda_w=w,
                          lambda_s=s,
                          model_name=model_name)
        if not np.isfinite(loss):
            loss = 1e6

        return dict(loss=(-min([loss, 1.]), 0.0))

    best_parameters, values, experiment, model = optimize(
            parameters=[
                        #{"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
                        {"name": "w", "type": "range", "bounds": [1e-2, 2.0]},
                        {"name": "s", "type": "range", "bounds": [1e-2, 2.0]},
                    ],
            evaluation_function=train_evaluate,
            objective_name='loss',
            total_trials=total_trials,
    )

    best_parameters['n_channels'] = n_channels
    best_parameters['batch_size'] = batch_size

    print("---------Best Parameters----------")
    print(best_parameters)

    json_params = json.dumps(best_parameters)
    parameter_directory = os.path.join('saved-models', experiment_name)
    if not os.path.exists(parameter_directory):
        os.makedirs(parameter_directory)

    with open(os.path.join(parameter_directory, 'best_parameters.json'), 'w') as f:
        f.write(json_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", type=str)
    parser.add_argument("--multivariate", dest='multivariate', action='store_true')
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--total_trials", default=2, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--experiment_name", default='bayesopt-default', type=str)
    parser.add_argument("--model_name", default='unet-medium', type=str)
    parser.set_defaults(multivariate=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(1)
    hyperparameter_optimization(args.n_channels,
                                args.multivariate,
                                args.experiment_name,
                                args.batch_size,
                                args.epochs,
                                args.total_trials,
                                args.model_name)
