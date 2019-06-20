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

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0,1,2,3", type=str)
parser.add_argument("--multivariate", dest='multivariate', action='store_true')
parser.add_argument("--n_channels", default=3, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--total_trials", default=2, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--experiment_name", default='bayesopt-default', type=str)
parser.set_defaults(multivariate=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

torch.manual_seed(1)

def train_evaluate(parameterization):
    w = parameterization['w']
    s = parameterization['s']
    lr = parameterization['lr']
    c = args.n_channels

    if args.multivariate:
        exp_name = '9min-%iChannels-MV' % c
    else:
        exp_name = '9min-%iChannels-SV' % c
    exp_name = 'LambdaW_%1.2f-LambdaS_%1.2f' % (w, s)

    example_directory = '/nobackupp10/tvandal/GOES-SloMo/data/training/9Min-%iChannels-Train-pt'
    #example_directory = '/nobackupp10/tvandal/GOES-SloMo/data/training/9Min-%iChannels-Train-small'
    model_directory = os.path.join('saved-models/', args.experiment_name, exp_name)

    print('Parameterization:', parameterization)

    loss = train_net.train_net(model_path=model_directory,
                      lr=lr,
                      batch_size=args.batch_size,
                      n_channels=c,
                      example_directory=example_directory % c,
                      epochs=args.epochs,
                      multivariate=args.multivariate,
                      lambda_w=w,
                      lambda_s=s)
    if not np.isfinite(loss):
        loss = 1e6

    return dict(loss=(min([loss, 1.]), 0.0))

best_parameters, values, experiment, model = optimize(
        parameters=[
                    {"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
                    {"name": "w", "type": "range", "bounds": [1e-6, 2.0]},
                    {"name": "s", "type": "range", "bounds": [1e-6, 2.0]},
                ],
        evaluation_function=train_evaluate,
        objective_name='loss',
        total_trials=args.total_trials,
)



best_parameters['n_channels'] = args.n_channels
best_parameters['batch_size'] = args.batch_size

print("---------Best Parameters----------")
print(best_parameters)

json_params = json.dumps(best_parameters)
with open(os.path.join('saved-models', args.experiment_name, 'best_parameters.json'), 'w') as f:
    f.write(json_params)

