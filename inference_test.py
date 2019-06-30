import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from torch.utils import data

from data import goes16s3
from tools import inference_tools

def _linear_interpolation(X0, X1, t):
    diff = X1 - X0
    return X0 + t * diff

def linear_interpolation(X0, X1, ts):
    pframes = [X0]
    for t in ts:
        _linear_interpolation(X0, X1, t)
        pframes.append(_linear_interpolation(X0, X1, t))
    pframes.append(X1)
    pframes = [frame.values[np.newaxis] for frame in pframes]
    pframes = inference_tools.block_predictions_to_dataarray(pframes, X0)
    return pframes

def time_rmse(x1, x2):
    diff = np.square(x1 - x2)
    return diff.mean(['x', 'y',])**0.5

def inference_day(year, dayofyear, n_minutes=15, n_channels=3):
    data_directory = '/nobackupp10/tvandal/data/goes16'
    inference_dir = 'data/v1-inference/%iChannel-%iminute' % (n_channels, n_minutes)

    # Initialize data reader
    channels = list(range(1,n_channels+1))
    noaadata = goes16s3.NOAAGOESS3(product='ABI-L1b-RadM', channels=channels,
                         save_directory=data_directory)

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    # Read checkpoints and models
    checkpoint_sv = './saved-models/v1/9Min-%iChannels-SV/' % n_channels
    checkpoint_mv = './saved-models/v1/9Min-%iChannels-MV/' % n_channels

    flownetsv, interpnetsv, warpersv = inference_tools.load_models(n_channels, checkpoint_sv,
                                                             multivariate=False)
    flownetmv, interpnetmv, warpermv = inference_tools.load_models(n_channels, checkpoint_mv,
                                                             multivariate=True)
    print("Day {}".format(dayofyear))

    # Get an iterator to loop over every n_minutes of the day
    iterator = noaadata.iterate_day(year, dayofyear, max_queue_size=n_minutes+1, min_queue_size=1)

    ts = np.linspace(1./n_minutes, 1-1./n_minutes, n_minutes-1 )

    for i, example in enumerate(iterator):
        # Select X0 and I1
        X0 = example.isel(t=0)
        X1 = example.isel(t=-1)

        # Single variate inference
        ressv = inference_tools.inference(X0, X1, flownetsv, interpnetsv, warpersv,
                                          multivariate=False, T=n_minutes-1)
        # multivariate inference
        resmv = inference_tools.inference(X0, X1, flownetmv, interpnetmv, warpermv,
                                          multivariate=True,  T=n_minutes-1)
        # linear interpolation estimate
        linearres = linear_interpolation(X0, X1, ts)

        # put results in xarray dataset
        ds = xr.Dataset({'slomo_sv': ressv, 'slomo_mv': resmv, 'linear': linearres, 'observed': example})
        ds['sv_rmse'] = time_rmse(ressv, example)
        ds['mv_rmse'] = time_rmse(resmv, example)
        ds['linear_rmse'] = time_rmse(linearres, example)

        # save to disk
        fpath = os.path.join(inference_dir, '%4i_%03i_Example-%03i.nc' % (year, dayofyear, i))
        if os.path.exists(fpath):
            os.remove(fpath)
        ds.to_netcdf(fpath)
        if i % 10 == 0:
            print('Day {}, Example {}'.format(dayofyear, i))

if __name__ == '__main__':
    # Set variables
    n_channels = 3
    n_minutes = 15
    year = 2019
    days = np.arange(1, 180, 5)

    # preform inference for each day
    inference_day(year, days[0], n_minutes=n_minutes, n_channels=n_channels)
