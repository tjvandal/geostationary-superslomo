import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from torch.utils import data

from data import goes16s3
from tools import inference_tools


n_channels = 3
n_minutes = 15
year = 2019

channels = list(range(1,n_channels+1))

data_directory = '/nobackupp10/tvandal/data/goes16'
noaadata = goes16s3.NOAAGOESS3(product='ABI-L1b-RadM', channels=channels,
                     save_directory=data_directory)
inference_dir = 'data/test-inference/%iChannel-%iminute' % (len(channels), n_minutes)

if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)

localfiles = noaadata.local_files(year=year)
days = localfiles.index.unique(level='dayofyear')
localfiles


checkpoint_sv = './saved-models/v1/9Min-%iChannels-SV/' % n_channels
checkpoint_mv = './saved-models/v1/9Min-%iChannels-MV/' % n_channels

flownetsv, interpnetsv, warpersv = inference_tools.load_models(n_channels, checkpoint_sv,
                                                         multivariate=False)
flownetmv, interpnetmv, warpermv = inference_tools.load_models(n_channels, checkpoint_mv,
                                                         multivariate=True)


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

for day in days:
    print("Day {}".format(day))
    iterator = noaadata.iterate_day(year, day, max_queue_size=n_minutes+1, min_queue_size=1)

    #ts = np.linspace(0.1, 0.9, 9)
    ts = np.linspace(1./n_minutes, 1-1./n_minutes, n_minutes-1 )

    for i, example in enumerate(iterator):
        X0 = example.isel(t=0)
        X1 = example.isel(t=-1)
        ressv = inference_tools.inference(X0, X1, flownetsv, interpnetsv, warpersv, 
                                          multivariate=False, T=n_minutes-1)    

        resmv = inference_tools.inference(X0, X1, flownetmv, interpnetmv, warpermv, 
                                          multivariate=True,  T=n_minutes-1)   

        linearres = linear_interpolation(X0, X1, ts)

        ds = xr.Dataset({'slomo_sv': ressv, 'slomo_mv': resmv, 'linear': linearres, 'observed': example})
        ds['sv_rmse'] = time_rmse(ressv, example)
        ds['mv_rmse'] = time_rmse(resmv, example)
        ds['linear_rmse'] = time_rmse(linearres, example)

        fpath = os.path.join(inference_dir, '%4i_%03i_Example-%03i.nc' % (year, day, i))
        if os.path.exists(fpath):
            os.remove(fpath)
        ds.to_netcdf(fpath)


print(sorted(os.listdir(inference_dir)))


