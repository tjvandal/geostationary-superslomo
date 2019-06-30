import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from data import goes16s3
from tools import utils, inference_tools, plotting

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import os
from PIL import Image

import torch

import cv2
import pandas as pd

#dayofyear = 281
#year = 2017
#dayofyear = dt.datetime(year, 9, 6).timetuple().tm_yday

year = 2017
month = 9
day = 8
n_channels = 8
t = 1.0
product = 'ABI-L1b-RadC'
data_directory = '/nex/datapoolne/goes16'
#data_directory = '/nobackupp10/tvandal/data/goes16'
hour = 18
minute = 2
minute_delta = 15


def lowres_flow(x, pool=25):
    x = torch.from_numpy(x[np.newaxis])
    xlr = torch.nn.AvgPool2d(pool, stride=pool)(x)
    return xlr.detach().numpy()[0]


dayofyear = dt.datetime(year, month, day).timetuple().tm_yday
multivariate = True
if multivariate:
    checkpoint = '../saved-models/v1/9Min-%iChannels-MV/' % n_channels
else:
    checkpoint = '../saved-models/v1/9Min-%iChannels-SV/' % n_channels


noaadata = goes16s3.NOAAGOESS3(product=product, channels=range(1,n_channels+1),
                              save_directory=data_directory, skip_connection=True)
I0, I1 = noaadata.load_snapshots(year, dayofyear, hour, minute, minute_delta=minute_delta)

if not os.path.exists('figures/network'):
    os.makedirs('figures/network')

flownet, interpnet, warper= inference_tools.load_models(n_channels, checkpoint,
                                                         multivariate=multivariate)
vector_data = inference_tools.single_inference_split(I0.values, I1.values, t,
                                                     flownet, interpnet, multivariate,
                                                    overlap=64)

plotting.plot_3channel_image(I0.values)

f_01 = vector_data['f_01']
f_10 = vector_data['f_10']

f_01_lr = lowres_flow(f_01)
f_10_lr = lowres_flow(f_10)

total_flow = f_01 + vector_data['delta_f_t1']
total_flow_lr = lowres_flow(total_flow, 25)

for c in range(n_channels):
    u = total_flow_lr[2*c]
    v = total_flow_lr[2*c+1]
    plotting.flow_quiver_plot(u, v, title='Band {}'.format(c+1))

plt.show()
