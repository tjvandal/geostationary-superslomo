import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from data import goes16s3
from tools import utils, inference_tools, plotting
from slomo import unet

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import os
from PIL import Image

import torch

import cv2
import pandas as pd

import seaborn as sns

sns.set_context("paper", font_scale=1.6)

#dayofyear = 281
#year = 2017
#dayofyear = dt.datetime(year, 9, 6).timetuple().tm_yday

year = 2017
month = 9
day = 8
n_channels = 8
t = 0.5
#product = 'ABI-L1b-RadC'
#data_directory = '/nex/datapoolne/goes16'
product = 'ABI-L1b-RadM'
data_directory = '/nobackupp10/tvandal/data/goes16'
hour = 18
minute = 2
minute_delta = 15
nn_model = unet.UNetMedium

discard = 64



dayofyear = dt.datetime(year, month, day).timetuple().tm_yday
multivariate = True
if multivariate:
    checkpoint = '../saved-models/1.3-unet-medium/9Min-%iChannels-MV/' % n_channels
else:
    checkpoint = '../saved-models/1.3-unet-medium/9Min-%iChannels-SV/' % n_channels


noaadata = goes16s3.NOAAGOESS3(product=product, channels=range(1,n_channels+1),
                              save_directory=data_directory, skip_connection=True)
I0, I1 = noaadata.load_snapshots(year, dayofyear, hour, minute, minute_delta=minute_delta)

print("I0: {}".format(I0.shape))

if not os.path.exists('figures/network'):
    os.makedirs('figures/network')

flownet, interpnet, warper= inference_tools.load_models(n_channels, checkpoint,
                                                         multivariate=multivariate,
                                                         nn_model=nn_model)
vector_data = inference_tools.single_inference_split(I0.values, I1.values, t,
                                                     flownet, interpnet, multivariate,
                                                     overlap=128, block_size=256+128,
                                                     discard=discard)

print("vector data keys: {}".format(vector_data.keys()))

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
plotting.plot_3channel_image(I1.values[:,discard:-discard, discard:-discard], ax=ax)
plt.tight_layout()
plt.savefig("figures/falsergb_image1.png", dpi=300, pad_inches=0)

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
plotting.plot_3channel_image((I1-I0).values[:,discard:-discard, discard:-discard]*2, ax=ax)
plt.tight_layout()
plt.savefig("figures/diff_images.png", dpi=300, pad_inches=0)

f_01 = vector_data['f_01']

total_flow = f_01 + vector_data['delta_f_t1']

for c in [7,]:
    u = total_flow[2*c]
    v = total_flow[2*c+1]
    ax = plotting.flow_quiver_plot(u, v)
    plt.savefig("figures/quiver_plot_band{}.png".format(c+1), dpi=300, pad_inches=0)

    visible = vector_data['V_t0'][c]
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(visible, cmap='Greys')
    ax.axis('off')
    plt.savefig("figures/visible_{}.png".format(c), dpi=300, pad_inches=0)

plt.show()

