import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from data import goes16s3
from tools import utils, inference_tools, plotting
from slomo import unet

import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import os

import seaborn as sns

sns.set_context("paper", font_scale=1.6)

year = 2017
month = 9
day = 8
n_channels = 8
t = 1.0
product = 'ABI-L1b-RadC'
data_directory = '/nex/datapoolne/goes16'
#product = 'ABI-L1b-RadM'
#data_directory = '/nobackupp10/tvandal/data/goes16'
zoom=False

nn_model = unet.UNetMedium
discard = 64

dayofyear = dt.datetime(year, month, day).timetuple().tm_yday
multivariate = True
checkpoint = '../saved-models/1.4.1-unet-medium/9Min-%iChannels-MV/' % n_channels


if product == 'ABI-L1b-RadC':
    down = 20
    frame_directory = 'figures/animation-conus'
    min_hour = 15
else:
    down = 7
    frame_directory = 'figures/animation-mesoscale'
    min_hour = 16
    if zoom:
        frame_directory += '-zoom'

if not os.path.exists(frame_directory):
    os.makedirs(frame_directory)

flownet, interpnet, warper= inference_tools.load_models(n_channels, checkpoint,
                                                         multivariate=multivariate,
                                                         nn_model=nn_model)

noaadata = goes16s3.NOAAGOESS3(product=product, channels=range(1,n_channels+1),
                              save_directory=data_directory, skip_connection=True)

channel_idxs = [c-1 for c in noaadata.channels]
files = noaadata.local_files(year=year, dayofyear=dayofyear)
files = files.dropna()
counter = 0
I0 = None
for j, row in enumerate(files.values):
    # (2017, 251, 0, 2, 168, 'RadC')

    year, dayofyear, hour, minute, dsecond, spatial = files.iloc[j].name
    if (product == 'ABI-L1b-RadM') and (spatial != 'RadM1'):
        continue

    if (product == 'ABI-L1b-RadM') and (minute % 5 != 0):
        continue

    if (hour < min_hour) or (hour > 22):
        I0 = None
        continue

    if I0 is None:
        I0 = goes16s3._open_and_merge_2km(row[channel_idxs])
        continue

    print("Frame: {}".format(counter))

    timestamp = dt.datetime(year, 1, 1, hour, minute) + dt.timedelta(days=dayofyear-1)
    I1 = goes16s3._open_and_merge_2km(row[channel_idxs])

    vector_data = inference_tools.single_inference_split(I0.values, I1.values, 1.,
                                                         flownet, interpnet, multivariate,
                                                         overlap=128, block_size=256+128,
                                                         discard=discard)
    total_flow = vector_data['f_01'] + vector_data['delta_f_t1']
    c = 0
    u = total_flow[2*c] * -1.
    v = total_flow[2*c+1]

    RGB = I0.values[[1,2,0]]
    RGB = np.transpose(RGB, (1,2,0))[discard:-discard,discard:-discard]
    #RGB = I0.values[7][discard:-discard,discard:-discard]

    if zoom:
        u = u[180:250, 180:250]
        v = v[180:250, 180:250]
        RGB = RGB[180:250, 180:250]
        down = 2

    ax = plotting.flow_quiver_plot(u, v, down=down, vmax=0.60, background_img=RGB)
    ax.text(0.07*total_flow.shape[0], 0.95*total_flow.shape[1], timestamp, fontsize=14,
            color='white')
    plt.savefig("{}/quiver_plot_band{}-{}-{:03d}.png".format(frame_directory, c+1, product, counter), dpi=300, pad_inches=0)
    plt.show()
    plt.close()

    ax = plotting.plot_3channel_image(I0.values)
    plt.savefig("{}/rbg-{:03d}.png".format(frame_directory, counter), dpi=300, pad_inches=0)
    plt.close()

    I0 = I1
    counter += 1
