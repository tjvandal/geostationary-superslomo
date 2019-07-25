import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

import matplotlib
#matplotlib.use("Agg")

from data import goes16s3
from tools import utils, inference_tools, plotting
from slomo import unet


import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import os

import seaborn as sns

sns.set_context("paper", font_scale=1.6)

year = 2017
month = 9
dayofyear = 251
hour = 18
minute = 2
minute_delta = 15

products = ['ABI-L1b-RadM', 'ABI-L1b-RadC', 'ABI-L1b-RadF']

for p in products:
    print("Product: {}".format(p))
    if p == 'ABI-L1b-RadM':
        data_directory = '/nobackupp10/tvandal/data/goes16'
    else:
        data_directory = '/nex/datapoolne/goes16'

    noaadata = goes16s3.NOAAGOESS3(product=p, channels=range(1,4),
                              save_directory=data_directory, skip_connection=True)
    files = noaadata.local_files(year=year, dayofyear=dayofyear)
    print(files.head())

    I0, I1 = noaadata.load_snapshots(year, dayofyear, hour, minute, minute_delta=minute_delta)

    ax = plotting.plot_3channel_image(I0.values)
    plt.savefig("figures/goes-scales-{}.png".format(p), dpi=200, pad_inches=0)
    plt.close()
    #plt.show()

