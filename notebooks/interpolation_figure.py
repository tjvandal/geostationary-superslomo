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

import seaborn as sns
import xarray as xr

import metpy

sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from tools import plotting

sns.set_context("paper", font_scale=4.0)

#dayofyear = 281
#year = 2017
#dayofyear = dt.datetime(year, 9, 6).timetuple().tm_yday


# Read checkpoints and models
results_directory = '../data/v1.4.1-inference-irma/8Channel-15minute'

#f = '2019_051_Example-081.nc'
f = '2017_251_Example-042.nc'
results_file = os.path.join(results_directory, f)

figure_dir = 'figures/slomo-interp/'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

ds = xr.open_dataset(results_file)

H = ds.y.shape[0]
W = ds.x.shape[0]

'''
observed     (t, band, y, x) float64 ...
slomo_sv     (t, band, y, x) float64 ...
linear       (t, band, y, x) float64 ...
slomo_mv     (t, band, y, x) float64 ...
sv_rmse      (t, band) float64 ...
mv_rmse      (t, band) float64 ...
linear_rmse  (t, band) float64 ...
'''

ds = ds.isel(x=range(300,500), y=np.arange(300,500))

# plot I0 and I1
plotting.plot_3channel_image(ds['observed'].isel(t=0).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'I0.png'), dpi=300, pad_inches=0)

plotting.plot_3channel_image(ds['observed'].isel(t=15).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'I1.png'), dpi=300, pad_inches=0)

plotting.plot_3channel_image(ds['observed'].isel(t=7).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'IT.png'), dpi=300, pad_inches=0)

plotting.plot_3channel_image(ds['observed'].isel(t=7).values[:,::-1] -
                             ds['observed'].isel(t=0).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'IT-minus-I1.png'), dpi=300, pad_inches=0)

#ax = plotting.plot_3channel_image((ds['observed'].isel(t=-1)-ds['observed'].isel(t=0)).values)
#plt.show()

ds_t = ds.isel(t=7)

ax = plotting.plot_3channel_image(ds['linear'].isel(t=7).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'IT-Linear.png'), dpi=300, pad_inches=0)

ax = plotting.plot_3channel_image(ds['slomo_sv'].isel(t=7).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'IT-sv.png'), dpi=300, pad_inches=0)

ax = plotting.plot_3channel_image(ds['slomo_mv'].isel(t=7).values[:,::-1])
plt.savefig(os.path.join(figure_dir, 'IT-mv.png'), dpi=300, pad_inches=0)



lv_error = (ds_t['linear'] - ds_t['observed'])**2
lv_error = lv_error.mean('band')**0.5
vmax = lv_error.max().values
ax = plotting.plot_1channel_image(lv_error.values[::-1], vmax=vmax)
ax.text(0.05, 0.92, "RMSE: {:.3g}".format(ds.isel(t=7)['linear_rmse'].mean().values),
       transform=ax.transAxes, color='white')
plt.savefig(os.path.join(figure_dir, 'IT-linear_rmse.png'), dpi=300, pad_inches=0)


sv_error = (ds_t['slomo_sv'] - ds_t['observed'])**2
sv_error = sv_error.mean('band')**0.5
ax = plotting.plot_1channel_image(sv_error.values[::-1], vmax=vmax)
ax.text(0.05, 0.92, "RMSE: {:.3g}".format(ds.isel(t=7)['sv_rmse'].mean().values),
       transform=ax.transAxes, color='white')
plt.savefig(os.path.join(figure_dir, 'IT-sv_rmse.png'), dpi=300, pad_inches=0)


mv_error = (ds_t['slomo_mv'] - ds_t['observed'])**2
mv_error = mv_error.mean('band')**0.5
ax = plotting.plot_1channel_image(mv_error.values[::-1], vmax=vmax)
ax.text(0.05, 0.92, "RMSE: {:.3g}".format(ds.isel(t=7)['mv_rmse'].mean().values),
       transform=ax.transAxes, color='white')
plt.savefig(os.path.join(figure_dir, 'IT-mv_rmse.png'), dpi=300, pad_inches=0)

