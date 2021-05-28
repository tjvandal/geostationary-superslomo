import os, sys
import datetime
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

import matplotlib
matplotlib.rc('text', usetex=True)

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sklearn.metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from goesr import GOESL1b, L1bBand, GroupBandTemporal
import interpolate
from networks import unet
import dataloader



'''
    Storm 1
    bands = [13,]
    year = 2017
    month = 3
    days = [6,7]
    spatial = 'RadM1'
    product = 'ABI-L1b-RadM'
    T = 10
    lat = 40
    lon = -95
'''

def main():
    ''' Plot mesoscale given band, datetime'''
    '''
    band = 13
    year = 2017
    month = 3
    days = [6,7]
    spatial = 'RadM1'
    product = 'ABI-L1b-RadM'
    T = 10
    lat = 40
    lon = -95
    '''

    band = 13
    year = 2019
    month = 5
    days = [23,]
    spatial = 'RadM1'
    product = 'ABI-L1b-RadM'
    T = 10    
    lat = 37
    lon = -95

    mu, std = dataloader.get_band_stats(band)
    
    time_min = datetime.datetime(year, month, days[0], hour=2)
    time_max= datetime.datetime(year, month, days[0], hour=3)

    checkpoint = '/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/V2/interp-ind/Channel-%02i/best.flownet.pth.tar' % band
    #checkpoint = '/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/slomo-mt/allbands-unetmedium/best.flownet.pth.tar'

    doy = lambda y, m, d: datetime.datetime(y, m, d).timetuple().tm_yday

    bands = [band,]
    geo = GOESL1b(channels=bands, product=product)
    files = [geo.local_files(year, doy(year, month, d), hour=None) for d in days]
    files = pd.concat(files)
    files = files.xs(spatial, level='spatial')

    group = GroupBandTemporal([])
    group_10m = GroupBandTemporal([])
    for index, row in files.iterrows():
        d = datetime.datetime(index[0], 1, 1, hour=index[2], minute=index[3])
        d += datetime.timedelta(days=index[1]-1)
        if (d < time_min) or (d > time_max):
            continue
        group.add(L1bBand(row.values[0]))
        minute = index[3]
        if minute % 10 == 0:
            group_10m.add(L1bBand(row.values[0]))

    times, ts = group.timeseries_latlon(lat, lon)
    times_10, ts_10 = group_10m.timeseries_latlon(lat, lon)

    interp = interpolate.Interpolate(checkpoint)
    ts_interp = []
    times_interp = []
    for i in range(1, len(group_10m)):
        data = group_10m.get_radiances(indices=[i-1,i])
        if data is None:
            continue
        t0 = group_10m.data[i-1].datetime
        x1 = data.values[:1]
        x2 = data.values[1:]
        indices = group_10m.data[i-1].latlon_lookup(lat, lon)
        if indices is None:
            continueget_band_

        ix, iy = indices
        times_interp.append(t0)
        ts_interp.append(x1[0,iy,ix])
        for t in range(1,T):
            y = interp.predict((x1-mu)/std, (x2-mu)/std, t/T) *  std + mu
            ts_interp.append(y[iy,ix])
            times_interp.append(t0 + datetime.timedelta(minutes=t))

    fig, ax = plt.subplots(figsize=(9,4))
    
    
    r2 = sklearn.metrics.r2_score(ts[:len(ts_interp)], ts_interp)
    
    
    ax.plot(times, ts, color='black', label='1 Minute', marker='.', markersize=3., alpha=1.0)
    ax.plot(times_10, ts_10, color='black', label='10 Minute (Linear)', ls='--', marker='.', markersize=3.)
    ax.plot(times_interp, ts_interp, color='green', label=f'SSM-T{band}', marker='d', markersize=3., alpha=0.8)
    
    ax.text(0.95, 0.05, f'R-Squared: {r2:0.3f}',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes)

    ax.format_xdata = mdates.DateFormatter('%m-%d %H:%M')
    fig.autofmt_xdate()
    
    plt.legend()
    #plt.xlabel("Datetime")
    plt.ylabel("Brightness Temperature (K)")
    plt.title("Interpolation of Band %02i\nLat: {:.1f}, Lon: {:.1f}".format(lat, lon) % band)
    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")
    plt.savefig("figures/band%02i_%04i%02i%02i.pdf" % (band, year, month, days[0]), bbox='tight', pad_inches=0.0)
    plt.show()


if __name__ == "__main__":
    main()
