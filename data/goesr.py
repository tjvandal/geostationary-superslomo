import os, sys
import glob
import xarray as xr
import pandas as pd
import numpy as np
import dask as da
import datetime as dt
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pyproj

from . import utils

def get_filename_metadata(f):
    channel = int(f.split('_')[1][-2:])
    spatial = f.split('-')[2]
    t1 = f.split('_')[3]
    year = int(t1[1:5])
    dayofyear = int(t1[5:8])
    hour = int(t1[8:10])
    minute = int(t1[10:12])
    second = int(t1[12:15])
    return dict(channel=channel, year=year, dayofyear=dayofyear, hour=hour,
                minute=minute, second=second, spatial=spatial)

class L1bBand(object):
    def __init__(self, fpath):
        self.fpath = fpath
        meta = get_filename_metadata(self.fpath)
        self.band = meta['channel']
        self.year = meta['year']
        self.dayofyear = meta['dayofyear']
        self.hour = meta['hour']
        self.minute = meta['minute']
        self.second = meta['second']
        self.spatial = meta['spatial']
        self.datetime = dt.datetime(self.year, 1, 1, self.hour, self.minute, self.second//10) +\
                dt.timedelta(days=self.dayofyear-1)

    def open_dataset(self, rescale=True, force=False):
        if (not hasattr(self, 'data')) or force:
            ds = xr.open_dataset(self.fpath, chunks=500)
            ds = ds.where(ds.DQF.isin([0, 1]))
            band = ds.band_id[0]
            # normalize radiance
            if rescale:
                radiance = ds['Rad']
                if band <= 6:
                    ds['Rad'] = ds['Rad'] * ds.kappa0
                else:
                    fk1 = ds.planck_fk1.values
                    fk2 = ds.planck_fk2.values
                    bc1 = ds.planck_bc1.values
                    bc2 = ds.planck_bc2.values
                    tmp = fk1 / ds["Rad"] + 1
                    tmp = np.where(tmp > 0, tmp, 1)
                    T = (fk2/(np.log(tmp))-bc1)/bc2
                    radiance.values = T
                ds['Rad'] = radiance
            self.data = ds
        return self

    def plot(self, ax=None, cmap=None, norm=None):
        if not hasattr(self, 'data'):
            self.open_dataset()

        # Satellite height
        sat_h = self.data['goes_imager_projection'].perspective_point_height
        # Satellite longitude
        sat_lon = self.data['goes_imager_projection'].longitude_of_projection_origin

        # The geostationary projection
        x = self.data['x'].values * sat_h
        y = self.data['y'].values * sat_h
        if ax is None:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            
        m = Basemap(projection='geos', lon_0=sat_lon, resolution='i',
                     rsphere=(6378137.00,6356752.3142),
                     llcrnrx=x.min(),llcrnry=y.min(),
                     urcrnrx=x.max(),urcrnry=y.max(),
                     ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        ax.set_title('GOES-16 -- Band {}'.format(self.band), fontweight='semibold', loc='left')
        ax.set_title('%s' % self.datetime.strftime('%H:%M UTC %d %B %Y'), loc='right')
        m.imshow(self.data['Rad'].values[::-1], cmap=cmap, norm=norm)
        return m
        
    def plot_infrared(self, ax=None):
        from metpy.plots import colortables
        # Use a colortable/colormap available from MetPy
        ir_norm, ir_cmap = colortables.get_with_range('ir_drgb_r', 190, 350)
        return self.plot(ax, cmap=ir_cmap, norm=ir_norm)
        
    def latlon(self):
        if not hasattr(self, 'lats'):
            self.open_dataset()
            # Satellite height
            sat_h = self.data['goes_imager_projection'].perspective_point_height
            # Satellite longitude
            sat_lon = self.data['goes_imager_projection'].longitude_of_projection_origin
            sat_sweep= self.data['goes_imager_projection'].sweep_angle_axis
            p = pyproj.Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
            X = self.data['x'].values * sat_h
            Y = self.data['y'].values * sat_h
            XX, YY = np.meshgrid(X, Y)
            lons, lats = p(XX, YY, inverse=True)
            self.lats = lats
            self.lons = lons
            NANs = np.isnan(self.data['Rad'].values)
            self.lats[NANs] = np.nan
            self.lons[NANs] = np.nan
        return self.lats, self.lons

    def latlon_lookup(self, lat, lon):
        self.latlon()
        if (lat > self.lats.min()) and (lat < self.lats.max()) and (lon > self.lons.min()) and (lon < self.lons.max()):
            dist = ((self.lats - lat)**2 + (self.lons - lon)**2)**0.5
            ix, iy = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            return ix, iy

class GroupBandTemporal(object):
    def __init__(self, data_list):
        self.data = data_list
        self.open_dataset()

    def open_dataset(self):
        for b in self.data:
            if not hasattr(b, 'data'):
                b.open_dataset()

    def get_radiances(self, indices=None):
        self.open_dataset()
        xequal = np.array_equal(self.data[0].data.x.values,
                       self.data[-1].data.x.values)
        yequal = np.array_equal(self.data[0].data.y.values,
                       self.data[-1].data.y.values)
        if (not xequal) or (not yequal):
            return

        if indices is None:
            indices = range(len(self.data))
        data = xr.concat([self.data[i].data['Rad'] for i in indices], 'time')
        return data.data

    def get_radiance_patches(self, patch_size):
        data = self.get_radiances()
        if data is None:
            return

        data = utils.block_array(data, 2,
                             size=patch_size,
                             stride=patch_size)
        data = utils.block_array(data, 2,
                                 size=patch_size,
                                 stride=patch_size)
        data= data.reshape(-1, len(self.data),
                                patch_size,
                                patch_size)
        return data

    def add(self, band):
        self.data.append(band)

    def __len__(self):
        return len(self.data)

    def timeseries(self, ix, iy):
        self.open_dataset()
        data = np.array([b.data['Rad'].isel(x=ix, y=iy).values for b in self.data])
        x = [b.datetime for b in self.data]
        return x, data

    def timeseries_latlon(self, lat, lon):
        print("Lat: {}, Lon: {}".format(lat, lon))
        self.open_dataset()
        indices = [b.latlon_lookup(lat, lon) for b in self.data]
        data = []
        x = []
        for i, b in enumerate(self.data):
            if indices[i] is None:
                print("Data bounds: ({}, {}), ({}, {})".format(b.lats.min(), b.lats.max(), b.lons.min(),
                                               b.lons.max()))
                continue
            data.append(b.data['Rad'].isel(x=indices[i][0], y=indices[i][1]).values)
            x.append(b.datetime)
        data = np.array(data)
        x = np.array(x)
        return x, data

class GOESL1b(object):
    def __init__(self, product='ABI-L1b-RadF',
                       channels=range(1,17),
                       data_directory='/nex/datapool/geonex/public/GOES16/NOAA-L1B/'):
        self.product = product
        self.channels = channels
        self.data_directory = os.path.join(data_directory, product)

    def local_files(self, year=None, dayofyear=None, hour=None):
        data = []
        base_dir = self.data_directory

        if year is None:
            year = "*"
        else:
            year = '%04i' % year
        if dayofyear is None:
            dayofyear = "*"
        else:
            dayofyear = '%03i' % dayofyear
        if hour is None:
            hour = "*"
        else:
            hour = '%02i' % hour 

        for c in self.channels:
            path_pattern = os.path.join(self.data_directory, year, dayofyear, hour,
                                        'OR_ABI-L1b-*C%02i_*.nc' % c)
            channel_files = glob.glob(path_pattern)
            for f in channel_files:
                meta = get_filename_metadata(os.path.basename(f))
                meta['file'] = f
                data.append(meta)

        data = pd.DataFrame(data)
        if len(data) > 0:
            new_index = ['year', 'dayofyear', 'hour', 'minute', 'second', 'spatial']
            data = data.set_index(new_index)
            data = data.pivot(columns='channel')
        return data

    def open_snapshot(self, year, dayofyear, hour, minute, spatial=None):
        hour_files = self.local_files(year, dayofyear, hour)

        minutes = hour_files.index.get_level_values('minute')
        if spatial is None:
            spatial = hour_files.index.get_level_values('spatial')[0]

        idx = np.argmin(np.abs(minutes-minute))

        minute_sel = minutes[idx]

        snapshot_files = hour_files.loc[year, dayofyear, hour, minute_sel]['file']        
        rads = []
        regrid = utils.regrid_1km
        for c in self.channels:
            band_files = snapshot_files[c]
            rad_c = open_band_radiance(band_files)
            rad_c = rad_c.expand_dims(dict(band=[c]))
            rad_c_regrid = regrid(rad_c, c)
            rads.append(rad_c_regrid)
            rads[-1] = rads[-1].assign_coords(x=rads[0].x.values,
                                          y=rads[0].y.values)

        rad = xr.concat(rads, 'band').data
        rad = rad.swapaxes(0, 1) # put time in front
        return rad

