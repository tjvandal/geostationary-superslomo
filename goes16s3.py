import os, sys
import datetime
import io
import time

import boto
import xarray as xr
import numpy as np
import pandas as pd
import scipy.misc

import torch
from torch.utils.data import Dataset, DataLoader

import tempfile

import utils


class NOAAGOESS3(object):
    '<Key: noaa-goes16,ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc>'
    def __init__(self, product='ABI-L1b-RadM', channels=range(1,17)):
        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.channels = channels
        self.conn = boto.connect_s3()
        self.goes_bucket = self.conn.get_bucket(self.bucket_name)

    def year_day_pairs(self):
        days = []
        for key_year in self.goes_bucket.list(self.product+"/", "/"):
            y = int(key_year.name.split('/')[1])
            if y == 2000:
                continue
            for key_day in self.goes_bucket.list(key_year.name, "/"):
                d = int(key_day.name.split('/')[2])
                days += [(y, d)]
        return days

    def day_keys(self, year, day, hours=range(12,24)):
        keybase = '%(product)s/%(year)04i/%(day)03i/' % dict(product=self.product,
                                                           year=year, day=day)
        data = []
        for key_hour in self.goes_bucket.list(keybase, "/"):
            hour = int(key_hour.name.split('/')[3])
            if hour not in hours:
                continue

            for key_nc in self.goes_bucket.list(key_hour.name, '/'):
                fname = key_nc.name.split('/')[4]
                info = fname.split('-')[3]
                c, g, t, _, _ = info.split("_")
                spatial = fname.split('-')[2]
                c = int(c[3:])
                if c not in self.channels:
                    continue
                minute = int(t[10:12])
                second = int(t[12:15])
                data.append(dict(channel=c, year=year, day=day, hour=hour,
                                 minute=minute, second=second, spatial=spatial, 
                                 keyname=key_nc.name))
                #if len(data) > 100:
                #    break

        return pd.DataFrame(data)

    def _open_file(self, f, normalize=True):
        ds = xr.open_dataset(f, backend_kwargs={'diskless': True})
        if normalize:
            mn = ds['min_radiance_value_of_valid_pixels'].values
            mx = ds['max_radiance_value_of_valid_pixels'].values
            ds['Rad'] = (ds['Rad'] - mn) / (mx - mn)
        return ds

    def read_nc_from_s3(self, keyname, normalize=True):
        k = boto.s3.key.Key(self.goes_bucket)
        k.key = keyname
        tmpf = tempfile.NamedTemporaryFile()
        content = k.get_contents_to_filename(tmpf.name)
        ds = self._open_file(tmpf.name, normalize=normalize)
        #tmpf.close()
        return ds

    def read_day(self, year, day, hours=range(12,24)):
        '''
        Reads and joins product data for entire day in temporal order.
        args:
            year: int
            day: int
        returns:
            list(xarray.DataArray)
        '''
        t0 = time.time()
        keys_df = self.day_keys(year, day, hours=hours)
        grouped_spatial = keys_df.groupby(by=['spatial'])
        daily_das = []
        for sname, sgrouped in grouped_spatial: #max of 2 groups
            grouped_hourly = sgrouped.groupby(by=["hour"])
            for hname, hgroup in grouped_hourly: #limited to 24
                grouped_minutes = hgroup.groupby(by=['minute'])
                for mname, mgroup in grouped_minutes: # 60 minutes
                    mds = []
                    print(hname, mname)
                    for i in mgroup.index:
                        ds = self.read_nc_from_s3(mgroup.loc[i].keyname)
                        if mgroup.loc[i].channel in [1,3,5]: #(1 km)
                            newds = utils.interp_da2d(ds.Rad, 1./2, fillna=False)
                        elif mgroup.loc[i].channel == 2: #(0.5 km)
                            newds = utils.interp_da2d(ds.Rad, 1./4, fillna=False)
                        else:
                            newds = ds.Rad

                            del newds['x_image']
                            del newds['y_image']
                            del newds['t']

                        newds['band'] = mgroup.loc[i].channel
                        mds.append(newds)
                        mds[-1]['x'].values = mds[0]['x'].values
                        mds[-1]['y'].values = mds[0]['y'].values

                    mds = xr.concat(mds, dim='band')
                    mds['t'] = ds['t']
                    daily_das.append(mds)
                    if len(daily_das) > 20:
                        break

                daily_das = xr.concat(daily_das, dim='t')
                return daily_das

def read_from_s3():
    goes = NOAAGOESS3(channels=range(1,17))
    pairs = goes.year_day_pairs()
    data = goes.read_day(pairs[-1][0], pairs[-1][1])
    blocked_data = utils.blocks(data)
    print(len(blocked_data), blocked_data[0].shape)

def write_example_blocks(blocks):
    counter = 0
    for blocks in self.noaagoes.iterate(block_size=360):
        split_blocks = np.concatenate([blocks[:,:6], blocks[:,4:10], blocks[:,7:]], axis=0)
        for i, b in enumerate(split_blocks):
            save_file = os.path.join(self.example_dir, "%07i.npy" % counter)
            np.save(save_file, b)
            counter += 1
        print("count=%i" % counter)


if __name__ == "__main__":
    # make_video_images()
    #for day in [0, 30, 60, 90, 120, 150, 180,
    #            210, 240, 270, 300, 330, 360]:
    #    download_day(day=day)
    #download_day(120, product='ABI-L1b-RadM')
    #obj = GOESDataset()
    #obj.write_example_blocks()
    #obj.__getitem__(0)
    #goesc = NOAAGOESM()
    #daydata = goesc.read_day(2017, 120)
    #print(daydata.keys())
    #goesc.read_day_hour_minute(2017, 120, 12, 2)

    #daytimes, daydata = goesc.read_day(2017, 120)
    #dayblocks = goesc.blocks(daydata, width=352)
    read_from_s3()
