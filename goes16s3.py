import os, sys
import datetime
import io
import time
import shutil

import boto
import xarray as xr
import numpy as np
import pandas as pd
import scipy.misc

import torch
from torch.utils.data import Dataset, DataLoader

import utils

# temporary directory is used to save file from s3 and read via xarray
TEMPORARY_DIR = './tmp'

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
        if not os.path.exists(TEMPORARY_DIR):
            os.makedirs(TEMPORARY_DIR)
        k = boto.s3.key.Key(self.goes_bucket)
        k.key = keyname
        tmpf = os.path.join(TEMPORARY_DIR, os.path.basename(keyname))
        content = k.get_contents_to_filename(tmpf)
        ds = self._open_file(tmpf, normalize=normalize)
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
                print("Year: %4i, Day: %i, Hour: %i" % (year, day, hname))
                hourly_das = []
                grouped_minutes = hgroup.groupby(by=['minute'])
                for mname, mgroup in grouped_minutes: # 60 minutes
                    mds = []
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
                    hourly_das.append(mds)
                    if len(hourly_das) > 20:
                        break

                hourly_das = xr.concat(hourly_das, dim='t')
                shutil.rmtree(TEMPORARY_DIR)
                yield hourly_das

def read_from_s3():
    goes = NOAAGOESS3(channels=range(1,17))
    pairs = goes.year_day_pairs()

def write_example_blocks_to_s3(year, day, bucket_name='nex-goes-slowmo'):
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)

    counter = 0
    goes = NOAAGOESS3(channels=range(1,7))


    for data in goes.read_day(year, day):
        blocked_data = utils.blocks(data)
        print(len(blocked_data), blocked_data[0].shape)

        if not os.path.exists(TEMPORARY_DIR):
            os.makedirs(TEMPORARY_DIR)

        # save blocks such that 15 minutes (16 timestamps) + 4 for randomness
        # overlap by 5 minutes

        n = 20
        for blocks in blocked_data:
            idxs = np.arange(0, blocks.shape[0], 16)
            for i in idxs:
                if i + n > blocks.shape[0]:
                    i = blocks.shape[0] - n

                b = blocks[i:i+n]
                fname = "%04i_%03i_%07i.npy" % (year, day, counter)
                save_file = os.path.join(TEMPORARY_DIR, fname)
                np.save(save_file, b)
                k= boto.s3.key.Key(bucket)
                k.key = 'train/%s' % fname
                k.set_contents_from_filename(save_file)
                os.remove(save_file)
                counter += 1
                print(fname, k, b.shape)

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
    write_example_blocks_to_s3(2018, 2)

