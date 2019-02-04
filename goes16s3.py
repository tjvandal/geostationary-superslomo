import os, sys
import datetime
import io
import time
import shutil

import boto
import boto3
import botocore
import xarray as xr
import numpy as np
import pandas as pd
import scipy.misc
import psutil

import torch
from torch.utils.data import Dataset, DataLoader

import utils

# temporary directory is used to save file from s3 and read via xarray
TEMPORARY_DIR = '/raid/tj/GOES/ABI-L1b-RadM/'

class NOAAGOESS3(object):
    '<Key: noaa-goes16,ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc>'
    def __init__(self, product='ABI-L1b-RadM', channels=range(1,17)):
        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.channels = channels
        self.conn = boto.connect_s3(host='s3.amazonaws.com')
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
        try:
            ds = xr.open_dataset(f)
        except IOError:
            os.remove(f)
            return None

        if normalize:
            mn = ds['min_radiance_value_of_valid_pixels'].values
            mx = ds['max_radiance_value_of_valid_pixels'].values
            ds['Rad'] = (ds['Rad'] - mn) / (mx - mn)
        return ds

    def download_from_s3(self, keyname, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        k = boto.s3.key.Key(self.goes_bucket)
        k.key = keyname
        data_file = os.path.join(directory, os.path.basename(keyname))
        if os.path.exists(data_file):
            pass
        elif k.exists():
            k.get_contents_to_filename(data_file)
        else:
            data_file = None
        return data_file

    def read_nc_from_s3(self, keyname, normalize=True):
        data_file = self.download_from_s3(keyname, TEMPORARY_DIR)
        if data_file is not None:
            ds = self._open_file(data_file, normalize=normalize)
            if ds is None:
                self.read_nc_from_s3(keyname, normalize=normalize)
        else:
            ds = None
            data_file = None
        return ds, data_file

    def read_day(self, year, day, hours=range(12,25),
                 store_files=True):
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
        if len(keys_df) == 0: return

        grouped_spatial = keys_df.groupby(by=['spatial'])
        daily_das = []
        for sname, sgrouped in grouped_spatial: #max of 2 groups
            grouped_hourly = sgrouped.groupby(by=["hour"])
            for hname, hgroup in grouped_hourly: #limited to 24
                print("Day", day, "Hour", hname)
                hourly_das = []
                hourly_files = []
                grouped_minutes = hgroup.groupby(by=['minute'])

                for mname, mgroup in grouped_minutes: # 60 minutes
                    mds = []
                    for i in mgroup.index:
                        ds, f = self.read_nc_from_s3(mgroup.loc[i].keyname)
                        hourly_files.append(f)
                        if f is None:
                            break

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
                        try:
                            mds[-1]['x'].values = mds[0]['x'].values
                            mds[-1]['y'].values = mds[0]['y'].values
                        except ValueError:
                            print('spatial', sname, 'hname', hname, 'mname', mname,
                                  'mds[-1]', mds[-1], 'mds[0]', mds[0])
                            raise
                    if f is None: # missing a minute in this hour, just skip it for now
                        print("f is none")
                        break

                    mds = xr.concat(mds, dim='band')
                    mds['t'] = ds['t']
                    hourly_das.append(mds)
                    # in this case, a snapshot is being captured every 30 seconds, ignore the hour
                    if np.unique(mds.band.values).size < mds.band.shape[0]:
                        print("This hour has more than 1 snapshot per minute, skip it.")
                        hourly_das = []
                        break

                if len(hourly_das) == 0: continue

                x0 = hourly_das[0].x.values
                xn = hourly_das[-1].x.values
                y0 = hourly_das[0].y.values
                yn = hourly_das[-1].y.values

                if not np.all(x0 == xn): print("x Indicies do not match"); continue
                if not np.all(y0 == yn): print("y Indicies do not match"); continue

                hourly_das = xr.concat(hourly_das, dim='t')
                yield hourly_das

                if not store_files:
                    [os.remove(f) for f in hourly_files]

class GOESDataset(Dataset):
    def __init__(self, example_directory='/raid/tj/GOES/SloMo-5min/',
                 n_upsample=5, n_overlap=3):
        self.example_directory = example_directory
        if not os.path.exists(self.example_directory):
            os.makedirs(self.example_directory)
        self._example_files()
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap

    def _example_files(self):
        self.example_files = [os.path.join(self.example_directory, f) for f in
                              os.listdir(self.example_directory) if 'npy' == f[-3:]]
        self.example_files = self.example_files[:1000]
        self.N_files = len(self.example_files)

    def _check_directory(self, year, day):
        yeardayfiles = [f for f in self.example_files if '%4i_%03i' % (year, day) in f]
        if len(list(yeardayfiles)) > 0:
            return True
        return False

    def write_example_blocks(self, year, day, channels=range(1,17), force=False):
        counter = 0
        goes = NOAAGOESS3(channels=channels)
        if (self._check_directory(year, day)) and (not force):  return

        for data in goes.read_day(year, day):
            blocked_data = utils.blocks(data, width=360)

            # save blocks such that 15 minutes (16 timestamps) + 4 for randomness n=20
            # overlap by 5 minutes

            n = self.n_upsample + self.n_overlap + 1
            for blocks in blocked_data:
                idxs = np.arange(0, blocks.shape[0], self.n_upsample)
                for i in idxs:
                    if i + n > blocks.shape[0]:
                        i = blocks.shape[0] - n

                    b = blocks[i:i+n]
                    if np.all(np.isfinite(b)):
                        fname = "%04i_%03i_%07i.npy" % (year, day, counter)
                        save_file = os.path.join(self.example_directory, fname)
                        print("saved file: %s" % save_file)
                        np.save(save_file, b)
                        #self.bucket.upload_file(save_file, '%s/%s' % (self.s3_base_path, fname))
                        #os.remove(save_file)
                        counter += 1
                    else:
                        print("Is not finite")
        self._example_files()

    def transform(self, block):
        n_select = self.n_upsample + 1
        # randomly shift temporally
        i = np.random.choice(range(0,self.n_overlap))
        block = block[i:n_select+i]

        # randomly shift vertically 
        i = np.random.choice(range(0,8))
        block = block[:,:,i:i+352]

        # randomly shift horizontally
        i = np.random.choice(range(0,8))
        block = block[:,:,:,i:i+352]

        # randomly flip up-down
        #if np.random.uniform() > 0.5:
            #block = block[:,::-1]
            #block = np.flip(block, axis=1).copy()

        # randomly flip right-left 
        #if np.random.uniform() > 0.5:
        #    block = block[:,:,::-1]
        #    block = np.flip(block, axis=2).copy()

        return block

    def __len__(self):
        return self.N_files

    def __getitem__(self, idx):
        f = self.example_files[idx]
        #obj = self.s3_keys[idx]
        #key = obj.key
        #if key is None:
        #    return self.__getitem__((idx + 1) % self.N_keys)

        #s3 = boto3.resource('s3')
        #obj = s3.Object(self.bucket_name, key)

        #try:
        #bio = io.BytesIO(obj.get()['Body'].read())
        #except botocore.Exceptions.ClientError as ex:
        #    if obj['Error']['Code'] == 'NoSuchKey':
        #        return self.__getitem__(idx + 1 % self.N_keys)
        #    else:
        #        raise ex
        try:
            block = np.load(f)
        except Exception as err:
            raise TypeError
        block = self.transform(block)

        I0 = torch.from_numpy(block[0])
        I1 = torch.from_numpy(block[-1])
        IT = torch.from_numpy(block[1:-1])

        return I0, I1, IT

# DO NOT USE
# this class needs to be updated and dependent on GOESDataset
class GOESDatasetS3(Dataset):
    def __init__(self, s3_bucket_name="nex-goes-slowmo",
                 s3_base_path="train/", buffer_size=60,
                 n_upsample=5, n_overlap=3):
        self.bucket_name = s3_bucket_name
        self.resource = boto3.resource('s3')
        self.bucket = self.resource.Bucket(s3_bucket_name)
        self.s3_base_path = s3_base_path
        self.buffer_size = buffer_size
        self.s3_keys = list(self.bucket.objects.filter(Prefix=self.s3_base_path))
        #self.s3_keys = self.s3_keys[:100]
        #print(self.s3_keys)
        print("Number of training sample", len(self.s3_keys))
        self.N_keys = len(self.s3_keys)
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap

    def in_s3(self, year, day):
        yeardaykeys = self.bucket.objects.filter(Prefix='%s/%4i_%03i' % (self.s3_base_path, year, day)) 
        if len(list(yeardaykeys)) > 0:
            return True
        return False

    def write_example_blocks_to_s3(self, year, day, channels=range(1,17)):
        counter = 0
        goes = NOAAGOESS3(channels=channels)
        if self.in_s3(year, day):  return

        for data in goes.read_day(year, day):
            print(data.shape)
            blocked_data = utils.blocks(data, width=360)

            if not os.path.exists(TEMPORARY_DIR):
                os.makedirs(TEMPORARY_DIR)

            # save blocks such that 15 minutes (16 timestamps) + 4 for randomness n=20
            # overlap by 5 minutes

            n = self.n_upsample + self.n_overlap + 1
            for blocks in blocked_data:
                idxs = np.arange(0, blocks.shape[0], self.n_upsample)
                for i in idxs:
                    if i + n > blocks.shape[0]:
                        i = blocks.shape[0] - n

                    b = blocks[i:i+n]
                    if np.all(np.isfinite(b)):
                        fname = "%04i_%03i_%07i.npy" % (year, day, counter)
                        save_file = os.path.join(TEMPORARY_DIR, fname)
                        print("saved file: %s" % save_file)
                        np.save(save_file, b)
                        self.bucket.upload_file(save_file, '%s/%s' % (self.s3_base_path, fname))
                        os.remove(save_file)
                        counter += 1

    def __len__(self):
        return self.N_keys

    def __getitem__(self, idx):
        obj = self.s3_keys[idx]
        key = obj.key
        #if key is None:
        #    return self.__getitem__((idx + 1) % self.N_keys)

        #s3 = boto3.resource('s3')
        #obj = s3.Object(self.bucket_name, key)

        #try:
        bio = io.BytesIO(obj.get()['Body'].read())
        #except botocore.Exceptions.ClientError as ex:
        #    if obj['Error']['Code'] == 'NoSuchKey':
        #        return self.__getitem__(idx + 1 % self.N_keys)
        #    else:
        #        raise ex
        try:
            block = np.load(bio)
        except Exception as err:
            raise TypeError
        block = self.transform(block)

        I0 = torch.from_numpy(block[0])
        I1 = torch.from_numpy(block[-1])
        IT = torch.from_numpy(block[1:-1])

        return I0, I1, IT

def training_set():
    for n_channels in [16,]:
        goespytorch = GOESDataset(example_directory='/raid/tj/GOES/5Min-%iChannels-Train/' % n_channels)
        year = 2017
        for day in range(75,365):
        #[75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]:
        #for day in [75, 100,]:
            goespytorch.write_example_blocks(year, day, channels=range(1,n_channels+1), force=False)

def test_set():
    for n_channels in [8,]:
        data = NOAAGOESS3(channels=range(1,n_channels+1))
        year = 2018
        # March 3 Noreaster
        noreaster_day = datetime.datetime(year, 3, 3).timetuple().tm_yday
        # July 18 Montana wild fire
        wildfire_day = datetime.datetime(year, 7, 18).timetuple().tm_yday
        # October 10 Hurricane Michael 
        hurricane_day = datetime.datetime(year, 10, 10).timetuple().tm_yday
        for day in [noreaster_day, wildfire_day, hurricane_day]:
            print("day", day)
            for dataarray in data.read_day(year, day):
                pass

if __name__ == "__main__":
    #noaagoes = NOAAGOESS3(channels=range(1,4))
    #noaagoes.read_day(2017, 74).next()
    training_set()
    #test_set()
