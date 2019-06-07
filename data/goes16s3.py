import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from tools import utils

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
from joblib import delayed, Parallel

import torch
from torch.utils.data import Dataset, DataLoader

import tools


# temporary directory is used to save file from s3 and read via xarray
#TEMPORARY_DIR = '/raid/tj/GOES/S3'
TEMPORARY_DIR = '/mnt/nexai-goes/GOES/S3'
EXAMPLE_DIR = '/raid/tj/GOES/SloMo'
TEST_DIR = '/raid/tj/GOES/SloMo-test'

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

def regrid_2km(da, band):
    if band in [1,3,5]: #(1 km)
        da = utils.interp_da2d(da, 1./2, fillna=False)
    elif band == 2: #(0.5 km)
        da = utils.interp_da2d(da, 1./4, fillna=False)
    return da

def _open_and_merge_2km(files, normalize=True):
    '''
    This method opens a list of S3 NOAA ABI files,
        normalizes by max and min radiances,
        and interpolations to 2km.
    Return:
        xarray.DataArray (band: len(files), x, y)
        
    https://www.star.nesdis.noaa.gov/goesr/docs/ATBD/Imagery.pdf    
    '''
    norm_factors = {1: (-26, 805), 2: (-20, 628), 3: (-12, 373),
                    4: (-4, 140), 5: (-3, 94), 6: (-1, 30), 7: (0, 25),
                    8: (0, 28), 9: (0, 28), 10: (0, 44), 11: (0, 79),
                    12: (0, 134), 13: (0, 183), 14: (0, 198), 15: (0, 211),
                    16: (0, 168)}
    das = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            band_id = ds.band_id.values[0]
        except Exception:
            raise ValueError("Cannot read file: {}".format(f))

        # normalize radiance
        if normalize:
            mn = norm_factors[band_id][0]
            mx = norm_factors[band_id][1]
            #mn = ds['min_radiance_value_of_valid_pixels'].values
            #mx = ds['max_radiance_value_of_valid_pixels'].values
            ds['Rad'] = (ds['Rad'] - mn) / (mx - mn)
            
        #ds['Rad'] *= 1e-3
        # regrid to 2km to match all bands
        newrad = regrid_2km(ds['Rad'], band_id)
        newrad = newrad.expand_dims(dim="band")
        newrad = newrad.assign_coords(band=ds.band_id.values)

        # for some reason, these coordinates are included in band 4
        if 't' in newrad.coords:
            newrad = newrad.drop('t')
        if 'y_image' in newrad.coords:
            newrad = newrad.drop('y_image')
        if 'x_image' in newrad.coords:
            newrad = newrad.drop('x_image')

        das.append(newrad)

        # reindex so the concatenation works correctly
        #das[-1] = das[-1].reindex({'x': das[0].x.values, 
        #                           'y': das[0].y.values})
        das[-1] = das[-1].assign_coords(x=das[0].x.values,
                                        y=das[0].y.values)
    # concatenate each file by band after interpolating to the same grid
    das = xr.concat(das, 'band')
    return das

class NOAAGOESS3(object):
    '<Key: noaa-goes16,ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc>'
    def __init__(self, product='ABI-L1b-RadM', channels=range(1,17),
                       save_directory='/mnt/nexai-goes/GOES/S3'):
        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.channels = channels
        self.conn = boto.connect_s3(host='s3.amazonaws.com')
        self.goes_bucket = self.conn.get_bucket(self.bucket_name)
        self.save_directory = os.path.join(save_directory, product)    

    def year_day_pairs(self):
        '''
        Gets all year and day pairs in S3 for the given product
        
        Return:
            list of pairs
        '''
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
            print("writing file to {}".format(data_file))
            k.get_contents_to_filename(data_file)
        else:
            data_file = None
        return data_file

    def read_nc_from_s3(self, keyname, normalize=True):
        data_file = self.download_from_s3(keyname, self.save_directory)
        if data_file is not None:
            ds = self._open_file(data_file, normalize=normalize)
            if ds is None:
                self.read_nc_from_s3(keyname, normalize=normalize)
        else:
            ds = None
            data_file = None
        return ds, data_file

    def download_day(self, year, day, hours=range(12, 25)):
        '''
        Downloads all files for a given year and dayofyear for the defined channels
        '''
        keys_df = self.day_keys(year, day, hours=hours)
        for i, row in keys_df.iterrows():
            save_dir= os.path.join(self.save_directory,
                                   '%04i/%03i/%02i/%02i/%s/' % (year, day, row.hour, row.minute,
                                                                row.spatial))
            data_file = self.download_from_s3(row.keyname, save_dir)

    def local_files(self, year=None, dayofyear=None):
        data = []
        base_dir = self.save_directory
        if year is not None:
            base_dir = os.path.join(base_dir, '%04i' % year)
            if dayofyear is not None:
                base_dir = os.path.join(base_dir, '%03i' % dayofyear)            
        #for f in os.listdir(self.save_directory):
        for directory, folders, files in os.walk(base_dir):
            for f in files:
                if f[-3:] == '.nc':
                    meta = get_filename_metadata(f)
                    meta['file'] = os.path.join(directory, f)
                    data.append(meta)

        data = pd.DataFrame(data)
        if len(data) > 0:
            data = data.set_index(['year', 'dayofyear', 'hour', 'minute', 'second', 'spatial'])
            data = data.pivot(columns='channel')
        return data

    def iterate_day(self, year, day, hours=range(12,25), max_queue_size=5, min_queue_size=3,
                    normalize=True):
        # get locally stored files
        file_df = self.local_files(year, day)
        if len(file_df) == 0: return

        grouped_spatial = file_df.groupby(by=['spatial'])
        for sname, sgrouped in grouped_spatial:
            running_samples = []
            for idx, row in sgrouped.iterrows():
                # (2018, 1, 12, 0, 577, 'RadM2')
                year, day, hour, minute, second, _ = idx
                if hour not in hours:
                    running_samples = []
                    continue
                
                files = row['file'][self.channels]
                try:
                    da = _open_and_merge_2km(files.values, normalize=normalize)
                    running_samples.append(da)
                except ValueError:
                    running_samples = []
                    continue


                if len(running_samples) > 1:
                    running_samples[-1]['x'].values = running_samples[0]['x'].values
                    running_samples[-1]['y'].values = running_samples[0]['y'].values

                if len(running_samples) == max_queue_size:
                    try:
                        yield xr.concat(running_samples, dim='t')
                    except Exception as err:
                        print([type(el) for el in running_samples])
                        print(err)
                    while len(running_samples) > min_queue_size:
                        running_samples.pop(0)


## SloMo Training Dataset on NOAA GOES S3 data
class GOESDataset(Dataset):
    def __init__(self, example_directory='/raid/tj/GOES/SloMo-5min/',
                 n_upsample=15, n_overlap=5):
        self.example_directory = example_directory
        if not os.path.exists(self.example_directory):
            os.makedirs(self.example_directory)
        self._example_files()
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap

    def _example_files(self):
        self.example_files = [os.path.join(self.example_directory, f) for f in
                              os.listdir(self.example_directory) if 'npy' == f[-3:]]
        self.N_files = len(self.example_files)

    def _check_directory(self, year, day):
        yeardayfiles = [f for f in self.example_files if '%4i_%03i' % (year, day) in f]
        if len(list(yeardayfiles)) > 0:
            return True
        return False

    def write_example_blocks(self, year, day, channels=range(1,17), force=False,
                             patch_width=128+6):
        counter = 0
        if (self._check_directory(year, day)) and (not force):  return
        goes = NOAAGOESS3(channels=channels)
        # save blocks such that 15 minutes (16 timestamps) + 4 for randomness n=20
        #           overlap by 5 minutes
        data_iterator = goes.iterate_day(year, day, max_queue_size=12, min_queue_size=1)

        for data in data_iterator:
            blocked_data = utils.blocks(data, width=patch_width)
            for b in blocked_data:
                if np.all(np.isfinite(b)):
                    fname = "%04i_%03i_%07i.npy" % (year, day, counter)
                    save_file = os.path.join(self.example_directory, fname)
                    print("saved file: %s" % save_file)
                    np.save(save_file, b)
                    counter += 1
                else:
                    pass
        self._example_files()

    def transform(self, block):
        n_select = self.n_upsample + 1
        # randomly shift temporally
        i = np.random.choice(range(0,self.n_overlap))
        block = block[i:n_select+i]

        # randomly shift vertically 
        i = np.random.choice(range(0,6))
        block = block[:,:,i:i+128]

        # randomly shift horizontally
        i = np.random.choice(range(0,6))
        block = block[:,:,:,i:i+128]

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
        try:
            block = np.load(f)
            block = self.transform(block)
            I0 = torch.from_numpy(block[0])
            I1 = torch.from_numpy(block[-1])
            IT = torch.from_numpy(block[1:-1])
        except Exception as err:
            os.remove(f)
            del self.example_files[idx]
            self.N_files -= 1
            raise TypeError("Cannot load file: {}".formate(f))

        return I0, I1, IT

class GOESDatasetPT(Dataset):
    def __init__(self, example_directory='/raid/tj/GOES/SloMo-5min/',
                 n_upsample=15, n_overlap=5):
        self.example_directory = example_directory
        if not os.path.exists(self.example_directory):
            os.makedirs(self.example_directory)
        self._example_files()
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap

    def _example_files(self):
        self.example_files = [os.path.join(self.example_directory, f) for f in
                              os.listdir(self.example_directory) if '.pt' == f[-3:]]
        self.N_files = len(self.example_files)

    def _check_directory(self, year, day):
        yeardayfiles = [f for f in self.example_files if '%4i_%03i' % (year, day) in f]
        if len(list(yeardayfiles)) > 0:
            return True
        return False

    def write_example_blocks(self, year, day, channels=range(1,17), force=False,
                             patch_width=128+6):
        counter = 0
        if (self._check_directory(year, day)) and (not force):  return
        goes = NOAAGOESS3(channels=channels)
        # save blocks such that 15 minutes (16 timestamps) + 4 for randomness n=20
        #           overlap by 5 minutes
        data_iterator = goes.iterate_day(year, day, max_queue_size=12, min_queue_size=1)

        for data in data_iterator:
            blocked_data = utils.blocks(data, width=patch_width)
            for b in blocked_data:
                if np.all(np.isfinite(b)):
                    #fname = "%04i_%03i_%07i.npy" % (year, day, counter)
                    fname = "%04i_%03i_%07i.pt" % (year, day, counter)
                    save_file = os.path.join(self.example_directory, fname)
                    print("saved file: %s" % save_file)
                    #np.save(save_file, b)
                    btensor = torch.from_numpy(b.values)
                    torch.save(btensor, save_file)
                    counter += 1
                else:
                    pass
        self._example_files()

    def transform(self, block):
        n_select = self.n_upsample + 1
        # randomly shift temporally
        i = np.random.choice(range(0,self.n_overlap))
        block = block[i:n_select+i]

        # randomly shift vertically 
        i = np.random.choice(range(0,6))
        block = block[:,:,i:i+128]

        # randomly shift horizontally
        i = np.random.choice(range(0,6))
        block = block[:,:,:,i:i+128]

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
        try:
            block = torch.load(f)
            block = self.transform(block)
            I0 = block[0]
            I1 = block[-1]
            IT = block[1:-1]
        except Exception as err:
            os.remove(f)
            del self.example_files[idx]
            self.N_files -= 1
            raise TypeError("Cannot load file: {}".formate(f))

        return I0, I1, IT

### Work in progress
class Nowcast(Dataset):
    def __init__(self, example_directory='/raid/tj/GOES/SloMo-5min/',
                 n_overlap=3):
        self.example_directory = example_directory
        if not os.path.exists(self.example_directory):
            os.makedirs(self.example_directory)

        self._example_files()
        self.n_upsample = n_upsample
        self.n_overlap = n_overlap

    def _example_files(self):
        self.example_files = [os.path.join(self.example_directory, f) for f in
                              os.listdir(self.example_directory) if 'npy' == f[-3:]]
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

        for data in goes.iterator_day(year, day):
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


def download_data(test=False):
    if test:
        years = [2019]
        days = np.arange(1, 150, 10)
    else:
        years = [2017, 2018]
        days = np.arange(1, 365, 5)
    for n_channels in [3,8]:
        noaa = NOAAGOESS3(channels=range(1,n_channels+1))
        for year in years:
            for day in days:
                noaa.download_day(year, day)

def training_set():
    years = [2017, 2018]
    for n_channels in [3,8]:
        example_directory = os.path.join(EXAMPLE_DIR, '9Min-%iChannels-Train-pt' % n_channels)
        goespytorch = GOESDataset(example_directory=example_directory)
        jobs = []
        for year in years:
            for day in np.arange(1,365):
                print('Year: {}, Day: {}'.format(year, day))
                jobs.append(delayed(goespytorch.write_example_blocks)(year, day,
                                     channels=range(1,n_channels+1),
                                     force=False))
        cpu_count = 6
        Parallel(n_jobs=cpu_count)(jobs)

'''def test_set():
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
'''

def download_conus_data():
    for n_channels in [3,]:
        noaa = NOAAGOESS3(product='ABI-L1b-RadC', channels=range(1,n_channels+1))
        for year in [2018]:
            for day in [198,199,200]:
                noaa.download_day(year, day)

if __name__ == "__main__":
    #noaagoes = NOAAGOESS3(channels=range(1,4))
    #noaagoes.read_day(2017, 74).next()
    #download_data()
    #download_data(test=True)
    training_set()
    #test_set()
    #download_conus_data()
    # move files
    '''
    for d in os.listdir(TEMPORARY_DIR):
        dfull = os.path.join(TEMPORARY_DIR, d)
        for f in os.listdir(dfull):
            if f[-3:] != '.nc':
                continue
            meta = get_filename_metadata(f)
            save_dir= os.path.join(dfull, '%04i/%03i/%02i/%02i/%s/' % (meta['year'],
                                                                       meta['dayofyear'],
                                                                       meta['hour'], meta['minute'],
                                                                       meta['spatial']))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            new_filepath = os.path.join(save_dir, f)
            old_filepath = os.path.join(dfull, f)
            os.rename(old_filepath, new_filepath)
    '''