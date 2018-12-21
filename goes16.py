import os, sys
import datetime

import boto
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.misc
#import tensorflow as tf

import torch
from torch.utils.data import Dataset, DataLoader


class Snapshot(object):
    def __init__(self, dir):
        self.dir = dir
        self.files = [os.path.join(dir, f) for f in os.listdir(dir) if f[-3:] == '.nc']
        self.channels = [os.path.basename(f).split('_')[1].split('-')[-1][3:]
                         for f in self.files]

    def read_snapshot(self):
        self.das = dict()
        for c, f in zip(self.channels, self.files):
            self.das[int(c)] = self.read_band(f)

    def rgb(self):
        if not hasattr(self, 'das'):
            self.read_snapshot()

        red = self.das[3].values
        green = cv2.resize(self.das[2].values, (0,0), fx=0.5, fy=0.5)
        blue = self.das[1].values
        rgb = np.concatenate([np.expand_dims(x, 2) for x in [red, green, blue]], axis=2)
        rgb = xr.DataArray(rgb, coords=[self.das[1].y, self.das[1].x, [3, 1, 2]],
                         dims=['x', 'y', 'channel'])
        return rgb

    def read_band(self, f):
        channel = os.path.basename(f).split('_')[1].split('-')[-1][3:]
        ds = xr.open_dataset(f)
        percent_valid = ds['valid_pixel_count'] / (ds.x.shape[0] * ds.y.shape[0])
        mn = ds['min_radiance_value_of_valid_pixels'].values
        mx = ds['max_radiance_value_of_valid_pixels'].values
        normalized = (ds['Rad'] - mn) / (mx - mn)
        normalized.name = channel
        return normalized

class SnapshotMinute(Snapshot):
    def __init__(self, dir):
        super(SnapshotMinute, self).__init__(dir)

    def read_snapshot(self):
        self.das = dict()
        for c, f in zip(self.channels, self.files):
            c = int(c)
            band = self.read_band(f)
            t = datetime.datetime.utcfromtimestamp(band.t.values.tolist()/1e9)
            t = t.replace(microsecond=0)
            if t not in self.das.keys():
                self.das[t] = dict()

            self.das[t][c] = self.read_band(f)

    def rgb(self):
        if not hasattr(self, 'das'):
            self.read_snapshot()
        times = self.das.keys()
        rgb = dict()
        for t in times:
            red = self.das[t][3].values
            green = cv2.resize(self.das[t][2].values, (0,0), fx=0.5, fy=0.5)
            blue = self.das[t][1].values
            rgb_vals = np.concatenate([np.expand_dims(x, 2) for x in [red, green, blue]], axis=2)
            rgb[t] = xr.DataArray(rgb_vals,
                            coords=[self.das[t][1].y, self.das[t][1].x, [3, 1, 2]],
                            dims=['x', 'y', 'channel'])
        return rgb

class NOAAGOES(object):
    '<Key: noaa-goes16,ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc>'
    def __init__(self, data_dir='/raid/tj/GOES16/', product='ABI-L1b-RadC', channels=[3,1,2]):
        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.data_dir = data_dir
        self.product_dir = os.path.join(data_dir, self.product)
        self.channels = [3, 1, 2]

    def download_day_hour(self, year, day, hour):
        conn = boto.connect_s3()
        bucket = conn.get_bucket(self.bucket_name)

        keybase = '%(product)s/%(year)04i/%(day)03i/%(hour)02i/' % dict(product=self.product,
                                                                       year=year, day=day,
                                                                       hour=hour)
        for key in bucket.list(keybase):
            filename = os.path.basename(key.key)
            channel = filename.split('_')[1].split('-')[-1][3:]
            channel = int(channel)
            timestamp = filename.split('_')[3]
            minute = timestamp[10:12]
            if channel in self.channels:
                print('Downloading', key)
                save_dir = os.path.join(self.product_dir, str(year), str(day), str(hour), minute)
                write_file = os.path.join(save_dir, filename)
                if os.path.exists(write_file):
                    continue
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                key.get_contents_to_filename(os.path.join(save_dir, filename))

    def years(self):
        return [int(d) for d in os.listdir(self.product_dir)]

    def get_year_directory(self, year):
        return os.path.join(self.product_dir, str(year))

    def get_day_directory(self, year, day):
        return os.path.join(self.product_dir, str(year), str(day))

    def get_hour_directory(self, year, day, hour):
        return os.path.join(self.product_dir, str(year), str(day), str(hour))

    def get_minute_directory(self, year, day, hour, minute):
        return os.path.join(self.get_hour_directory(year, day, hour), str(minute))

    def iterate_hours(self):
        for y in self.years():
            ypath = self.get_year_directory(y)
            days = sorted([int(d) for d in os.listdir(ypath)])
            for d in days:
                dpath = self.get_day_directory(y, d)
                hours = sorted([int(h) for h in os.listdir(dpath)])
                for h in hours:
                    yield self.get_hour_directory(y, d, h)

    def blocks(self, data, width=512):
        n = data.time.shape[0]
        w = data.x.shape[0]
        h = data.y.shape[0]
        d = data.channel.shape[0]

        hs = np.arange(0, h, width)
        ws = np.arange(0, w, width)
        blocks = []
        for hindex in hs:
            if hindex+width > h:
                hindex = h - width

            for windex in ws:
                if windex+width > w:
                    windex = w - width

                blocks.append(data.isel(y=range(hindex, hindex+width),
                                        x=range(windex, windex+width)))

        return blocks

    def slowmo_examples(self):
        # Returns:
        #    I0, I1, IT
        block_size = 512
        for BS in self.iterate(shuffle=True, block_size=block_size+8):
            for B in BS:
                yield np.concatenate([B[:9], B[3:]], axis=0)

    def iterate(self, shuffle=False, block_size=352):
        hour_dirs = sorted([f for f in self.iterate_hours()])
        hour_data = None
        for hdir  in hour_dirs:
            print("Hour directory: %s" % hdir)
            hour_next = self.read_hour(hdir) #(12,512,512,3)
            if hour_data is None:
                print("Hour data is None")
                hour_data = hour_next
                continue

            if hour_next is not None:
                hour_data_cat = np.concatenate([hour_data, hour_next[0,np.newaxis]], 0)

            blocks = self.blocks(hour_data_cat, width=block_size)
            if shuffle:
                idxs = range(0, blocks.shape[0])
                np.random.shuffle(idxs)
                blocks = blocks[idxs]

            print("Blocks Shape", blocks.shape)
            yield blocks
            hour_data = hour_next

class NOAAGOESC(NOAAGOES):
    def __init__(self, data_dir='/raid/tj/GOES16/'):
        super(NOAAGOESC, self).__init__(data_dir=data_dir, product='ABI-L1b-RadC')

    def read_hour(self, hour_dir):
        minutes = sorted([int(m) for m in os.listdir(hour_dir)])
        minutes = np.array(minutes)
        diff = minutes[1:] - minutes[:-1]

        assert len(minutes) == 12 # there should be 12 snapshots per hour

        data = []
        for m in minutes:
            minute_dir = os.path.join(hour_dir, '%02i' % m)
            snap = Snapshot(minute_dir)
            data.append(snap.rgb().expand_dims('time'))

        #data = np.concatenate(data, axis=0) #(12,5000,3000,3)
        data = xr.concat(data, 'time').assign_coords(time=minutes)
        return data

    def read_day(self, year, day):
        dir = self.get_day_directory(year, day)
        hours_avail = np.array(sorted([int(h) for h in os.listdir(dir)]))
        diff = hours_avail[1:] - hours_avail[:-1]
        assert np.all(diff == 1)

        times, daydata = [], []
        daytime = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
        for h in hours_avail:
            hdata = self.read_hour(os.path.join(dir, str(h)))
            times = [daytime + datetime.timedelta(hours=h, minutes=m) for m in hdata.time.values]
            hdata.time.values = times
            daydata.append(hdata)

        #hdata = np.concatenate(daydata, axis=0)
        hdata = xr.concat(daydata, 'time')
        return hdata

class NOAAGOESM(NOAAGOES):
    def __init__(self, data_dir='/raid/tj/GOES16/'):
        super(NOAAGOESM, self).__init__(data_dir=data_dir, product='ABI-L1b-RadM')

    def read_hour(self, hour_dir):
        minutes = sorted([int(m) for m in os.listdir(hour_dir)])
        minutes = np.array(minutes)
        diff = minutes[1:] - minutes[:-1]

        #assert len(minutes) == 12 # there should be 12 snapshots per hour
        assert np.unique(diff)[0] == 1

        data = []
        group_to_seconds = dict()
        for m in minutes:
            minute_dir = os.path.join(hour_dir, '%02i' % m)
            snap = SnapshotMinute(minute_dir)
            rgb_das = snap.rgb()
            for key, item in rgb_das.items():
                s = key.second
                if s not in group_to_seconds.keys():
                    group_to_seconds[s] = []
                group_to_seconds[s].append(item.expand_dims('time'))

        for s, group in group_to_seconds.items():
            does_x_match = np.all([(g.x == g[0].x).all().values for g in group])
            does_y_match = np.all([(g.y == g[0].y).all().values for g in group])
            assert does_x_match and does_y_match
            group_to_seconds[s] = xr.concat(group, 'time').assign_coords(time=minutes)

        return group_to_seconds

    def read_day(self, year, day):
        dir = self.get_day_directory(year, day)
        hours_avail = np.array(sorted([int(h) for h in os.listdir(dir)]))
        diff = hours_avail[1:] - hours_avail[:-1]
        assert np.all(diff == 1)

        times, daydata = dict(), dict()
        daytime = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
        for h in hours_avail:
            hdata = self.read_hour(os.path.join(dir, str(h)))
            for s, curr_data in hdata.items():
                if s not in daydata.keys():
                    daydata[s] = list()

                times = [daytime + datetime.timedelta(hours=h, minutes=m, seconds=s)
                         for m in curr_data.time.values]
                curr_data.time.values = times
                daydata[s].append(curr_data)

        daydata = {k: xr.concat(d, 'time') for k, d in daydata.items()}
        return daydata

class GOESDataset(Dataset):
    def __init__(self,  goes_data_dir='/raid/tj/GOES16/',
                 example_dir='/raid/tj/GOES16/pytorch-training/',
                 buffer_size=60):
        self.buffer_size = buffer_size
        self.noaagoes = NOAAGOES(data_dir=goes_data_dir)
        self.example_dir = example_dir
        if not os.path.exists(self.example_dir):
            os.makedirs(self.example_dir)

        self.example_files = [os.path.join(self.example_dir, f) for f in
                              os.listdir(self.example_dir) if f[-3:] == 'npy']

        if len(self.example_files) == 0:
            self.write_example_blocks()
            self.example_files = [os.path.join(self.example_dir, f) for f in
                              os.listdir(self.example_dir) if f[-3:] == 'npy']

    def write_example_blocks(self):
        counter = 0
        for blocks in self.noaagoes.iterate(block_size=360):
            split_blocks = np.concatenate([blocks[:,:6], blocks[:,4:10], blocks[:,7:]], axis=0)
            for i, b in enumerate(split_blocks):
                save_file = os.path.join(self.example_dir, "%07i.npy" % counter)
                np.save(save_file, b)
                counter += 1
            print("count=%i" % counter)

    def transform(self, block):
        # randomly shift temporally
        i = np.random.choice(range(0,3))
        block = block[i:4+i]

        # randomly shift vertically 
        i = np.random.choice(range(0,8))
        block = block[:,i:i+352]

        # randomly shift horizontally
        i = np.random.choice(range(0,8))
        block = block[:,:,i:i+352]

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
        return len(self.example_files)

    def __getitem__(self, idx):
        block = np.load(self.example_files[idx])
        block = self.transform(block)

        I0 = torch.from_numpy(block[0])
        I1 = torch.from_numpy(block[-1])
        IT = torch.from_numpy(block[1:-1])

        I0 = I0.permute(2,0,1)
        I1 = I1.permute(2,0,1)
        IT = IT.permute(0,3,1,2)
        return I0, I1, IT

def download_day(day=251, year=2017, channels=[3,2,1], product='ABI-L1b-RadC'):
    obj = NOAAGOES(product=product, channels=channels)
    hours = range(12,24)
    for h in hours:
        obj.download_day_hour(year, day, h)

def make_video_images():
    obj = GOESDataset()
    hours = range(12,24)
    for h in hours:
        obj.download_day_hour(2017, 251, h, channels=[3,2,1])

    j = 0
    for h in hours:
        dir = 'data/ABI-L1b-RadC/2017/251/%i' % h
        for i, minute in enumerate(sorted(os.listdir(dir))):
            snap = Snapshot('%s/%s' % (dir, minute))
            rgb = (255*snap.rgb()).astype(int)
            rgb[rgb < 0] = 0
            rgb[rgb > 255] = 255
            scipy.misc.imsave('images/%03i.jpg' % j, rgb)
            j += 1


if __name__ == "__main__":
    # make_video_images()
    #for day in [0, 30, 60, 90, 120, 150, 180,
    #            210, 240, 270, 300, 330, 360]:
    #    download_day(day=day)
    download_day(120, product='ABI-L1b-RadM')
    #obj = GOESDataset()
    #obj.write_example_blocks()
    #obj.__getitem__(0)
    #goesc = NOAAGOESM()
    #daydata = goesc.read_day(2017, 120)
    #print(daydata.keys())
    #goesc.read_day_hour_minute(2017, 120, 12, 2)
    #daytimes, daydata = goesc.read_day(2017, 120)
    #dayblocks = goesc.blocks(daydata, width=352)

