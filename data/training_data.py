import os, sys
import itertools
import numpy as np
import dask

import utils
from goesr import GOESL1b, L1bBand, GroupBandTemporal

import torch
from torch.utils.data import Dataset, DataLoader

from joblib import delayed, Parallel

def iterate_examples(files):
    '''
    Iterate through consecutive examples
    '''
    if files.shape[0] == 0:
        return

    for i, row in files.iterrows():
        snapshot = L1bBand(row['file'].values[0])
        snapshot.open_dataset()
        yield row.index, snapshot

def training_samples(bands, year=None, dayofyear=None, hour=None,
                     data_directory='GOES16/',
                     product='ABI-L1b-RadC', spatial='RadC',
                     patch_size=136, sample_size=5,
                     zarr_directory=None):
    if (min(bands) < 6) and (hour < 12):
        return

    if os.path.exists(zarr_directory):
        #data = dask.from_zarr(zarr_directory)
        return

    geo = GOESL1b(data_directory=data_directory, channels=bands,
                  product=product)
    files = geo.local_files(year, dayofyear, hour=hour)
    print(data_directory, product, year, dayofyear, hour, files.shape)
    if len(files) == 0:
        return

    if spatial not in files.index.get_level_values('spatial'):
        return


    files = files.xs(spatial, level='spatial')
    iterator = iterate_examples(files)

    curr_samples = []
    all_samples = []
    for i, (idx, example) in enumerate(iterator):
        # filter out visible nighttime examples
        if (example.band) < 6 and (example.hour < 12):
            curr_samples = []
            continue

        if len(curr_samples) == sample_size:
            # save sample and continue
            group = GroupBandTemporal(curr_samples)
            sample_data = group.get_radiance_patches(patch_size)
            if sample_data is not None:
                all_samples.append(sample_data)
            curr_samples = []
        else:
            curr_samples.append(example)

    if len(all_samples) == 0:
        return

    data = dask.array.concatenate(all_samples, 0)
    # remove examples with nan and inf values
    finite = dask.array.isfinite(data).any(axis=(1,2,3))
    data = data[finite]
    data.compute_chunk_sizes()
    data = data.rechunk((1, -1, -1, -1))
    if zarr_directory is not None:
        print("Saving to directory: {}".format(zarr_directory))
        try:
            data.to_zarr(zarr_directory, overwrite=True)
        except ZeroDivisionError:
            print(zarr_directory)
        return
    return data

def training_dataset(zarr_directory, band, years, days, hours,
                     data_directory='GOES16/',
                     product='ABI-L1b-RadC', spatial=None, n_times=20,
                     cpu_count=16, patch_size=136):
    if spatial is not None:
        pass
    elif product == 'ABI-L1b-RadC':
        spatial = ['RadC']
    elif product == 'ABI-L1b-RadF':
        spatial = ['RadF']
    elif product == 'ABI-L1b-RadM':
        spatial = ['RadM1', 'RadM2']

    inter = [(y, d, h, s) for y in years for d in days for h in hours for s in spatial]
    jobs = []
    for y, d, h, s in inter:
        sample_directory = os.path.join(zarr_directory, '%04i_%03i_%02i_%s' % (y, d, h, s))
        if os.path.exists(sample_directory):
            continue
        #jobs.append(delayed(training_samples)([band], year=y, dayofyear=d, hour=h,
        #                                      product=product, spatial=s,
        #                                      sample_size=n_times, zarr_directory=sample_directory,
        #                                      patch_size=patch_size))
        training_samples([band], year=y, dayofyear=d, hour=h,
                          product=product, spatial=s,
                          data_directory=data_directory,
                          sample_size=n_times, zarr_directory=sample_directory,
                          patch_size=patch_size)
    #Parallel(n_jobs=cpu_count)(jobs)

def main():
    #bands = range(1,17)
    bands = [1,]
    product = 'ABI-L1b-RadM'
    zarr_path = 'training-data/Interp-{}-15min-264x264'.format(product)
    data_directory = '/nex/datapool/geonex/public/GOES16/NOAA-L1B/'
    year = 2017
    days = np.arange(1, 365, 5)
    hours = np.arange(0, 24, 1)
    for b in bands:
        output_path = os.path.join(zarr_path, 'Channel-%02i' % b)
        print(data_directory, output_path)
        training_dataset(output_path, b, [year], days, hours,
                         product=product, n_times=15, cpu_count=8,
                         data_directory=data_directory,
                         patch_size=264)

if __name__ == '__main__':
    main()
