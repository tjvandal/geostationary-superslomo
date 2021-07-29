import torch
from torch.utils.data import Dataset, DataLoader

import json

import dask
import dask.array
import numpy as np

import os
import random

#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' #str(rank % N_GPUS)

def get_band_stats(band):
    filedir = os.path.dirname(os.path.abspath(__file__))
    stats = json.load(open(os.path.join(filedir, 'band_statistics.json'),'r'))
    mu = stats[str(band)]['mean']
    sd = stats[str(band)]['std']
    return mu, sd

class GOESLoaderZarr(Dataset):
    def __init__(self,
                 example_directory='./training-channel1-1hour/',
                 train=True,
                 normalize=True):
        self.example_directory = example_directory
        self.arr = dask.array.from_zarr(example_directory)
        self.N = self.arr.shape[0]
        #self.C = self.arr.shape[2]
        self.train = train
        print("self.arr", self.arr)

    def transform(self, sample):
        if np.random.uniform() > 0.5:
            sample = sample.flip(2)
        if np.random.uniform() > 0.5:
            sample = sample.flip(3)

        k = int((np.random.uniform()*4) % 4)
        if k > 0:
            sample = sample.rot90(k, (2,3))
        return sample

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = self.arr[idx].compute()
        sample = np.nan_to_num(sample)
        sample[~np.isfinite(sample)] = 0.0
        #sample[sample < 0] = 0.0
        sample = torch.from_numpy(sample).float()
        if self.train:
            sample = self.transform(sample)
        sample[sample == float("Inf")] = 0
        return sample

class ZarrLoader(Dataset):
    def __init__(self,
                 example_directory='./.tmp/training-data/',
                 train=True,
                 patch_size=None):
        self.example_directory = example_directory
        self.arr = dask.array.from_zarr(example_directory)
        self.N = self.arr.shape[0]
        self.train = train
        self.patch_size = patch_size

    def transform(self, sample):
        if np.random.uniform() > 0.5:
            sample = sample.flip(1)
        if np.random.uniform() > 0.5:
            sample = sample.flip(2)
        k = int((np.random.uniform()*4) % 4)
        if k > 0:
            sample = sample.rot90(k, (1,2))

        if self.patch_size is not None:
            ix = random.randint(0, sample.shape[1]-self.patch_size)
            iy = random.randint(0, sample.shape[2]-self.patch_size)
            sample = sample[:,ix:ix+self.patch_size,iy:iy+self.patch_size]

        return sample

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = self.arr[idx].compute()
        sample = torch.from_numpy(sample).float()
        mean = sample[torch.isfinite(sample)].mean()
        sample[~torch.isfinite(sample)] = mean
        if self.train:
            sample = self.transform(sample)
        return sample

class InterpLoader(Dataset):
    def __init__(self,
                 example_directory='.tmp/training-data/Channel-01/',
                 train=True,
                 patch_size=128,
                 mean=0,
                 std=1):

        directories = [os.path.join(example_directory, d) for d in os.listdir(example_directory)]
        self.datasets = [ZarrLoader(d, patch_size=patch_size, train=train) for d in directories]

        self.Ns = np.array([len(loader) for loader in self.datasets])
        self.N = sum(self.Ns)
        self.N_cumsum = np.cumsum(self.Ns)
        self.train = train
        self.mu = mean
        self.sd = std

    def sample_time(self, x):
        T = x.shape[0]
        gap = 10
        t0 = random.randint(0, T-gap-1)
        t1 = t0 + gap
        t = random.randint(t0+1, t1-1)
        tt = (t-t0) / (t1 - t0)
        return x[[t0, t, t1]], tt

    def statistics(self, n=100):
        print("Generating statistics")
        n_per = max([n // len(self.datasets),1])
        mu, sd = [], []
        for i, d in enumerate(self.datasets):
            rand_indices = np.random.choice(range(len(d)), n_per)
            d_mu, d_sd = [], []
            for j in rand_indices:
                d_mu.append(d[j].mean().numpy())
                d_sd.append(d[j].std().numpy())
            mu.append(np.nanmean(d_mu))
            sd.append(np.nanmean(d_sd))
        self.mu = np.nanmean(mu)
        self.sd = np.nanmean(sd)
        return self.mu, self.sd

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        prev_d = 0
        #mu, sd = self.statistics()
        for i, d in enumerate(self.N_cumsum):
            if idx < d:
                example = self.datasets[i][idx - prev_d]
                example = (example - self.mu) / self.sd
                example = self.sample_time(example)
                return example
            prev_d = d

class InterpLoaderMultitask(Dataset):
    def __init__(self, data_paths, bands, train=True, patch_size=256):

        self.datasets = []
        for path, b in zip(data_paths, bands):
            mean, std = get_band_stats(b)
            self.datasets.append(InterpLoader(path, patch_size=patch_size, train=train, mean=mean,
                                              std=std))

        self.Ns = np.array([len(loader) for loader in self.datasets])
        self.N = sum(self.Ns)
        self.N_cumsum = np.cumsum(self.Ns)
        self.train = train

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        prev_d = 0
        for i, d in enumerate(self.N_cumsum):
            if idx < d:
                return self.datasets[i][idx - prev_d]
            prev_d = d

if __name__ == '__main__':
    band = 1
    zarr_path = f'training-data/Interp-ABI-L1b-RadM-15min-264x264/Channel-{band:02g}/'
    mu, sd = get_band_stats(band)
    loader = InterpLoader(zarr_path, mean=mu, std=sd)
    print("Number of examples: {}".format(len(loader)))
    i = 1
    example = loader[i]

    import matplotlib.pyplot as plt
    plt.imshow(example[0][2].numpy())
    plt.show()
