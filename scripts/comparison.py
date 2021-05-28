import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc("text", usetex=True)

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(sys.path[-1]))

from goesr import GOESL1b, L1bBand, GroupBandTemporal
import dataloader
from networks import unet
import interpolate

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def compare(predicted, label, data_range=None):
    mask = ~np.isfinite(label * predicted)
    label[mask] = 0. #np.nanmean(label)
    predicted[mask] = 0. #np.nanmean(predicted)

    res = dict()
    res['RMSE'] = np.nanmean((predicted - label)**2)**0.5
    
    if res['RMSE'] > 90:
        print("predicted", np.nanmean(predicted), "label", np.nanmean(label))
        err = (predicted-label).flatten()
        err = err[np.isfinite(err)]
        print(np.histogram(err))
        
        im = np.concatenate([label, predicted], 1)
        plt.imsave('figures/high-error.pdf', im)
        
    res['SSIM'] = structural_similarity(predicted, label)
    if data_range is not None:
        res['PSNR'] = peak_signal_noise_ratio(predicted, label, data_range=data_range)
    return res

def get_interpolators(band, stride=32, trim=32):
    models = dict()
    slomo_ind= f'/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/V2/interp-ind/'\
               f'Channel-{band:02d}/best.flownet.pth.tar'
    slomo_ms= f'/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/V2/interp-ind-ms/'\
               f'Channel-{band:02d}/best.flownet.pth.tar'
    slomo_ms2 =  f'/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/V1/interp-ms-large/'\
                 f'/Channel-{band:02d}/best.flownet.pth.tar'
    slomo_mt=  '/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/V2/global/'\
               'best.flownet.pth.tar'
    models['Slomo'] = interpolate.Interpolate(slomo_ind, nn_model=unet.UNetMedium, 
                                              stride=stride, trim=trim)
    models['Slomo-MS'] = interpolate.Interpolate(slomo_ms, nn_model=unet.UNetMultiscale, 
                                                 stride=stride, trim=trim)
    #models['Slomo-MS2'] = interpolate.Interpolate(slomo_ms2, nn_model=unet.UNetMultiscaleV2)
    models['Slomo-G'] = interpolate.Interpolate(slomo_mt, nn_model=unet.UNetMedium, 
                                                stride=stride, trim=trim)
    models['Linear'] = interpolate.Linear()
    #models['phase'] = interpolate.PhaseBased()

    #slomo_ind_ms= f'/nobackupp10/tvandal/nex-ai-opticalflow/geo/.tmp/models/slomo-ind-ms-v1/'\
    #              f'Channel-{band:02d}/best.flownet.pth.tar'
    #models['slomo_ind_ms'] = interpolate.Interpolate(slomo_ind, nn_model=unet.UNetMultiscale)

    return models

def random_dayhours(N=5, year=2019):
    days = list(range(1,200))
    hours = list(range(12,24))
    pairs = [(d, h) for d in days for h in hours]
    np.random.shuffle(pairs)
    return pairs[:N]

def get_dayhour_example(year, dayofyear, hour, band, T=10, t=0.5):
    geo = GOESL1b(channels=[band,], product='ABI-L1b-RadM')
    fs = geo.local_files(year, dayofyear, hour)
    if (fs.empty) or (len(fs) <= T):
        return None, None, None, None

    spatial = fs.index.levels[-1][0]
    fs = fs.xs(spatial, level='spatial')

    # make a temporal group
    I0 = L1bBand(fs.iloc[0]['file'][band])
    It = L1bBand(fs.iloc[int(T*t)]['file'][band])
    I1 = L1bBand(fs.iloc[T]['file'][band])

    group = GroupBandTemporal([I0, It, I1])
    if group is None:
        return None, None, None, None

    rad = group.get_radiances()
    if not hasattr(rad, 'values'):
        return None, None, None, None
    
    rad = rad.values
    N, H, W = rad.shape
    patch_size = 256
    ih = np.random.randint(0, H-patch_size)
    iw = np.random.randint(0, W-patch_size)

    rad = rad[:,ih:ih+patch_size,iw:iw+patch_size]

    if rad is not None:
        return rad[0], rad[1], rad[2], t

def interpolate_example(I0, I1, t, model, band):
    mu, sd = dataloader.get_band_stats(band)
    X1 = (I0[np.newaxis]-mu)/sd
    X2 = (I1[np.newaxis]-mu)/sd
    Xt_pred = model.predict(X1, X2, t)
    return Xt_pred * sd + mu

def example_statistics(year, day, hour, band, 
                       interpolators=None, T=10, 
                       random_crop_size=None,
                       trim=16, t=0.5):
    I0, It, I1, t = get_dayhour_example(year, day, hour, band, T=T, t=t)   # t=T/2
    if I0 is None:
        return
    if interpolators is None:
        interpolators = get_interpolators(band, trim=trim)
    if band >= 6:
        datarange = 350-190.
    else:
        datarange = 8.
        I0 /= 100.
        I1 /= 100.
        It /= 100.

    stats = []
    for name, model in interpolators.items():
        I_that = interpolate_example(I0, I1, t, model, band)
        print(f"Band: {band}, Name: {name}, It max={np.nanmax(It)}, Predicted max: {np.nanmax(I_that)}")
        if trim > 0:
            I_that = I_that[trim:-trim,trim:-trim]
            It_sub = It[trim:-trim,trim:-trim]
        res = compare(I_that, It_sub, datarange)
        res.update(dict(Model=name, year=year, day=day, hour=hour, Band=band, t=t, T=T))
        stats.append(res)
    return pd.DataFrame(stats)


def make_trange_plot():
    N = 1
    T = 10
    year = 2019
    trange = np.linspace(0.1, 0.9, 9)
    print("range of t", trange)
    bands = [1,]
    pairs = random_dayhours(N, year)
    b = bands[0]
    jobs = []
    for b in bands:
        for p in pairs:
            for t in trange:
                jobs.append(delayed(example_statistics)(year, p[0], p[1], b, T=T, t=t))
            
    collect_stats = Parallel(n_jobs=2)(jobs)
    print(collect_stats)
    all_stats = pd.concat(collect_stats)
    print(all_stats)
    

def main():
    N = 1
    T = 10
    year = 2019
    bands = [1,] #range(1,2)
    #bands = range(1, 17)

    pairs = random_dayhours(N, year)
    b = bands[0]
    jobs = []
    for b in bands:
        for p in pairs:
            jobs.append(delayed(example_statistics)(year, p[0], p[1], b, T=T))

    print("number of jobs", len(jobs))
    collect_stats = Parallel(n_jobs=12)(jobs)
    #collect_stats.append(p_stats)
    all_stats = pd.concat(collect_stats)
    table = pd.pivot_table(all_stats, values=['RMSE', 'SSIM', 'PSNR'], index=['Band'], columns=['Model'])
    
    table['Model'][table['Model'] == 'Slomo'] = 'SSM-T'
    table['Model'][table['Model'] == 'Slomo-MS'] = 'SSM-TMS'
    table['Model'][table['Model'] == 'Slomo-G'] = 'SSM-G'
    
    print(table.to_latex(float_format="{:0.3f}".format))
    return table

if __name__ == '__main__':
    make_trange_plot()
