import xarray as xr
import numpy as np
import scipy.interpolate
import dask as da

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    newlength = int(len(x) * scale)
    y = np.linspace(x0, xlast, num=newlength, endpoint=False)
    return y

def blocks(data, width=352):
    #n = data.t.shape[0]
    w = data.x.shape[0]
    h = data.y.shape[0]
    d = data.band.shape[0]

    hs = np.arange(0, h, width)
    ws = np.arange(0, w, width)
    blocks = []
    for hindex in hs:
        if hindex+width > h:
            hindex = h - width

        for windex in ws:
            if windex+width > w:
                windex = w - width
            blocks.append(data.sel(y=data.y.values[hindex:hindex+width],
                                    x=data.x.values[windex:windex+width]))
    return blocks

def block_array(arr, axis, size=128, stride=128):
    arr = da.array.swapaxes(arr, axis, 0)
    n = arr.shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr[j:j+size])
    stack = da.array.stack(stack)
    stack = da.array.swapaxes(stack, axis+1, 1)
    return stack

def interp(da, scale, fillna=False):
    xnew = interp_dim(da['x'].values, scale)
    ynew = interp_dim(da['y'].values, scale)
    newcoords = dict(x=xnew, y=ynew)
    return da.interp(newcoords)

def regrid_2km(da, band):
    if band == 2:
        return interp(da, 1. / 4, fillna=False)
    elif band in [1, 3, 5]:
        return interp(da, 1. / 2, fillna=False)
    return da

def regrid_1km(da, band):
    if band == 2: #(0.5 km)
        return interp(da, 1./2, fillna=False)
    elif band not in [1, 3, 5]: # 2km
        return interp(da, 2., fillna=False)
    return da

def regrid_500m(da, band):
    if band == 2: # 500m
        return da
    elif band in [1, 3, 5]: # 1km
        return interp(da, 2., fillna=False)
    return interp(da, 4., fillna=False) # 2km
