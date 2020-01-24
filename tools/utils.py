import xarray as xr
import numpy as np
import cv2

import scipy.interpolate

def fillmiss(x):
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    newlength = int(len(x) * scale)
    y = np.linspace(x0, xlast, num=newlength, endpoint=False)
    return y

def interp_tensor(X, scale, fill=True, how=cv2.INTER_NEAREST):
    nlt = int(X.shape[1]*scale)
    nln = int(X.shape[2]*scale)
    newshape = (X.shape[0], nlt, nln)
    scaled_tensor = np.empty(newshape)
    for j, im in enumerate(X):
        # fill im with nearest neighbor
        if fill:
            #im = fillmiss(im)
            im[np.isnan(im)] = 0

        scaled_tensor[j] = cv2.resize(im, (newshape[2], newshape[1]),
                                     interpolation=how)
    return scaled_tensor

def interp_da(da, scale, how=cv2.INTER_LINEAR):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    tensor = da.values

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[2]].values, scale)

    # lets store our interpolated data
    scaled_tensor = interp_tensor(tensor, scale, fill=True, how=how)

    if latnew.shape[0] != scaled_tensor.shape[1]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, latnew, lonnew],
                 dims=da.dims)

def interp_da2d(da, scale, fillna=False, how=cv2.INTER_NEAREST):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    # lets store our interpolated data
    newshape = (int(da.shape[0]*scale),int(da.shape[1]*scale))
    im = da.values
    scaled_tensor = np.empty(newshape)
    # fill im with nearest neighbor
    if fillna:
        filled = fillmiss(im)
    else:
        filled = im
    scaled_tensor = cv2.resize(filled, dsize=(0,0), fx=scale, fy=scale,
                              interpolation=how)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[0]].values, scale)

    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[lonnew, latnew],
                 dims=da.dims)

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
