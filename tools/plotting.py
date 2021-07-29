import matplotlib.pyplot as plt
import torch
import torch.nn
import numpy as np
from mpl_toolkits.basemap import Basemap
from metpy.plots import colortables
import cv2

def downsample(x, pool=25):
    x = torch.from_numpy(x[np.newaxis])
    xlr = torch.nn.AvgPool2d(pool, stride=pool)(x)
    return xlr.detach().numpy()[0]

def flow_quiver_plot(u, v, x=None, y=None, ax=None, 
                     down=25, vmax=None, latlon=False):
    intensity = (u**2 + v**2) ** 0.5

    u_l =  downsample(u, down)
    v_l =  downsample(v, down)

    intensity_l = ( u_l ** 2 + v_l**2 ) ** 0.5
    #intensity_l = ( u_l ** 2 + v_l**2 ) ** 0.25

    u_l = u_l #/ intensity_l
    v_l = v_l #/ intensity_l

    if x is None:
        x = np.arange(0, u_l.shape[1])  * down + down/2.
    if y is None:
        y = np.arange(0, v_l.shape[0])  * down + down/2.

    X, Y = np.meshgrid(x, y)
    
    #if X.shape[0] != u_l.shape[0]:
    #Y = cv2.resize(Y, dsize=v_l.shape, interpolation=cv2.INTER_LINEAR)
    #X = cv2.resize(X, dsize=u_l.shape, interpolation=cv2.INTER_LINEAR)

    
    if not ax:
        ratio = 1.*u.shape[0] / u.shape[1]
        hi = int(ratio * 10.)
        wi = int(10.)
        fig = plt.figure(figsize=(wi,hi), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
    if vmax is None:
        vmax = np.nanmax(intensity)
            
    ax.imshow(intensity, origin='upper', cmap='jet', vmax=vmax)
    ax.quiver(X, Y, v_l, u_l, pivot='middle', 
              width=0.001, headwidth=2, headlength=2)
    #ax.quiver(X, Y, u_l, v_l, pivot='middle', width=0.002, 
    #          headwidth=4, headlength=4)

    if hasattr(ax, 'xaxis'):
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
    return ax

def plot_infrared(data, x, y, sat_lon, ax=None, cmin=190, cmax=350):
    # The geostationary projection
    if ax is None:
        fig = plt.figure(figsize=[8,8], frameon=False)
        ax = fig.add_subplot(111)

    m = Basemap(projection='geos', lon_0=sat_lon, resolution='i',
                 rsphere=(6378137.00,6356752.3142),
                 llcrnrx=x.min(),llcrnry=y.min(),
                 urcrnrx=x.max(),urcrnry=y.max(),
                 ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    ax.axis('off')
    
    # Use a colortable/colormap available from MetPy
    ir_norm, ir_cmap = colortables.get_with_range('ir_drgb_r', cmin, cmax)
    # Plot the data using imshow
    im = m.imshow(data, origin='upper', cmap=ir_cmap, norm=ir_norm)
    plt.colorbar(im, pad=0, aspect=50, ticks=range(cmin,cmax,10), shrink=0.85)
    return ax