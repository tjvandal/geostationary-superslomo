import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import xarray as xr
import metpy

def plot_3channel_image(x, img_file=None, ax=None):
    if ax is None:
        ratio = 1.*x.shape[1] / x.shape[2]
        hi = int(ratio * 10.)
        wi = int(10.)

        fig = plt.figure(figsize=(wi,hi), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        
    x_img = x[[1,2,0]]
    x_img = np.transpose(x_img, (1,2,0))

    if img_file is not None:
        plt.imsave(img_file, x_img)

    ax.imshow(x_img)
    ax.axis('off')
    return ax

def plot_1channel_image(x, img_file=None, ax=None, vmax=None):
    if ax is None:
        ratio = 1.*x.shape[0] / x.shape[1]
        hi = int(ratio * 10.)
        wi = int(10.)

        fig = plt.figure(figsize=(wi,hi), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(x, vmax=vmax, cmap='jet')
    ax.axis('off')
    return ax

def get_conus_projection():
    dummy_file='/nex/datapoolne/goes16/ABI-L1b-RadC/2018/001/20/OR_ABI-L1b-RadC-M3C07_G16_s20180012002199_e20180012004583_c20180012005020.nc'
    ds_dummy = xr.open_dataset(dummy_file)
    return ds_dummy.metpy.parse_cf('Rad')

def plot_3channel_image_projection(da, ax=None):
    if ax is None:
        ratio = 1.*da.shape[1] / da.shape[2]
        hi = int(ratio * 10.)
        wi = int(10.)
        fig = plt.figure(figsize=(wi,hi), frameon=False)


    ds_dummy = get_conus_projection()

    R = da.sel(band=2).data
    G = da.sel(band=3).data
    B = da.sel(band=1).data


    RGB = np.stack([R, G, B], axis=2)
    geos = ds_dummy.metpy.cartopy_crs
    ax = fig.add_subplot(1, 1, 1, projection=geos)

    x = ds_dummy.x
    y = ds_dummy.y

    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()),
              transform=geos)
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.axis('off')
    return ax

def opticalflow(flow):
    hsv = np.ones((flow.shape[0], flow.shape[1], 3))*255.

    # Use Hue, Saturation, Value colour model 
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    return ang

    hsv[:,:, 0] = ang * 180 / np.pi / 2
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255., cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)
    return bgr

def downsample(x, pool=25):
    x = torch.from_numpy(x[np.newaxis])
    xlr = torch.nn.AvgPool2d(pool, stride=pool)(x)
    return xlr.detach().numpy()[0]

def plot_optical_flow(x, img_file=None):
    x = np.transpose(x, (1,2,0))
    bgr = opticalflow(x)
    f = plt.figure(figsize=(10,10))
    if img_file is not None:
        plt.imsave(img_file, bgr)

    plt.imshow(bgr)
    plt.axis('off')

def flow_quiver_plot(u, v, ax=None, down=50,vmax=None, background_img=None):
    intensity = (u**2 + v**2) ** 0.5

    u_l =  downsample(u, down)
    v_l =  downsample(v, down)

    intensity_l = ( u_l ** 2 + v_l**2 ) ** 0.5
    print("maximum intensity: {}".format(intensity.max()))
    u_l = u_l #/ intensity_l
    v_l = v_l #/ intensity_l

    x = np.arange(0, u_l.shape[1]) * down + down/2.
    y = np.arange(0, v_l.shape[0]) * down + down/2.
    X, Y = np.meshgrid(x, y)
    if not ax:
        ratio = 1.*u.shape[0] / u.shape[1]
        hi = int(ratio * 10.)
        wi = int(10.)
        fig = plt.figure(figsize=(wi,hi), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

    if background_img is not None:
        ax.imshow(background_img)
    else:
        ax.imshow(intensity, vmax=vmax)

    ax.quiver(X, Y, u_l, v_l, pivot='middle', width=0.002, headwidth=2, headlength=4,
              color='black', minlength=0.5)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')
    return ax
