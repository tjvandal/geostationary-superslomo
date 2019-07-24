import matplotlib.pyplot as plt
import numpy as np
import torch


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

def plot_1channel_image(x, img_file=None, cmap=None, vmin=None, vmax=None):
    x_img = x.detach().numpy()[0,0]
    if img_file is not None:
        plt.imsave(img_file, x_img, cmap=cmap, vmin=vmin, vmax=vmax)
        
    plt.imshow(x_img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.show()

def plot_2channel_image(x, img_file=None):
    img1 = x.detach().numpy()[0,0]
    img2 = x.detach().numpy()[0,1]
    
    img = np.concatenate([img1, img2], axis=1)
    if img_file is not None:
        plt.imsave(img_file, img)
        
        
    plt.imshow(img)
    plt.axis('off')
    plt.show()

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

def flow_quiver_plot(u, v, ax=None, down=50):
    intensity = (u**2 + v**2) ** 0.5

    u_l =  downsample(u, down)
    v_l =  downsample(v, down)

    intensity_l = ( u_l ** 2 + v_l**2 ) ** 0.5
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

    ax.imshow(intensity)
    ax.quiver(X, Y, u_l, v_l, pivot='middle', width=0.001, headwidth=2, headlength=4)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')
    return ax
