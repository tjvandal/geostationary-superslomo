import matplotlib.pyplot as plt
import numpy as np

def plot_3channel_image(x, img_file=None):
    x_img = x[[1,2,0]]
    x_img = np.transpose(x_img, (1,2,0))
    if img_file is not None:
        plt.imsave(img_file, x_img)

    plt.imshow(x_img)
    plt.axis('off')

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

def plot_optical_flow(x, img_file=None):
    x = np.transpose(x, (1,2,0))
    bgr = opticalflow(x)
    f = plt.figure(figsize=(10,10))
    if img_file is not None:
        plt.imsave(img_file, bgr)

    plt.imshow(bgr)
    plt.axis('off')

def flow_quiver_plot(u, v, ax=None, title=None):
   # u = u.T
   # v = v.T
    x = np.arange(0, u.shape[1])
    y = np.arange(0, v.shape[0])
    X, Y = np.meshgrid(x, y)
    if not ax:
        fig, ax = plt.subplots(figsize=(20, 10))


    #ax.quiver(X, Y, flow[1,::-1,::-1], flow[0,::-1,::-1])
    uu = u.flatten()
    uu = uu[np.isfinite(uu)]
    uu[np.abs(uu) > 1e3] = 1.

    vv = v.flatten()
    vv = vv[np.isfinite(vv)]
    vv[np.abs(vv) > 1e3] = 1.

    ax.quiver(X, Y, u[::-1], v[::-1])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')
    ax.set_title(title)
