import os, sys
import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from goesr import GOESL1b, L1bBand
import interpolate

def main():
    ''' Plot mesoscale given band, datetime'''
    bands = [13,]
    year = 2017
    month = 3
    day = 6
    hour = 22
    dayofyear = datetime.datetime(year, month, day).timetuple().tm_yday
    spatial = 'RadM1'
    product = 'ABI-L1b-RadM'
    T = 10
    checkpoint = '/nobackupp10/tvandal/nex-ai-opticalflow/geo/models/slomo-ind-v2/Channel-13/best.flownet.pth.tar'

    geo = GOESL1b(channels=bands, product=product)
    files = geo.local_files(year, dayofyear, hour=hour)
    files = files.xs(spatial, level='spatial')

    l1b = L1bBand(files.values[0,0])
    l1b.open_dataset()
    l1b_10 = L1bBand(files.values[T,0])
    l1b_10.open_dataset()

    interp = interpolate.Interpolate(checkpoint)
    X1 = l1b.data['Rad'].values
    X2 = l1b_10.data['Rad'].values
    Xt = interp.predict(X1[np.newaxis], X2[np.newaxis], 0.5)

    l1b_5 = L1bBand(files.values[5,0])
    l1b_5.open_dataset()
    Xt_true = l1b_5.data['Rad'].values

    fig, axs = plt.subplots(2,2, figsize=(12,12))
    axs = np.ravel(axs)
    l1b.plot(ax=axs[0])
    l1b_10.plot(ax=axs[1])

    axs[2].imshow(Xt[0,0])
    axs[2].axis('off')
    axs[3].imshow(((Xt[0,0]-Xt_true)**2)**0.5)
    axs[3].axis('off')
    plt.show()

if __name__ == "__main__":
    main()
