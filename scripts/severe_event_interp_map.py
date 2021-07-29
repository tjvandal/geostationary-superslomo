import os, sys
import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.goesr import GOESL1b, L1bBand
import interpolate

def main(args):
    ''' Plot mesoscale given band, datetime'''
    dayofyear = datetime.datetime(args.year, args.month, args.day).timetuple().tm_yday
    product = args.product #'ABI-L1b-RadM'
    T = args.step_size #10
    checkpoint = args.checkpoint #f'../model_weights/slomo-ind-v2/Channel-{band:02g}/best.flownet.pth.tar'

    geo = GOESL1b(channels=[args.band,], product=product)
    files = geo.local_files(args.year, dayofyear, hour=args.hour)
    files = files.xs(args.spatial, level='spatial')

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
    axs[2].imshow(Xt)
    axs[2].set_title("Predicted Frame")
    axs[2].axis('off')
    axs[3].imshow(((Xt-Xt_true)**2)**0.5)
    axs[3].axis('off')
    axs[3].set_title('Squared Error')
    
    plt.tight_layout()
    plt.savefig(f'figures/severe_event_map_{args.year}{args.month:02g}{args.day:02g}_{args.spatial}.png',
               dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="../model_weights/interp-ind/Channel-13/best.flownet.pth.tar", type=str)
    parser.add_argument("--model_name", default="unet-medium", type=str)
    parser.add_argument("--band", default=13, type=int)
    parser.add_argument("--year", default=2017, type=int)
    parser.add_argument("--day", default=10, type=int)
    parser.add_argument("--month", default=9, type=int)
    parser.add_argument("--hour", default=22, type=int)
    parser.add_argument("--step_size", default=10, type=int)
    parser.add_argument("--product", default='ABI-L1b-RadM', type=str)
    parser.add_argument("--spatial", default='RadM1', type=str)
    args = parser.parse_args()

    main(args)
