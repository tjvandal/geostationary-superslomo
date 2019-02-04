import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

window_size = 50

df = pd.read_csv("saved-models/5min-samples-batch5-5bands/loss.txt")
df.columns = ['recon_loss', 'warp_loss', 'smooth_loss', 'total_loss']
#df[['recon_loss', 'warp_loss', 'total_loss']].rolling(100).mean().plot()
#df[['total_loss']].rolling(100).mean().plot()


#print(df.head(20))
#print(df.max())
#plt.show()


dfmv = pd.read_csv("saved-models/5min-samples-mv-5bands-novis/loss.txt")
dfmv.columns = ['recon_loss', 'warp_loss', 'total_loss']
#df[['recon_loss', 'warp_loss', 'total_loss']].rolling(100).mean().plot()

dfmv_roll = dfmv[['recon_loss']].rolling(window_size).mean()
#dfmv_roll.plot()

ymin = np.percentile(df['recon_loss'].values, 1.)
ymax= np.percentile(df['recon_loss'].values, 98.)
dfroll = df[['recon_loss']][:dfmv.shape[0]].rolling(window_size).mean()

print('slomo', dfroll.tail())
print('multi-variate', dfmv_roll.tail())
print("ymin", ymin, "ymax", ymax)
plt.plot(dfroll.values, label='rbg')
plt.plot(dfmv_roll.values, label='mv')
#plt.ylim([ymin,ymax])
plt.legend()
plt.show()
