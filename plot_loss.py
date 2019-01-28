import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("saved-models/100-sample-mv/loss.txt")
df.columns = ['recon_loss', 'warp_loss', 'total_loss']
#df[['recon_loss', 'warp_loss', 'total_loss']].rolling(100).mean().plot()
df[['total_loss']].rolling(20).mean().plot()

print(df.head(20))
print(df.max())
plt.show()
