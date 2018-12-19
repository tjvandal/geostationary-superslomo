import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("saved-models/default/loss.txt")
df.columns = ['recon_loss', 'warp_loss', 'smooth_loss', 'total_loss']
df[['recon_loss', 'warp_loss', 'total_loss']].rolling(100).mean().plot()
print(df.head(20))
plt.show()
