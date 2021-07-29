import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


labels = ['1xK40', '1xV100', '2xV100', '4xV100', '8xV100']
ys = [42, 343, 479, 644, 600]
xs = list(range(1, len(labels)+1))

plt.Figure(figsize=(10,8))
plt.bar(xs, ys)
plt.title("Training Efficieny of GOES SloMo - PyTorch")
plt.xlabel("Hardware Configuration")
plt.ylabel("Examples/Second")
plt.xticks(xs, labels)
plt.savefig("nas_pytorch_performance.png", dpi=300)
plt.show()
