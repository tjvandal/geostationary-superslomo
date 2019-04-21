import numpy as np
import torch

def psnr(img1, img2):
    mse = torch.mean((img1 - img2)**1)
    if mse == 0:
        return 100
    return 20 * torch.log10(1./mse)
