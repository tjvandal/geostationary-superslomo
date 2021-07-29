import numpy as np
import torch
import torch.nn as nn

class BackwardWarp(nn.Module):
    def __init__(self):
        super(BackwardWarp, self).__init__()
        #self.device = device
    # end

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1)
            if torch.cuda.is_available():
                self.tensorGrid = self.tensorGrid.cuda()
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                 tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
        out = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                              mode='bilinear', padding_mode='border')
        return out
    # end

class BackwardWarpMV(nn.Module):
    def __init__(self):
        super(BackwardWarpMV, self).__init__()
        self.warper = BackwardWarp()

    def forward(self, tensorInput, tensorFlow):
        num_channels = tensorFlow.size(1)//2
        U = tensorFlow[:,:num_channels]
        V = tensorFlow[:,num_channels:]
        out = []
        for j in range(num_channels):
            #uv = tensorFlow[:,[j,j+num_channels],:,:]
            uv = torch.cat([U[:,j:j+1], V[:,j:j+1]], 1)
            out_j = self.warper(tensorInput[:,j:j+1], uv)
            out.append(out_j)

        return torch.cat(out, 1)
    
