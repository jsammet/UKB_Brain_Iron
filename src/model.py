import torch
from torch import optim, cuda, nn

import network_blocks as nb

class Iron_CNN(nn.Module):
    def __init__(self, bilinear=False):
        super().__init__()
        ndims = len(inshape)
        assert ndims in [3], 'ndims should be 3. found: %d' % ndims

        n_layer = len(channel_number)
        self.model_ = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.model_.add_module('conv_%d' % i,
                                                  nb(in_channel,
                                                                  out_channel,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.model_.add_module('conv_%d' % i,
                                                  nb(in_channel,
                                                                  out_channel,
                                                                  kernel_size=1,
                                                                  padding=0,
                                                                  final=True))

    def forward(self, x):
        x_c1, x1 = self.initial(x)
        x_c2, x2 = self.down1(x1)
        x_c3, x3 = self.down2(x2)
        x_c4, x4 = self.down3(x3)
        x5 = self.floor(x4)
        x = self.up1(x5, x_c4)
        x = self.up2(x, x_c3)
        x = self.up3(x, x_c2)
        x = self.up4(x, x_c1)
        return self.out(x)
