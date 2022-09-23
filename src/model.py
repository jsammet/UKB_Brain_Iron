import torch
from torch import optim, cuda, nn
import torch.nn.functional as F

from . import network_blocks as nb

class Iron_NN(nn.Module):
    """
    3D CNN implementation for Iron level prediction based on local brain MRI 

    Image input dimensions: 256 x 288 x48

    """
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], dropout=True):
        super().__init__()
        ndims = len(inshape)
        assert ndims in [3], 'ndims should be 3. found: %d' % ndims

        n_layer = len(channel_number)
        self.convolve = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-2:
                self.convolve_.add_module('conv_%d' % i,
                                                  Conv_Block(in_channel,
                                                                  out_channel,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.convolve_.add_module('conv_%d' % i,
                                                  Conv_Block(in_channel,
                                                                  out_channel,
                                                                  kernel_size=1,
                                                                  padding=0,
                                                                  no_pad=True))
        self.classifier = nn.Sequential()
        avg_shape = [16, 18, 3] # Image shape after Convolutions
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, 40, padding=0, kernel_size=1))
        self.classifier.add_module('log_softmx', F.log_softmax(x, dim=1))


    def forward(self, x):
        
        x1 = self.convolve_(x)
        x2 = self.classifier(x1)
        return x2

class Conv_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_stride=2, bias=False, no_pad=False):
        super().__init__()
        if no_pad == False:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding, bias),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=pool_stride),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding, bias),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv_block(x)


