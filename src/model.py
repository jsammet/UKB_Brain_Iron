import torch
from torch import optim, cuda, nn
import torch.nn.functional as F

class Iron_NN(nn.Module):
    """
    3D CNN implementation for Iron level prediction based on local brain MRI
    Image input dimensions: 256 x 288 x48
    """
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(Iron_NN, self).__init__()
        #ndims = len(inshape)
        #assert ndims in [3], 'ndims should be 3. found: %d' % ndims

        n_layer = len(channel_number)

        self.convolve = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-2:
                self.convolve.add_module('conv_%d' % i,
                                                  Conv_Block(in_channel,
                                                                  out_channel,
                                                                  kernel_sz=3,
                                                                  padding_=1))
            else:
                self.convolve.add_module('conv_%d' % i,
                                                  Conv_Block(in_channel,
                                                                  out_channel,
                                                                  kernel_sz=1,
                                                                  padding_=0,
                                                                  no_pool=True))

        self.classifier = nn.Sequential()
        avg_shape = [16, 18, 3] # Image shape after Convolutions
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % 6,#i,
                                   nn.Conv3d(in_channel, 40, kernel_size=1, padding=0))

    def forward(self, input):
        x1 = self.convolve(input)
        x2 = self.classifier(x1)
        out =  F.log_softmax(x2, dim=1)
        return out

class Conv_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_sz=3, padding_=0, pool_stride=2, no_pool=False):
        super(Conv_Block, self).__init__()
        self.pool_bool = no_pool
        if no_pool == False:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sz, stride=1, padding=padding_)
            self.batch = nn.BatchNorm3d(out_channels)
            self.pool = nn.MaxPool3d(2, stride=pool_stride)
            self.ReLU = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sz, stride=1, padding=padding_)
            self.batch = nn.BatchNorm3d(out_channels)
            self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pool_bool == False:
            x = self.conv1(x)
            x = self. batch(x)
            x = self.pool(x)
            out = self.ReLU(x)
        else:
            x = self.conv1(x)
            x = self. batch(x)
            out = self.ReLU(x)
        return out
