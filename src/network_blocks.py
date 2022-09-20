import torch
from torch import optim, cuda, nn

class Conv_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_stride=2, bias=False, final=False):
        super().__init__()
        if final == False:
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
