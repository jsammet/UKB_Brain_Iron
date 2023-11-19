"""
Model file of the Brain_Iron_NN.
Contains the Neural Network used by the system.
NN based on Peng et al. (2021).

Created by Joshua Sammet

Last edited: 03.01.2023
"""
import torch
from torch import optim, cuda, nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    """(Convolution => BatchNorm => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Conv_Pool_Block(nn.Module):
    """(Convolution => BatchNorm => MaxPool => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Conv_Block_noNorm(nn.Module):
    """
    Conv Block without batch normlization
    (Convolution => ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Conv_Pool_Block_noNorm(nn.Module):
    """
    Conv Block without batch normlization
    (Convolution => MaxPool => ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.MaxPool3d(2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class Iron_NN(nn.Module):
    """
    Network model containing batch norm in Conv-Blocks.
    Consists of:
    4 Convolution-Pooling-Layer
    3 Convolution-Layer
    1 DropOut
    Reordering of elements into 1D array
    3 FC-layers
    Softmax activation function
    
    Arguments:
    channel_num: Channel number for the different layers
    class_nb: Number of classes to be predicted
    """
    def __init__(self, channel_num = [32, 64, 128, 256, 256, 64], class_nb=3):

        super().__init__()

        # configure core model, image size: 176, 208, 176
        self.initial = Conv_Pool_Block(1, channel_num[0]) #88, 104
        self.down1 = Conv_Pool_Block(channel_num[0], channel_num[1]) #44, 52
        self.down2 = Conv_Pool_Block(channel_num[1], channel_num[2]) #22, 26
        self.down3 = Conv_Pool_Block(channel_num[2], channel_num[3]) #11, 13
        self.down4 = Conv_Block(channel_num[3], channel_num[4])
        self.down5 = Conv_Block(channel_num[4], channel_num[5])
        self.down6 = Conv_Block(channel_num[5], 32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*9*11*9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, class_nb)
        #self.activation = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.dropout(x7)
        x8_ = x8.view(-1, 32*9*11*9)
        x9 = self.fc1(x8_)
        x10 = self.fc2(x9)
        x = self.fc3(x10)
        return self.activation(x)

class Iron_NN_no_batch(nn.Module):
    """
    Network model not using batch norm in Conv-Blocks.
    Consists of:
    4 Convolution-Pooling-Layer
    3 Convolution-Layer
    1 DropOut
    Reordering of elements into 1D array
    3 FC-layers
    Sigmoid activation function
    
    Arguments:
    channel_num: Channel number for the different layers
    class_nb: Number of classes to be predicted
    """
    def __init__(self, channel_num = [32, 64, 128, 256, 256, 64], class_nb=3):

        super().__init__()

        # configure core model, image size: 176, 208, 176
        self.initial = Conv_Pool_Block_noNorm(1, channel_num[0]) #88, 104
        self.down1 = Conv_Pool_Block_noNorm(channel_num[0], channel_num[1]) #44, 52
        self.down2 = Conv_Pool_Block_noNorm(channel_num[1], channel_num[2]) #22, 26
        self.down3 = Conv_Pool_Block_noNorm(channel_num[2], channel_num[3]) #11, 13
        self.down4 = Conv_Block_noNorm(channel_num[3], channel_num[4])
        self.down5 = Conv_Block_noNorm(channel_num[4], channel_num[5])
        self.down6 = Conv_Block_noNorm(channel_num[5], 32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*9*11*9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, class_nb)
        # self.activation = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.dropout(x7)
        x8_ = x8.view(-1, 32*9*11*9)
        x9 = self.fc1(x8_)
        x10 = self.fc2(x9)
        x = self.fc3(x10)
        return self.activation(x)
