import torch
from torch import optim, cuda, nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    """(convolution => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv_block(x)

class Conv_Pool_Block(nn.Module):
    """Conv then downscale with Max Pool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.MaxPool3d(2),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Iron_NN(nn.Module):
    """
    TODO description
    """
    def __init__(self, channel_num = [32, 64, 128, 256, 256, 64], class_nb=3):

        super().__init__()

        # configure core unet model
        self.initial = Conv_Pool_Block(1,channel_num[0]) #24
        self.down1 = Conv_Pool_Block(channel_num[0], channel_num[1]) #12
        self.down2 = Conv_Pool_Block(channel_num[1], channel_num[2]) #6
        self.down3 = Conv_Pool_Block(channel_num[2], channel_num[3]) #3
        self.down4 = Conv_Block(channel_num[3], channel_num[4])
        self.down5 = Conv_Block(channel_num[4], channel_num[5])
        self.dropout = nn.Dropout(0.5)
        self.lastconv = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False)
        self.lastRELU = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(32*14*16*1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, class_nb)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x6_ = self.dropout(x6)
        x7 = self.lastconv(x6_)
        x7 = self.lastRELU(x7)
        x7_ = x7.view(-1, 32*14*16)
        x8 = self.fc1(x7_)
        x9 = self.fc2(x8)
        x = self.fc3(x9)
        return self.activation(x)
