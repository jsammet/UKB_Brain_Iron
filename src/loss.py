"""
Loss function of the Brain_Iron_NN.
Contains a very simple loss function.

Created by Joshua Sammet

Last edited: 03.01.2023
"""
import matplotlib.pyplot as pp
from torch import nn
import math
import pandas as pd
import numpy as np

class loss_func(nn.Module):
    """Returns Cross Entropy loss
    Class-like implementation of the default PyTorch nn.CrossEntropyLoss
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y):
        loss = self.loss(x, y) 
        return loss