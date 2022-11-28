import torch
import matplotlib.pyplot as pp
from torch import nn
import math
import pandas as pd
import numpy as np
from sklearn.utils import compute_class_weight

class loss_func(nn.Module):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    def __init__(self, alpha, weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss() #weights, reduction='mean') # KLDivLoss(reduction='sum')#nn.MSELoss()
        self.alpha = alpha

    def forward(self,x,y):
        #std_div = torch.std(x) - torch.std(y)
        #loss = self.loss(x, y) + self.alpha * std_div
        #print(loss)
        #y += 1e-16
        #n = y.shape[0]
        loss = self.loss(x, y) #/ n
        return loss

def weight_calc(indices,percentile_val, params):
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure']]]
    class_ = np.empty(len(indices))
    for i in range(len(indices)):
            label_val = label_file.loc[label_file['ID'] == indices.item(i)].iloc[0,1]
            class_[i] = np.sum(label_val < percentile_val)
    class_ = class_.astype('int32')

    return torch.from_numpy(compute_class_weight('balanced', np.unique(class_), class_)).float()

class my_KLDivLoss(nn.Module):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    def __init__(self, weights=None):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='sum')
        
    def forward(self,x,y):
        y += 1e-16
        n = y.shape[0]
        loss = self.loss(x, y) / n
        return loss