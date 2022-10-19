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
        self.loss = nn.CrossEntropyLoss(weights, reduction='mean') # KLDivLoss(reduction='sum')#nn.MSELoss()
        self.alpha = alpha

    def forward(self,x,y):
        #std_div = torch.std(x) - torch.std(y)
        #loss = self.loss(x, y) + self.alpha * std_div
        #print(loss)
        #y += 1e-16
        #n = y.shape[0]
        loss = self.loss(x, y) #/ n
        return loss

def weight_calc(indices, params):
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure']]]
    up_end = np.percentile(label_file.iloc[:, 1],99.7)
    low_end = np.percentile(label_file.iloc[:, 1],0.3)
    class_sz = (up_end - low_end) / params['class_nb']
    class_ = np.empty(len(indices))
    for i in range(len(indices)):
        label_val = label_file.loc[label_file['ID'] == indices.item(i)].iloc[0,1]
        if label_val >= up_end:
            class_[i] = int(params['class_nb'] - 1)
        elif label_val < low_end:
            class_[i] = 0
        else:
            label_idx = math.floor( (label_val - low_end) / class_sz)
            class_[i] = label_idx
    class_ = class_.astype('int32')

    return torch.from_numpy(compute_class_weight('balanced', np.unique(class_), class_)).float()
