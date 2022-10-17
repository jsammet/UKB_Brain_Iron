import torch
from torch import nn

class loss_func(nn.Module):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    def __init__(self, alpha):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()# KLDivLoss(reduction='sum')#nn.MSELoss()
        self.alpha = alpha

    def forward(self,x,y):
        #std_div = torch.std(x) - torch.std(y)
        #loss = self.loss(x, y) + self.alpha * std_div
        #print(loss)
        #y += 1e-16
        #n = y.shape[0]
        loss = self.loss(x, y) #/ n
        return loss
