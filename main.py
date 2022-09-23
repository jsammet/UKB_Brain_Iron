import os

import numpy as np
import pandas as pd

from src.model import Iron_CNN
from src.dataset import swi_dataset
from src.training import trainer
import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils import data

image_path='../SWI_images'
label_path='final_brain_vol_info.csv'
params = {
    'iron_measure': 'Hct_percent',
    'test_percent': 0.1,
    'val_percent': 0.04,
    'batch_size': 1,
    'nb_epochs': 100,
    'shuffle': False,
    'num_workers': 1,
    'channels': [32, 64, 128, 256, 256, 64]
}

# Create dataset
dataset= swi_dataset(image_path,label_path)
# Create model
model=Iron_CNN(params['channels'])

#create training-test split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(params['test_percent'] * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

#train model
model = trainer(model, dataset, train_indices, params)

# evaluate on test set
test_sampler = RandomSampler(train_indices)
test_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler)



