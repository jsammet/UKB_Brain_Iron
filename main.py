import os

import numpy as np
import pandas as pd

from src.model import Iron_NN
from src.swi_dataset import swi_dataset
from src.training import trainer, tester
import torch


image_path='../SWI_images'
label_path='swi_brain_vol_info.csv'
params = {
    'iron_measure': 'Hct_percent',
    'test_percent': 0.1,
    'val_percent': 0.04,
    'batch_size': 1,
    'nb_epochs': 100,
    'shuffle': False,
    'num_workers': 1,
    'channels': [32, 64, 128, 256, 256, 64],
    'model_dir': 'src/',
    'device': 'cuda'
}

# Create dataset & model
dataset= swi_dataset(image_path,label_path,params)
model=Iron_NN(params['channels'])

#create training-test split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(params['test_percent'] * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

#train model
final_model = trainer(model, dataset, train_indices, params)

# evaluate on test set
tester(final_model, dataset, test_indices, params)
