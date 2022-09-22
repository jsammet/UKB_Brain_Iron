import os

import numpy as np
import pandas as pd

from src.model import Iron_CNN
from src.dataset import swi_dataset
from src.file_read import file_read

image_path='path/to/swi/images'
label_path='path/to/swi/label.csv'
params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 1,
    'channels': [32, 64, 128, 256, 256, 64]
}

# Create dataset
dataset=swi_dataset(image_path,label_path)
# Create model
model=Iron_CNN(params['channels'])

#create training, validation and test set
#TODO here

#train model
#TODO create in seperate file

# evaluate on test set
#TODO here


