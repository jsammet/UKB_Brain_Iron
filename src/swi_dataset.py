import os
import pandas as pd
import nibabel as nib
import numpy as np
import math
from torch.utils import data
import torch
import pdb

class swi_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    def __init__(self,indices,params):
        '''
        image_path: Path pointing to folder containing NifTi images
        label_path: csv-file containg information about iron measures
        '''
        # make classes and their weights
        label_full_table = pd.read_csv(params['label_path'])
        self.label_file = label_full_table[['ID',params['iron_measure']]]
        self.up_end = np.percentile(self.label_file.iloc[:, 1],99.7)
        self.low_end = np.percentile(self.label_file.iloc[:, 1],0.3)
        self.class_nb = params['class_nb']
        self.class_sz = (self.up_end - self.low_end) / params['class_nb']
        
        val_ = np.empty(len(indices))
        class_ = np.empty(len(indices))
        for i in range(len(indices)):
            label_val = self.label_file.loc[self.label_file['ID'] == indices.item(i)].iloc[0,1]
            val_[i] = label_val
            if label_val >= self.up_end:
                class_[i] = int(params['class_nb'] - 1)
            elif label_val < self.low_end:
                class_[i] = 0
            else:
                label_idx = math.floor( (label_val - self.low_end) / self.class_sz)
                class_[i] = label_idx
        
        # make image path a self var of dataset
        self.img_dir=params['image_path']
        #make self lists
        self.idx_list = indices
        self.val_list = val_
        self.class_list = class_.astype('int32')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idx_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Create image path and read image
        #print("index of index: ", self.indices[index])
        img_path = os.path.join(self.img_dir, str(self.idx_list[index])+'_SWI.nii.gz')
        image = np.asarray(nib.load(img_path).get_fdata())
        image = image[np.newaxis, ...]

        # Create label information accordingly
        label_val = self.val_list[index]
        label = torch.zeros(self.class_nb)
        label[self.class_list[index]] = 1
        
        #return both together
        return image, label, label_val, self.idx_list[index]
