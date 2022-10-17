import os
import pandas as pd
import nibabel as nib
import numpy as np
import math
from torch.utils import data
import torch

class swi_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    def __init__(self, img_path,label_path,params):
        '''
        image_path: Path pointing to folder containing NifTi images
        label_path: csv-file containg information about iron measures
        '''
        # make label file ready first
        assert label_path.endswith('.csv')
        label_full_table = pd.read_csv(label_path)
        self.label_file = label_full_table[['ID',params['iron_measure']]]
        self.up_end = np.percentile(self.label_file.iloc[:, 1],99.5)
        self.low_end = np.percentile(self.label_file.iloc[:, 1],0.5)
        self.class_nb = params['class_nb']
        self.class_sz = (self.up_end - self.low_end) / params['class_nb']

        cnt_label = np.zeros(self.class_nb)
        for i in range(len(self.label_file.iloc[:, 1])):
            label_val = self.label_file.iloc[i, 1]
            if label_val >= self.up_end:
                cnt_label[-1] += 1
            elif label_val < self.low_end:
                cnt_label[0] += 1
            else:
                label_idx = math.floor( (label_val - self.low_end) / self.class_sz)
                cnt_label[label_idx] += 1
        
        self.label_weight = torch.zeros(self.class_nb)
        for i in range(len(self.label_weight)):
            self.label_weight[i] = np.sum(cnt_label) / cnt_label[i]

        # make image path a self var of dataset
        self.img_dir=img_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_file)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Create image path and read image
        img_path = os.path.join(self.img_dir, str(self.label_file.iloc[index, 0])+'_SWI.nii.gz')
        image = np.asarray(nib.load(img_path).get_fdata())
        image = image[np.newaxis, ...]

        # Create label information accordingly
        label_val = self.label_file.iloc[index, 1]
        label = torch.zeros(self.class_nb)
        if label_val >= self.up_end:
            label[-1] = 1
        elif label_val < self.low_end:
            label[0] = 1
        else:
            label_idx = math.floor( (label_val - self.low_end) / self.class_sz)
            label[label_idx] = 1
        """
        if self.label_file.iloc[index, 1] >= 31.52:
            label = torch.tensor([1, 0, 0])
        elif self.label_file.iloc[index, 1] >= 30.35:
            label = torch.tensor([0, 1, 0])
        else:
            label = torch.tensor([0, 0, 1])
        """
        #return both together
        return image, label, label_val, self.label_file.iloc[index, 0]
