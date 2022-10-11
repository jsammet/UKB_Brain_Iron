import os
import pandas as pd
import nibabel as nib
import numpy as np
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
        label = self.label_file.iloc[index, 1]
        """
        if self.label_file.iloc[index, 1] >= 31.52:
            label = torch.tensor([1, 0, 0])
        elif self.label_file.iloc[index, 1] >= 30.35:
            label = torch.tensor([0, 1, 0])
        else:
            label = torch.tensor([0, 0, 1])
        """
        #return both together
        return image, label, self.label_file.iloc[index, 0]
