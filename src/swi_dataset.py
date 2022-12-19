import os
import pandas as pd
import nibabel as nib
import numpy as np
import math
import random

from scipy.ndimage.interpolation import rotate
from torch.utils import data
import torch
import pdb

class swi_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    def __init__(self,percentile_val,params):
        '''
        image_path: Path pointing to folder containing NifTi images
        label_path: csv-file containg information about iron measures
        '''
        # make classes and their weights 
        assert params['label_path'].endswith('.csv')
        label_full_table = pd.read_csv(params['label_path'])
        self.label_file = label_full_table[['ID',params['iron_measure']]]
        
        self.class_nb = params['class_nb']
        self.percent = percentile_val
        
        # make image path a self var of dataset
        self.img_dir=params['image_path']

        # parameters if images should be augmented
        self.flip = params['flip']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_file)

    def random_rotation_3d(self, image, max_angle):
        """ Randomly rotate an image by a random angle (-max_angle, max_angle).

        Arguments:
        max_angle: `float`. The maximum rotation angle.

        Returns:
        batch of rotated 3D images
        """
        size = image.shape
        batch_rot = np.zeros(image.shape)
        angle = np.zeros(3)
        if bool(random.getrandbits(1)):
            image1 = image
            # rotate along z-axis
            angle[0] = random.uniform(-max_angle, max_angle)
            image2 = rotate(image1, angle[0], mode='nearest', axes=(0, 1), reshape=False)
            # rotate along y-axis
            angle[1] = random.uniform(-max_angle, max_angle)
            image3 = rotate(image2, angle[1], mode='nearest', axes=(0, 2), reshape=False)
            # rotate along x-axis
            angle[2] = random.uniform(-max_angle, max_angle)
            batch_rot = rotate(image3, angle[2], mode='nearest', axes=(1, 2), reshape=False)
        else:
            batch_rot = image
        return batch_rot.reshape(size)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Create image path and read image
        #print("index of index: ", self.indices[index])
        img_path = os.path.join(self.img_dir, str(self.label_file.iloc[index, 0])+'_SWI.nii.gz')
        nifti_img = nib.load(img_path)
        image = np.asarray(nifti_img.get_fdata())

        # rotates 50% of images with maximum angle of 10°. Returns image and angles to undo rotation
        if self.flip == True:
            image = self.random_rotation_3d(image, 10)

        image = image[np.newaxis, ...]

        # Create label information accordingly
        label_val = self.label_file.iloc[index, 1]
        #label = self.label_file.iloc[index, 1]
        label = torch.zeros(self.class_nb)
        label_idx = np.sum(label_val > self.percent)
        label[label_idx] = 1
        
        #return both together
        return image, label, label_val, self.label_file.iloc[index, 0], nifti_img.affine

    