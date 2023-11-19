"""
Dataset class of the Brain_Iron_NN.
Provides a customized dataloading and dataset functionality

Created by Joshua Sammet

Last edited: 03.01.2023
"""
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

class t2_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    def __init__(self,percentile_val,params):
        '''
        percentile_val: Array containing boundary values of percentlies
        params: Parameter file containg information
        '''

        # ensure label file is csv
        assert params['label_path'].endswith('.csv')

        # Load info file and store ID and iron value
        label_full_table = pd.read_csv(params['label_path'])
        self.label_file = label_full_table[['ID',params['iron_measure']]]
        
        # store class number and percentile boundaries
        self.class_nb = params['class_nb']
        self.percent = percentile_val
        
        # store image path a self var of dataset
        self.img_dir=params['image_path']

        # parameter if images should be augmented
        self.flip = params['flip']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_file)

    def random_rotation_3d(self, image, max_angle):
        """ Randomly rotate an image by a random angle (-max_angle, max_angle).

        Arguments:
        image: Image to be rotated
        max_angle: `float`. The maximum rotation angle.

        Returns:
        batch of rotated 3D images
        """
        # Get iamge size and create dummy vars
        size = image.shape
        batch_rot = np.zeros(image.shape)
        angle = np.zeros(3)
        
        # 50% chance to rotate the image
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
            # pass original image if no rotation
            batch_rot = image

        return batch_rot.reshape(size)

    def __getitem__(self, index):
        """Generates one sample of data
        Arguments:
        index: Number of element in dataset

        Returns:
        image: swMRI of subject for respective index
        label: One-hot encoded iron measure of subject for respective index
        label_val: Original iron measure of subject for respective index
        self.label_file.iloc[index, 0]: ID of subject for respective index
        nifti_img.affine: affine transformation of NIFTI file of image (needed for activation map storage)
        """
        # Create image path and make image to numpy array
        img_path = os.path.join(self.img_dir, str(self.label_file.iloc[index, 0])+'_T2star.nii.gz')
        nifti_img = nib.load(img_path)
        image = np.asarray(nifti_img.get_fdata())

        # rotates 50% of images with maximum angle of 10°. Returns image
        if self.flip == True:
            image = self.random_rotation_3d(image, 10)

        # Expand image with a channel dimension
        image = image[np.newaxis, ...]

        # Create label information in one-hot encoding
        label_val = self.label_file.iloc[index, 1]
        label = torch.zeros(self.class_nb)
        label_idx = np.sum(label_val > self.percent)
        label[label_idx] = 1
        
        #return both together
        return image, label, label_val, self.label_file.iloc[index, 0], nifti_img.affine

    
