"""
Receive Voxel file used in the GLM.
Gathers Voxels of images from a list of subjects.

Created by Joshua Sammet

Last edited: 03.01.2023
"""
import os
import torch
import numpy as np
import nibabel as nib

class receive_voxel():
    def __init__(self, indices):
        # empty list to append elements
        x_data = []

        # per subject: load nifti file, get image and add to list
        for i in range(len(indices)):
            img_path = os.path.join('../SWI_images', str(indices[i])+'_SWI.nii.gz')
            image = np.asarray(nib.load(img_path).get_fdata())
            x_data.append(image)
        
        self.data = np.asarray(x_data)
