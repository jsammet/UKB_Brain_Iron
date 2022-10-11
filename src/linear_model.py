import os
import torch
import numpy as np
import cupy as cp
import nibabel as nib

def receive_voxel(indices,x,y):
    x_data = []
    for i in range(len(indices)):
        img_path = os.path.join('../SWI_images', str(indices[i])+'_SWI.nii.gz')
        image = cp.asarray(nib.load(img_path).get_fdata())
        x_data.append(cp.ravel(image[x:x+8,y,:]))
    x_data = cp.asarray(x_data)
    return x_data
