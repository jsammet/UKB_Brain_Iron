import os
import torch
import numpy as np
import nibabel as nib

class receive_voxel():
    def __init__(self, indices):
        x_data = []
        for i in range(len(indices)):
            img_path = os.path.join('../SWI_images', str(indices[i])+'_SWI.nii.gz')
            image = np.asarray(nib.load(img_path).get_fdata())
            x_data.append(image)
        
        self.data = np.asarray(x_data)
