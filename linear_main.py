import os

import numpy as np
import cupy as cp
import pandas as pd
import nibabel as nib

from src.linear_model import receive_voxel
from torch import nn, optim
import torch

def main():
    params = {
        'iron_measure': 'mean_corp_hb', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'image_path': '../SWI_images',
        'label_path': 'swi_brain_vol_info.csv'
    }
    print(params)
    cp.cuda.Device(0).use()

    print("Read in data file")
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure']]]

    result_img = torch.empty((256,288,48))

    C_ = cp.expand_dims(cp.array(label_file.iloc[:, 1]), axis=1)

    print("---------------------------------------------START TRAINING---------------------------------------------")
    for i in range(32): #instead 256, 8 elems at a time
        for j in range(288):
            print(f"Current state: {i*8}, {j}")
            Y_ = receive_voxel(label_file.iloc[:, 0],i*8,j)
            C_t_C_inv = cp.linalg.inv(cp.matmul(cp.transpose(C_), C_))
            C_t_Y = cp.matmul(cp.transpose(C_), Y_)
            beta_ = cp.matmul(C_t_C_inv, C_t_Y)
            result_img[i*8:i*8+8,j,:] = cp.reshape(beta_, (8,1,48))

        result_img = cp.asnumpy(result_img)
    print("Save beta image")
    ni_img = nib.Nifti1Image(result_img, affine=np.eye(4))
    nib.save(ni_img, "results/linear_model_loss_map"+".nii")
    # Test part
if __name__ == "__main__":
    main()
