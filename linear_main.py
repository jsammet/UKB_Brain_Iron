import os

import numpy as np
import cupy as cp
import pandas as pd
import nibabel as nib
import time

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
    #cp.cuda.Device(0).use()

    print("Read in data file")
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure'],'age','sex']]
    C_ = cp.expand_dims(cp.array(label_file.iloc[:, 1:]), axis=1)[:,0,:]
    result_img = np.empty([256,288,48])
    age_img = np.empty([256,288,48])
    sex_img = np.empty([256,288,48])

    step_start_time = time.time()
    Y_full = receive_voxel(label_file.iloc[:, 0])
    retrieve_time = time.time() - step_start_time
    print("Duration of voxel retrieval: ", retrieve_time)

    print("---------------------------------------------START TRAINING---------------------------------------------")
    for i in range(128): #256 divided by 4
        curr_vox = i*2
        print(f"Current state: {curr_vox} of 256")
        Y_ = cp.asarray(Y_full.data[:,curr_vox:curr_vox+2,:,:])
        Y_ = Y_.reshape(Y_.shape[0],-1)
        print(C_.shape)
        print(cp.transpose(C_).shape)
        C_t_C_inv = cp.linalg.inv(cp.matmul(cp.transpose(C_), C_))
        C_t_Y = cp.matmul(cp.transpose(C_), Y_)
        beta_ = cp.matmul(C_t_C_inv, C_t_Y)
        # results
        beta_r = cp.reshape(beta_[0], (2,288,48))
        print("Beta calculated, adding to result image")
        result_img[curr_vox:curr_vox+2,:,:] = cp.asnumpy(beta_r)
        # age
        beta_a = cp.reshape(beta_[1], (2,288,48))
        print("Adding to age image")
        age_img[curr_vox:curr_vox+2,:,:] = cp.asnumpy(beta_a)
        # sex
        beta_s = cp.reshape(beta_[2], (2,288,48))
        print("Adding to sex image")
        sex_img[curr_vox:curr_vox+2,:,:] = cp.asnumpy(beta_s)

    print("Save beta images")
    ni_img = nib.Nifti1Image(result_img, affine=np.eye(4))
    nib.save(ni_img, "results/linear_model_iron_map"+".nii")
    ni_img = nib.Nifti1Image(age_img, affine=np.eye(4))
    nib.save(ni_img, "results/linear_model_age_map"+".nii")
    ni_img = nib.Nifti1Image(sex_img, affine=np.eye(4))
    nib.save(ni_img, "results/linear_model_sex_map"+".nii")

print("Starting linear model evaluation")
main()
