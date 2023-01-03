"""
Main file to run the Linear Model for brain and iron.
It is run using the linear_run.sh bash script.
Uses receive_voxel file stored in /src dir.

Created by Joshua Sammet

Last edited: 03.01.2023
"""
import os

import numpy as np
import cupy as cp
import pandas as pd
import nibabel as nib
import time

from src.receive_voxel import receive_voxel
from scipy import stats
import pdb

# Small function to shuffle an array along a specific axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def main():
    # Predefined parameters
    params = {
        'iron_measure': 'Hct_percent', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'image_path': '../SWI_images',
        'label_path': 'stat_analysis/swi_brain_vol_info_additional.csv'
    }
    print(params)
    # set seeds
    np.random.seed(42)

    print("Read in data file")
    # Load datafile and extract import datapoints
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure'],'age','sex','head_vol','T1_SWI_diff','Scan_lat_X','Scan_trans_Y','Scan_long_Z','Scan_table_pos']]
    # Cut out ID from label file
    C_ = cp.array(label_file.iloc[:, 1:])
    # create X
    X_ = C_[:,0:]
    # Pre-calculation of X^T X --> save computation
    X_t_X_inv = cp.linalg.inv(cp.matmul(cp.transpose(X_), X_))
    # print(f"Nan in X_t_X_inv: {cp.any(cp.isnan(X_t_X_inv))}")  # Used to chcek for errors in calculation

    # Create empty numpy arrays to store sub sections of newly created images within linear model
    iron_img = np.empty([256,288,48])
    shuffle_img = np.empty([256,288,48])
    pval_img = np.empty([256,288,48])
    shuffle_pval_img = np.empty([256,288,48])
    age_img = np.empty([256,288,48])
    sex_img = np.empty([256,288,48])
    acc_img = np.empty([256,288,48])
    abs_acc_img = np.empty([256,288,48])
    shuffle_acc_img = np.empty([256,288,48])
    shuffle_abs_acc_img = np.empty([256,288,48])

    # create affinity matrix for nibabel to make images nicely viewable
    img_path = os.path.join('../SWI_images', '5363743'+'_SWI.nii.gz') # using subject with biggest head volume
    aff_img = nib.load(img_path)
    aff_mat = aff_img.affine

    step_start_time = time.time()
    # get voxel-images for all subjects and store in CPU RAM
    Y_full = receive_voxel(label_file.iloc[:, 0])
    retrieve_time = time.time() - step_start_time
    # print retrieval duration 
    print("Duration of voxel retrieval: ", retrieve_time)

    print("---------------------------------------------START TRAINING---------------------------------------------")
    # For GLM: calculate beta =(X^T X)^⁻1 X^T y
    # to speed up the calculation, it is not done per voxel but in groups of 4 voxels (more might be possible)
    for i in range(64): #256 divided by 4
        i_ = i*4
        print(f"slice {i_} of 256")
        
        # get sub-set of images in CUDA mode
        Y_ = cp.asarray(Y_full.data[:,i_:i_+4,:,:])
        # flatten Y_
        Y_2d = Y_.reshape(Y_.shape[0],-1)

        # Calculate X^T Y
        X_t_Y = cp.matmul(cp.transpose(X_), Y_2d)

        # Calculate beta and reshape
        beta_orig = cp.matmul(X_t_X_inv, X_t_Y)
        beta_ = beta_orig.reshape((X_.shape[1],4,288,48))

        # results: Store betas in the respective result image
        # Iron
        print("Adding to iron image")
        beta_r = beta_[0]
        iron_img[i_:i_+4,:,:] = cp.asnumpy(beta_r)
        # age
        print("Adding to age image")
        beta_a = beta_[1]
        age_img[i_:i_+4,:,:] = cp.asnumpy(beta_a)
        # sex
        print("Adding to sex image")
        beta_s = beta_[2]
        sex_img[i_:i_+4,:,:] = cp.asnumpy(beta_s)
        
        # Calculate results to be compared to Y_original
        res_ = cp.matmul(X_,beta_orig)
        res_ = res_.reshape((X_.shape[0],4,288,48))
        
        for j in range(4):
            j_ = i_+j
            for k in range(288):
                for l in range(48):
                    # p-value calculation based on glm formula in R 
                    if cp.std(beta_[:,j,k,l]) == 0:
                        # if betas are all zero, set pval to 1.
                        iron_pval = 1
                    else:
                        iron_pval= 2 * stats.norm.cdf(cp.asnumpy(-abs(beta_[0,j,k,l]) / cp.std(beta_[:,j,k,l])))
                    pval_img[j_,k,l] = iron_pval
                    
                    # Compare calculated and original Y and store difference and absolute difference
                    acc_img[j_,k,l] = np.median(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l])))
                    abs_acc_img[j_,k,l] = np.median(np.abs(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l]))))
        

    # Save all the images
    print("Save beta images")
    ni_img = nib.Nifti1Image(iron_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_iron_map"+".nii")
    ni_img = nib.Nifti1Image(age_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_age_map"+".nii")
    ni_img = nib.Nifti1Image(sex_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_sex_map"+".nii")

    ni_img = nib.Nifti1Image(pval_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_iron_pval"+".nii")
    ni_img = nib.Nifti1Image(acc_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_iron_acc"+".nii")
    ni_img = nib.Nifti1Image(abs_acc_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_iron_abs_acc"+".nii")

    print("---------------------------------------------START SHUFFLE---------------------------------------------")
    # Shuffle all images along first axis, i.e., shuffle across subjects, not within image
    Y_shuffle = shuffle_along_axis(np.asarray(Y_full.data), axis=0)
    # Do same calculations as above
    for i in range(64):
        i_ = i*4
        print(f"slice {i_} of 256")
        Y_ = cp.asarray(Y_shuffle[:,i_:i_+4,:,:])
        Y_2d = Y_.reshape(Y_.shape[0],-1)
        X_t_Y = cp.matmul(cp.transpose(X_), Y_2d)
        beta_orig = cp.matmul(X_t_X_inv, X_t_Y)
        beta_ = beta_.reshape((X_.shape[1],4,288,48))
        # Only save iron beta map, others could also be created but are not of interest
        print("Beta calculated, adding to shuffle iron image")
        beta_r = beta_[0]
        shuffle_img[i_:i_+4,:,:] = cp.asnumpy(beta_r)

        res_ = cp.matmul(X_,beta_orig)
        res_ = res_.reshape((X_.shape[0],4,288,48))
        for j in range(4):
            j_ = i_+j
            for k in range(288):
                for l in range(48):
                    # p-value calculation based on glm formula in R 
                    if cp.std(beta_[:,j,k,l]) == 0:
                        # if betas are all zero, set pval to 1.
                        iron_pval = 1
                    else:
                        iron_pval= 2 * stats.norm.cdf(cp.asnumpy(-abs(beta_[0,j,k,l]) / cp.std(beta_[:,j,k,l])))
                    shuffle_pval_img[j_,k,l] = iron_pval
                    
                    # Compare calculated and original Y and store difference and absolute difference
                    shuffle_acc_img[j_,k,l] = np.median(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l])))
                    shuffle_abs_acc_img[j_,k,l] = np.median(np.abs(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l]))))
    
    # Save shuffled images and shuffle results
    ni_img = nib.Nifti1Image(shuffle_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_shuffle_map"+".nii")
    ni_img = nib.Nifti1Image(shuffle_pval_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_shuffle_pval"+".nii")
    ni_img = nib.Nifti1Image(shuffle_acc_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_shuffle_acc"+".nii")
    ni_img = nib.Nifti1Image(shuffle_abs_acc_img, affine=aff_mat)
    nib.save(ni_img, "results/linear/linear_model_"+params['iron_measure']+"_shuffle_abs_acc"+".nii")
    example = np.ravel(iron_img)
    np.savetxt('results/linear/iron_'+params['iron_measure']+'_beta_lin_model.csv', example, delimiter=",")
    example = np.ravel(pval_img)
    np.savetxt('results/linear/iron_'+params['iron_measure']+'_pval_lin_model.csv', example, delimiter=",")
    example = np.ravel(shuffle_img)
    np.savetxt('results/linear/iron_'+params['iron_measure']+'_beta_shuffle.csv', example, delimiter=",")
    example = np.ravel(shuffle_pval_img)
    np.savetxt('results/linear/iron_'+params['iron_measure']+'_pval_shuffle.csv', example, delimiter=",")
            

print("Starting linear model evaluation")
main()
