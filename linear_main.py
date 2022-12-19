import os

import numpy as np
import cupy as cp
import pandas as pd
import nibabel as nib
import time

from src.linear_model import receive_voxel
from scipy import stats
import pdb

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def main():
    params = {
        'iron_measure': 'Hct_percent', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'image_path': '../SWI_images',
        'label_path': 'stat_analysis/swi_brain_vol_info_additional.csv'
    }
    print(params)
    np.random.seed(42)
    #cp.cuda.Device(0).use()

    print("Read in data file")
    label_full_table = pd.read_csv(params['label_path'])
    label_file = label_full_table[['ID',params['iron_measure'],'age','sex','head_vol','T1_SWI_diff','Scan_lat_X','Scan_trans_Y','Scan_long_Z','Scan_table_pos']]
    C_ = cp.array(label_file.iloc[:, 1:])
    X_ = C_[:,0:] # create X
    X_t_X_inv = cp.linalg.inv(cp.matmul(cp.transpose(X_), X_))
    print(f"Nan in X_t_X_inv: {cp.any(cp.isnan(X_t_X_inv))}") 

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
    Y_full = receive_voxel(label_file.iloc[:, 0])
    retrieve_time = time.time() - step_start_time
    print("Duration of voxel retrieval: ", retrieve_time)

    print("---------------------------------------------START TRAINING---------------------------------------------")
    # calculate beta =(X^T X)^⁻1 X^T y
    for i in range(64): #256 divided by 4
        i_ = i*4
        print(f"slice {i_} of 256")
        Y_ = cp.asarray(Y_full.data[:,i_:i_+4,:,:])
        Y_2d = Y_.reshape(Y_.shape[0],-1)
        X_t_Y = cp.matmul(cp.transpose(X_), Y_2d)
        beta_orig = cp.matmul(X_t_X_inv, X_t_Y)
        beta_ = beta_orig.reshape((X_.shape[1],4,288,48))
        # results
        beta_r = beta_[0]
        #print("Beta calculated, adding to result image")
        iron_img[i_:i_+4,:,:] = cp.asnumpy(beta_r)
        # age
        beta_a = beta_[1]
        #print("Adding to age image")
        age_img[i_:i_+4,:,:] = cp.asnumpy(beta_a)
        # sex
        beta_s = beta_[2]
        #print("Adding to sex image")
        sex_img[i_:i_+4,:,:] = cp.asnumpy(beta_s)
        res_ = cp.matmul(X_,beta_orig)
        res_ = res_.reshape((X_.shape[0],4,288,48))
        
        for j in range(4):
            j_ = i_+j
            for k in range(288):
                for l in range(48):
                    if cp.std(beta_[:,j,k,l]) == 0:
                        iron_pval = 1
                    else:
                        iron_pval= 2 * stats.norm.cdf(cp.asnumpy(-abs(beta_[0,j,k,l]) / cp.std(beta_[:,j,k,l])))
                    # print(f"pval : {iron_pval}")
                    pval_img[j_,k,l] = iron_pval
                    
                    acc_img[j_,k,l] = np.median(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l])))
                    abs_acc_img[j_,k,l] = np.median(np.abs(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l]))))
                    # print(f"absolut difference in images : {abs_acc_img[j_,k,l]}")
        

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
    Y_shuffle = shuffle_along_axis(np.asarray(Y_full.data), axis=0)
    for i in range(64):
        i_ = i*4
        print(f"slice {i_} of 256")
        Y_ = cp.asarray(Y_shuffle[:,i_:i_+4,:,:])
        Y_2d = Y_.reshape(Y_.shape[0],-1)
        X_t_Y = cp.matmul(cp.transpose(X_), Y_2d)
        beta_orig = cp.matmul(X_t_X_inv, X_t_Y)
        beta_ = beta_.reshape((X_.shape[1],4,288,48))
        # results
        beta_r = beta_[0]
        #print("Beta calculated, adding to result image")
        shuffle_img[i_:i_+4,:,:] = cp.asnumpy(beta_r)

        res_ = cp.matmul(X_,beta_orig)
        res_ = res_.reshape((X_.shape[0],4,288,48))
        for j in range(4):
            j_ = i_+j
            for k in range(288):
                for l in range(48):
                    if cp.std(beta_[:,j,k,l]) == 0:
                        iron_pval = 1
                    else:
                        iron_pval= 2 * stats.norm.cdf(cp.asnumpy(-abs(beta_[0,j,k,l]) / cp.std(beta_[:,j,k,l])))
                    shuffle_pval_img[j_,k,l] = iron_pval
                    shuffle_acc_img[j_,k,l] = np.median(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l])))
                    shuffle_abs_acc_img[j_,k,l] = np.median(np.abs(cp.asnumpy(cp.subtract(Y_[:,j,k,l],res_[:,j,k,l]))))
    
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
