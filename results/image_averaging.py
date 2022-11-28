import os
import nibabel as nib
import numpy as np
import pandas as pd

test_df = pd.read_csv("test_IntGrad_hb_3class_60_0.0001_5e-07_hb_concent.csv")
test_elems = test_df['Name'].values

avg_img_class_0 = np.zeros([256,288,48])
cnt_cl_0 = 0
avg_img_class_1 = np.zeros([256,288,48])
cnt_cl_1 = 0
avg_img_class_2 = np.zeros([256,288,48])
cnt_cl_2 = 0

for subj in test_elems:
    #print("check")
    #print(subj)
    #print(test_df.loc[test_df['Name'] == subj, 'True_Label'].iloc[0])
    if test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0] != test_df.loc[test_df['Name'] == subj, 'True_Label'].iloc[0]:
        continue
    img_path = os.path.join('IntGrad_Maps', 'IntegratedGradient_'+str(subj)+'_hb_concent_3_classes.nii')
    nifti_img = nib.load(img_path)
    image = np.asarray(nifti_img.get_fdata())
    print(f"Image size: {image.shape}")
    if test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0] == 0:
        avg_img_class_0 = np.add(avg_img_class_0, image)
        cnt_cl_0 += 1
    elif test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0] == 1:
        avg_img_class_1 = np.add(avg_img_class_1, image)
        cnt_cl_1 += 1
    elif test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0] == 2:
        avg_img_class_2 = np.add(avg_img_class_2, image)
        cnt_cl_2 += 1

print(f"affinity matrix: {nifti_img.affine}")
aff_ = np.eye(4)
if cnt_cl_0 != 0:
    print(f"Image size: {avg_img_class_0.shape} and number {cnt_cl_0}")
    res_cl_0 = avg_img_class_0 / cnt_cl_0
    res_img = nib.Nifti1Image(res_cl_0, affine=aff_)
    nib.save(res_img,"average_salmap_class0.nii")
if cnt_cl_1 != 0:
    print(f"Image size: {avg_img_class_1.shape} and number {cnt_cl_1}")
    res_cl_1 = avg_img_class_1 / cnt_cl_1
    res_img = nib.Nifti1Image(res_cl_1, affine=aff_)
    nib.save(res_img,"average_salmap_class1.nii")
if cnt_cl_2 != 0:
    print(f"Image size: {avg_img_class_2.shape} and number {cnt_cl_2}")
    res_cl_2 = avg_img_class_2 / cnt_cl_2
    res_img = nib.Nifti1Image(res_cl_2, affine=aff_)
    nib.save(res_img,"average_salmap_class2.nii")