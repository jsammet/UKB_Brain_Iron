import os
import nibabel as nib
import numpy as np
import pandas as pd

import pdb

def create_avg_img(image, img_orig, avg_img_class, avg_orig_class, cnt_cl):
    avg_img_class = np.add(avg_img_class, image)
    avg_orig_class = np.add(avg_orig_class, img_orig)
    cnt_cl += 1
    return avg_img_class, avg_orig_class, cnt_cl

def create_nifti(avg_img_, cnt_, affine_):
    print(f"Image size: {avg_img_.shape} and number {cnt_}")
    res_cl = avg_img_ / cnt_
    res_img = nib.Nifti1Image(res_cl, affine=affine_)
    print(f"NIFTI size: {res_img.get_fdata().shape} and number {cnt_}")
    return res_img

np.random.seed(420)
folder_and_prefix="avgs/average_hb_concent_no_batch_" #avgs/average


test_df = pd.read_csv("test__NT_IntGrad_no_batch_model_hb_concent_3class_60_0.0001_5e-07.csv")
info_df = pd.read_csv("../stat_analysis/swi_brain_vol_info_cognition.csv")
test_elems = test_df['Name'].values
class_nb = 3
cogn_nb = 15

avg_img_class = []
avg_orig_class = []
cnt_cl = []
avg_img_sex = []
avg_orig_sex = []
cnt_sex = []
avg_img_sexcl = []
cnt_sexcl = []
avg_orig_sexcl = []

avg_img_cogn = []
avg_orig_cogn = []
cnt_cogn = []

for i in range(class_nb):
    avg_img_class.append(np.zeros([256,288,48]))
    avg_orig_class.append(np.zeros([256,288,48]))
    cnt_cl.append(0)

for i in range(cogn_nb):
    avg_img_cogn.append(np.zeros([256,288,48]))
    avg_orig_cogn.append(np.zeros([256,288,48]))
    cnt_cogn.append(0)

# Sex
for j in range(2):
    avg_img_sex.append(np.zeros([256,288,48]))
    avg_orig_sex.append(np.zeros([256,288,48]))
    cnt_sex.append(0)
    # sex per class in order 00, 01, 02, 10, 11, 12
    for i in range(class_nb):
        avg_img_sexcl.append(np.zeros([256,288,48]))
        avg_orig_sexcl.append(np.zeros([256,288,48]))
        cnt_sexcl.append(0)

for subj in test_elems:
    if test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0] != test_df.loc[test_df['Name'] == subj, 'True_Label'].iloc[0]:
        continue
    img_path = os.path.join('nt_sq_maps', 'NT_IntGrad_no_batch_model_'+str(subj)+'_hb_concent_3_classes.nii')
    nifti_img = nib.load(img_path)
    image = np.asarray(nifti_img.get_fdata())

    orig_path = os.path.join('..', '..','SWI_images', str(subj)+'_SWI.nii.gz')
    nifti_orig = nib.load(orig_path)
    img_orig = np.asarray(nifti_orig.get_fdata())

    print(f"Image shape: {image.shape}")
    # create class maps
    class_val = test_df.loc[test_df['Name'] == subj, 'Prediction'].iloc[0]
    avg_img_class[class_val], avg_orig_class[class_val], cnt_cl[class_val] = create_avg_img(image, img_orig, avg_img_class[class_val], avg_orig_class[class_val], cnt_cl[class_val])
    # Create sex maps
    sex_val = info_df.loc[info_df['ID'] == subj, 'sex'].iloc[0]
    avg_img_sex[sex_val], avg_orig_sex[sex_val], cnt_sex[sex_val] = create_avg_img(image, img_orig, avg_img_sex[sex_val], avg_orig_sex[sex_val], cnt_sex[sex_val])
    # Create sex-class interaction map
    sex_cl_val = sex_val*(class_nb) + class_val
    avg_img_sexcl[sex_cl_val], avg_orig_sexcl[sex_cl_val], cnt_sexcl[sex_cl_val] = create_avg_img(image, img_orig, avg_img_sexcl[sex_cl_val], avg_orig_sexcl[sex_cl_val], cnt_sexcl[sex_cl_val])
  
    # Create cognition map
    cog_val = info_df.loc[info_df['ID'] == subj, 'fluid_int_v2'].iloc[0]
    avg_img_cogn[cog_val], avg_orig_cogn[cog_val], cnt_cogn[cog_val] = create_avg_img(image, img_orig, avg_img_cogn[cog_val], avg_orig_cogn[cog_val], cnt_cogn[cog_val])

aff_path = os.path.join('..', '..','SWI_images', '1214900_SWI.nii.gz')
nifti_aff = nib.load(aff_path)
print(f"affinity matrix: {nifti_aff.affine} of subject: {str(subj)}, with orientation: {nib.aff2axcodes(nifti_aff.affine)}") # 1214900 1361222 nifti_orig
affine_ = nifti_aff.affine

print("Create class maps")
for i in range(class_nb):
    res_img = create_nifti(avg_img_class[i], cnt_cl[i], affine_) 
    nib.save(res_img,folder_and_prefix+"_salmap_class_"+str(class_nb)+"cl_"+str(i)+".nii")
    # Original image for this calss to check distortion
    res_orig = create_nifti(avg_orig_class[i], cnt_cl[i], affine_)
    nib.save(res_orig,folder_and_prefix+"_original_class_"+str(class_nb)+"cl_"+str(i)+".nii")

# Sex

for j in range(2):
    print("Create sex maps")
    res_img = create_nifti(avg_img_sex[j], cnt_sex[j], affine_)
    nib.save(res_img,folder_and_prefix+"_salmap_sex_"+str(class_nb)+"cl_"+str(j)+".nii")
    # Original image for this calss to check distortion
    res_orig = create_nifti(avg_orig_sex[j], cnt_sex[j], affine_)
    nib.save(res_orig,folder_and_prefix+"_original_sex_"+str(class_nb)+"cl_"+str(j)+".nii")
    print("Create class-sex maps")
    for i in range(class_nb):
        s_c_v = j*class_nb + i
        res_img = create_nifti(avg_img_sexcl[s_c_v], cnt_sexcl[s_c_v], affine_) 
        nib.save(res_img,folder_and_prefix+"_salmap_sexclass_"+str(class_nb)+"cl_"+str(j)+str(i)+".nii")
        # Original image for this calss to check distortion
        res_orig = create_nifti(avg_orig_sexcl[s_c_v], cnt_sexcl[s_c_v], affine_)
        nib.save(res_orig,folder_and_prefix+"_original_sexclass_"+str(class_nb)+"cl_"+str(j)+str(i)+".nii")

print("Create cognition maps")
for i in range(cogn_nb):
    res_img = create_nifti(avg_img_cogn[i], cnt_cogn[i], affine_) 
    nib.save(res_img,folder_and_prefix+"_salmap_cognition_"+str(cogn_nb)+"cl_"+str(i)+".nii")
    # Original image for this calss to check distortion
    res_orig = create_nifti(avg_orig_cogn[i], cnt_cogn[i], affine_)
    nib.save(res_orig,folder_and_prefix+"_original_cognition_"+str(cogn_nb)+"cl_"+str(i)+".nii")