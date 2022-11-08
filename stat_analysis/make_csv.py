import nibabel as nib
import numpy as np
import csv

image = np.asarray(nib.load("../results/linear_model_iron_map.nii").get_fdata())
example = image.reshape(256, -1)
np.savetxt('iron_beta_lin_model.csv', example, delimiter=",")