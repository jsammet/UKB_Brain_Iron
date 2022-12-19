### Create attention maps for a given network and a given list of subjects
import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
# Import the library

from src.model import Iron_NN
from src.loss import loss_func
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr._utils import visualization as viz

# Took central concepts from captum.attr.visualization.visualize_image_attr
def create_attr_map(image, outlier_perc: Union[int, float] = 2):
    image = image.squeeze(0).squeeze(0)
    image = image.cpu().numpy()
    # _threshold = viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc)
    # img_norm = image / _threshold
    return normalize_scale(image, viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc))
def normalize_scale(attr: np.ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, 0, 1)

def get_saliency(model, data_list, params):

    model = IntegratedGradients(model,multiply_by_inputs=True)
    model = NoiseTunnel(model)
    for i in len(data_list):
        name = data_list.iloc[i, 0]
        img_path = os.path.join(params['image_path'], str(name)+'_SWI.nii.gz')
        nifti_img = nib.load(img_path)
        image = np.asarray(nifti_img.get_fdata())
        image = image[np.newaxis,np.newaxis, ...] # add batch and channels

        # Create label information accordingly
        label_val = data_list.iloc[i, 1]
        #label = self.label_file.iloc[index, 1]
        label = torch.zeros(params['class_nb'])
        label_idx = np.sum(label_val > percentile_val)
        label[label_idx] = 1

        attribution = model.attribute(image, nt_type='smoothgrad_sq', nt_samples=10, nt_samples_batch_size=1, internal_batch_size=6, stdevs=20.5, target=label)
        attr_img = create_attr_map(attribution)

        ni_img = nib.Nifti1Image(attr_img, affine=nifti_img.affine.cpu().numpy())
        nib.save(ni_img, os.path.join(params['sal_maps'], "dementia_" + str(name)+"_"+params['iron_measure']+"_"+str(params['class_nb'])+"_classes"+".nii"))
        print(f"Finished {str(name)}")


params = {
        'iron_measure':'Hct_percent', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'test_percent': 0.1,
        'val_percent': 0.04,
        'nb_epochs': 100,
        'batch_size': 30,
        'num_workers': 20,
        'shuffle': True,
        'lr': 1e-4,
        'alpha': 5e-7,
        'class_nb': 3,
        'channels': [32, 64, 128, 256, 256, 64],
        'flip': False,
        'model_dir': 'src/models',
        'test_file': 'results/test_',
        'sal_maps': 'results/nt_sq_maps', # nt_maps nt_sq_maps GradCam_maps flip_maps IntGrad_Maps GGC_maps
        'image_path':'../SWI_images',
        'label_path':'stat_analysis/swi_brain_vol_info_additional.csv',
        'device': 'cuda',
        'sal_batch': 1,
        'sal_workers': 1,
        'activation_type': 'NT_IntGrad' # IntegratedGradient GuidedGradCam Occlusion LayerGradCam NT_IntGrad flip_NT_IntGrad
    }

if torch.cuda.is_available():
    device = torch.device(params['device'])
    model=Iron_NN(params['channels'],params['class_nb']).to(device)
    print(summary(model, input_size=(1,1,256,288,48))) # batch size set to 1 instead params['batch_size']
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
    criterion = loss_func(params['alpha']).to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
else:
    raise Exception("Sorry, CUDA is neccessary.")

label_full_table = pd.read_csv(params['label_path'])
# Create classes
data_list = label_full_table[['ID',params['iron_measure'],'dementia_source','ad_source']]
data_list.head()
data_list = data_list[data_list['dementia_source'] != -1].values

value_list = label_full_table[params['iron_measure']].values
class_sz = 100 / params['class_nb']
class_per =  [class_sz*(c+1) for c in range(params['class_nb'] - 1)]
percentile_val = [np.around(np.percentile(value_list,cl_), decimals=2) for cl_ in class_per]

model.load_state_dict(torch.load(os.path.join(params['model_dir'],'603class__0.0001_5e-07_hb_concentfinal0059.pt')))
model.eval()
# saliency retrival
get_saliency(model, data_list, percentile_val, params)