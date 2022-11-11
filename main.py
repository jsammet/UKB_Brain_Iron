import os

import numpy as np
import pandas as pd
import nibabel as nib
from captum.attr import Saliency

from src.model import Iron_NN
from src.swi_dataset import swi_dataset
from src.training import trainer, tester, test_saliency
from src.loss import loss_func, weight_calc
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import pdb
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    params = {
        'iron_measure':'hb_concent', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'test_percent': 0.1,
        'val_percent': 0.04,
        'batch_size': 20,
        'nb_epochs': 60,
        'num_workers': 16,
        'shuffle': True,
        'lr': 1e-4,
        'alpha': 5e-7,
        'class_nb': 3,
        'channels': [32, 64, 128, 256, 256, 64],
        'model_dir': 'src/models',
        'test_file': 'results/test_sal_hb_',
        'sal_maps': 'results/sal_maps',
        'image_path':'../SWI_images',
        'label_path':'swi_brain_vol_info.csv',
        'device': 'cuda',
        'sal_batch': 1,
        'sal_workers': 1,
        'activation_type': 'GuidedGradCam' # IntegratedGradient GuidedGradCam Occlusion
    }
    print(params)

    # Define parameters
    np.random.seed(420)
    torch.manual_seed(42)


    # Create data list
    label_full_table = pd.read_csv(params['label_path'])
    # Create classes
    value_list = label_full_table[params['iron_measure']].values
    class_sz = 100 / params['class_nb']
    class_per =  [class_sz*(c+1) for c in range(params['class_nb'] - 1)]
    percentile_val = [np.around(np.percentile(value_list,cl_), decimals=2) for cl_ in class_per]
    print(percentile_val)
    # Create dataset
    dataset= swi_dataset(percentile_val,params)
    # create training-test split
    dataset_size = len(dataset)
    print(f'Dataset length: {dataset_size}')
    indices = list(range(dataset_size))
    split = int(np.floor(params['test_percent'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # create tensor for class weights of loss
    #loss_weights = weight_calc(idx_list,percentile_val, params)

    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    if torch.cuda.is_available():
        device = torch.device(params['device'])
        model=Iron_NN(params['channels'],params['class_nb']).to(device)
        print(summary(model, input_size=(1,1,256,288,48))) # batch size set to 1 instead params['batch_size']
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        criterion = loss_func(params['alpha']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    else:
        raise Exception("Sorry, CUDA is neccessary.")

    # train model
    model, history = trainer(model, dataset, train_indices, params, optimizer, criterion, scheduler, percentile_val)
    df = pd.DataFrame(data={"ID": list(range(len(history['train_loss']))), "train_loss": history['train_loss'], "valid_loss": history['valid_loss']})
    df.to_csv("results/train_valid_" + params['activation_type'] + "_" +str(params['nb_epochs'])+'_'+str(params['class_nb'])+'class_'+ \
        "_"+str(params['lr'])+params['iron_measure']+".csv", sep=',',index=False)

    # evaluate on test set
    tester(model, dataset, test_indices, params, criterion)

    # saliency retrival
    test_saliency(model, dataset, test_indices, params)

if __name__ == "__main__":
    main()
