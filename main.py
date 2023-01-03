"""
Main file to run the Brain_Iron_NN.
It is run using the run.sh bash script.
Uses multiple files stored in /src dir.

Created by Joshua Sammet

Last edited: 03.01.2023
"""

import os

import numpy as np
import pandas as pd
import nibabel as nib
# Import the library
import argparse
# import from /src
from src.model import Iron_NN, Iron_NN_no_batch
from src.swi_dataset import swi_dataset
from src.training import trainer, tester, create_saliency
from src.loss import loss_func
# Import NN funtionalities 
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
# Import python debugger
import pdb



def main(iron_, class_, eps_, batch_, flip_, image_create_):
    # Define parameters of system
    params = {
        'iron_measure':'hb_concent', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'test_percent': 0.1,
        'val_percent': 0.04,
        'nb_epochs':100,
        'batch_size': 25, # normal 25
        'num_workers': 16, # normal 16
        'shuffle': True,
        'lr': 1e-4,
        'alpha': 5e-7,
        'class_nb': 10,
        'channels': [32, 64, 128, 256, 256, 64],
        'flip': False,
        'model': 'batch', # batch no_batch
        'model_dir': 'src/models',
        'test_file': 'results/test_',
        'sal_maps': 'results/nt_maps', # nt_maps nt_sq_maps GradCam_maps IntGrad_Maps GGC_maps
        'image_path':'../SWI_images',
        'label_path':'swi_brain_vol_info.csv',
        'device': 'cuda',
        'sal_batch': 1,
        'sal_workers': 1,
        'activation_type': 'NT_IntGrad', # IntegratedGradient GuidedGradCam Occlusion LayerGradCam NT_IntGrad flip_NT_IntGrad
        'create_maps': True
    }
    # Replace parameters using command line inputs
    params['iron_measure'] = iron_
    params['class_nb'] = class_
    params['nb_epochs'] = eps_
    params['model'] = batch_
    params['flip'] = flip_
    params['create_maps'] = image_create_

    #Print the parameter setup
    print(params)

    # Define seeds
    np.random.seed(420)
    torch.manual_seed(42)


    # Read data list
    label_full_table = pd.read_csv(params['label_path'])
    # Create perecntile values as borders for classes
    value_list = label_full_table[params['iron_measure']].values
    class_sz = 100 / params['class_nb']
    class_per =  [class_sz*(c+1) for c in range(params['class_nb'] - 1)]
    percentile_val = [np.around(np.percentile(value_list,cl_), decimals=2) for cl_ in class_per]
    print(percentile_val)
    # Create dataset
    dataset = swi_dataset(percentile_val,params)
    # create training-test split
    dataset_size = len(dataset)
    print(f'Dataset length: {dataset_size}')
    indices = list(range(dataset_size))
    split = int(np.floor(params['test_percent'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    ### CUDA has to be avaiable
    if torch.cuda.is_available():

        device = torch.device(params['device'])

        # Check if batch normalization is used and choose correct model
        if params['model'] == 'batch':
            model=Iron_NN(params['channels'],params['class_nb']).to(device)
        else:
            model=Iron_NN_no_batch(params['channels'],params['class_nb']).to(device)
        
        # Print a model picture
        print(summary(model, input_size=(1,1,256,288,48))) # batch size set to 1 instead params['batch_size']

        # Parallelize the model
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])

        # Implement elements for training
        criterion = loss_func(params['alpha']).to(device)   
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    else:
        raise Exception("Sorry, CUDA is neccessary.")
    
    # train model
    model, history = trainer(model, dataset, train_indices, params, optimizer, criterion, scheduler)
    # save train and valdiation loss
    df = pd.DataFrame(data={"ID": list(range(len(history['train_loss']))), "train_loss": history['train_loss'], "valid_loss": history['valid_loss']})
    df.to_csv("results/train_valid_" +"_"+params['iron_measure'] + "_" + str(params['model'])+ "_model_" + str(params['flip']) + "_augment_" + \
        str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+ str(params['lr'])+"_lr_" +params['activation_type']+".csv", sep=',',index=False)

    ### THIS CAN BE USED IF A ALREADY TRAINED SYSTEM SHOULD BE RETESTED --> comment out 3 lines above
    # model.load_state_dict(torch.load(os.path.join(params['model_dir'],'flip_False10010class__0.0001_5e-07_hb_concentfinal0099.pt')))
    # model.eval()

    # evaluate on test set
    tester(model, dataset, test_indices, params, criterion)
    
    # saliency retrival if indicated by command
    if params['create_maps'] == True:
        create_saliency(model, dataset, test_indices, params)




if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--iron', type=str, default=None)
    parser.add_argument('--class_nb', type=int, default=None)
    parser.add_argument('--eps', type=int, default=None)
    parser.add_argument('--batch', type=str, default=None)
    parser.add_argument('--augment', type=bool, default=None)
    parser.add_argument('--create_maps', type=bool, default=None)
    # Parse the argument
    args = parser.parse_args()

    main(args.iron, args.class_nb, args.eps, args.batch, args.augment, args.create_maps)
