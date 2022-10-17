import os

import numpy as np
import pandas as pd
import nibabel as nib
from captum.attr import Saliency

from src.model import Iron_NN
from src.swi_dataset import swi_dataset
from src.training import trainer, tester, test_saliency
from src.loss import loss_func
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    image_path='../SWI_images'
    label_path='swi_brain_vol_info.csv'
    params = {
        'iron_measure':'mean_corp_hb', #'Hct_percent' 'hb_concent'  'mean_corp_hb'
        'test_percent': 0.1,
        'val_percent': 0.04,
        'batch_size': 30,
        'nb_epochs': 50,
        'shuffle': True,
        'num_workers': 20,
        'lr': 1e-4,
        'alpha': 5e-7,
        'class_nb': 3,
        'channels': [32, 64, 128, 256, 256, 64],
        'model_dir': 'src/models',
        'test_file': 'results/test_3class_',
        'device': 'cuda'
    }
    print(params)

    # Define parameters
    torch.manual_seed(42)

    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    if torch.cuda.is_available():
        device = torch.device(params['device'])
        model=Iron_NN(params['channels'],params['class_nb']).to(device)
        print(summary(model, input_size=(1,1,256,288,48))) # batch size set to 1 instead params['batch_size']
        model = nn.DataParallel(model)
        criterion = loss_func(params['alpha']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    else:
        raise Exception("Sorry, CUDA is neccessary.")

    # Create dataset
    dataset= swi_dataset(image_path,label_path,params)
    # create training-test split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(params['test_percent'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    print("model device main ", next(model.parameters()).get_device())
    # train model

    model, history = trainer(model, dataset, train_indices, params, optimizer, criterion, scheduler)
    df = pd.DataFrame(data={"ID": list(range(len(history['train_loss']))), "train_loss": history['train_loss'], "valid_loss": history['valid_loss']})
    df.to_csv("results/train_valid_"+str(params['nb_epochs'])+params['iron_measure']+".csv", sep=',',index=False)

    #model.load_state_dict(torch.load(os.path.join(params['model_dir'],'10hb_concentfinal0009.pt')))

    # evaluate on test set
    tester(model, dataset, test_indices, params, criterion)

    # saliency retrival
    test_saliency(model, dataset, test_indices, params, criterion)

if __name__ == "__main__":
    main()
