import os
from src.swi_dataset import swi_dataset
import torch
import numpy as np
import pandas as pd
import time
import nibabel as nib
from typing import Union

from .loss import loss_func
from torch.utils.data.sampler import RandomSampler
from torch.utils import data
from captum.attr import GuidedGradCam, IntegratedGradients, Occlusion, LayerGradCam, NoiseTunnel
from captum.attr._utils import visualization as viz
from captum.attr._utils import attribution as attr_
# Import M3d-CAM
from medcam import medcam

import pdb


def trainer(model, dataset, indices, params, optimizer, criterion, scheduler, percentile_val):
    '''
    Contains
    1. split of training set into train and validation
    2. Train the given model
    Returns
    1. Best performing model
    '''
    # Put device correctly
    device = torch.device(params['device'])
    model.to(device)
    criterion.to(device)

    # Split of indices into train and validate
    dataset_size = len(indices)
    split = int(np.floor(params['val_percent'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # generate DataLoaders
    train_sampler = RandomSampler(train_indices)
    train_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=params['num_workers'])
    val_sampler = RandomSampler(val_indices)
    val_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=params['num_workers'])
    history = {'train_loss': [], 'valid_loss': []}
    epoch_step_time = []

    for epoch in range(0, params['nb_epochs']):
        print("Epoch: {}".format(epoch))
        # save model checkpoint
        if epoch % 20 == 0:
            torch.save(model.state_dict(),os.path.join(params['model_dir'], str(params['nb_epochs'])+ \
            "_"+str(params['lr'])+"_"+str(params['alpha'])+"_"+params['iron_measure']+'%04d.pt' % epoch))


        print("---------------------------------------------START TRAINING---------------------------------------------")
        step_start_time = time.time()
        model.train()
        train_loss=0.0
        # go through once

        for image, label_true, label_val, name, aff_mat in train_loader:

            image = image.float().to(device)
            label_true = label_true.float().to(device)
            label_pred = model(image)
            # calculate total loss
            loss = criterion(label_pred.squeeze(1), label_true)
            # print(loss)
            train_loss += loss
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Print last validation results as example
        print(f'Result of last validation items (avg): \t\t True class: {label_true[-1]} \t Orig. value: {label_val[-1]} \t Prediction class: {torch.argmax(label_pred[-1]).item()}')
        print(f'Length of training set: \t\t { len(train_loader)}')
        # get compute time
        train_loss = train_loss / len(train_loader)
        print("train_loss: ", train_loss)
        epoch_step_time.append(time.time() - step_start_time)
        history['train_loss'].append(train_loss.item())

        ### VALIDATION
        print("---------------------------------------------START VALIDATION---------------------------------------------")
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
        # go thorugh one batch
            for image, label_true, label_val, name, aff_mat in val_loader:

                image = image.float().to(device)
                label_true = label_true.float().to(device)
                label_pred = model(image)
                # calculate total loss
                loss = criterion(label_pred, label_true) #loss = criterion(label_pred.squeeze(1), label_true)
                valid_loss += loss

            # Leraning Rate Scheduler
            scheduler.step((valid_loss))

            # Print last validation results as example
            #example_diff = torch.sum(torch.sub(label_pred,label_true)).item() / label_pred.size(dim=0)
            print(f'Result of last validation items (avg): \t\t True class: {label_true[-1]} \t Orig. value: {label_val[-1]} \t Prediction class: {label_pred[-1]}')
            # print epoch info
            valid_loss = valid_loss / len(val_loader)
            history['valid_loss'].append(valid_loss.item())
            print(f'Epoch {epoch} \t\t Training - Cross-Entropy Loss: {train_loss} \t\t Validation - CE Loss: {valid_loss}')
            epoch_info = 'Epoch %d/%d' % (epoch + 1, params['nb_epochs'])
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            print(' - '.join((epoch_info, time_info)), flush=True)

      # final model save
    torch.save(model.state_dict(keep_vars=True),os.path.join(params['model_dir'], str(params['nb_epochs'])+'_'+str(params['class_nb'])+'class_'+ \
        "_"+str(params['lr'])+"_"+str(params['alpha'])+"_"+params['iron_measure']+'final%04d.pt' % epoch))
    return model, history


def tester(model, dataset, indices, params,criterion):
    # Put model into CUDA
    test_sampler = RandomSampler(indices)
    test_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=params['num_workers'])
    device = torch.device(params['device'])

    print("---------------------------------------------START TEST---------------------------------------------")
    test_loss = 0.0
    model.eval()
    test_pred = []
    test_true = []
    test_val = []
    test_name = []
    with torch.no_grad():
    # go through test set
        for image, label_true, label_val, name, aff_mat in test_loader:

            image = image.float().to(device)
            label_true = label_true.float().to(device)

            label_pred = model(image)
            for i in range(len(label_pred)):
                test_pred.append(torch.argmax(label_pred[i]).item())
                test_true.append(torch.argmax(label_true[i]).item())
                test_val.append(label_val[i].item())
                test_name.append(name[i].item())
            # calculate total loss
            loss = criterion(label_pred, label_true) #loss = criterion(label_pred.squeeze(1), label_true)
            print(loss)
            test_loss += loss

    # Print last test results as example
    # example_diff = torch.sum(torch.sub(label_pred,label_true)).item() / label_pred.size(dim=0)
    print(f'Result of last test items (avg): \t\t True class: {label_true} \t Prediction class: {torch.argmax(label_pred).item()}')
    #print test info
    test_loss = test_loss / len(test_loader)
    print(f'Final test result: \t\t Test set loss - CE Loss: {test_loss}')

    df = pd.DataFrame(data={"ID": list(range(len(test_pred))),"Name": test_name, "Orig. true val": test_val, "Prediction": test_pred, "True_Label": test_true})
    df.to_csv(params['test_file']+"_"+params['activation_type']+"_"+params['iron_measure']+"_"+str(params['class_nb'])+'class_'+str(params['nb_epochs'])+ \
        "_"+str(params['lr'])+"_"+str(params['alpha'])+".csv", sep=',',index=False)

    return 0

def test_saliency(model, dataset, indices, params):
    """
    Test function that incldues saliency aps based on grads
    """
    # Put model into CUDA
    test_sampler = RandomSampler(indices)
    test_loader = data.DataLoader(dataset, batch_size=params['sal_batch'], sampler=test_sampler, num_workers=params['sal_workers'])
    device = torch.device(params['device'])

    print("---------------------------------------------START SALIENCY MAP AND TEST---------------------------------------------")
    USE_CUDA = True
    WORLD_SIZE = 5
    model.eval()
    if params['activation_type'] == 'GuidedGradCam':
        model = GuidedGradCam(model, model.module.lastconv, device_ids=[0, 1, 2, 3])
    elif params['activation_type'] == 'IntegratedGradient':
        model = IntegratedGradients(model,multiply_by_inputs=True)
    elif params['activation_type'] == 'NT_IntGrad':
        model = IntegratedGradients(model,multiply_by_inputs=True)
        model = NoiseTunnel(model)
    elif params['activation_type'] == 'Occlusion':
        model = Occlusion(model)
    elif params['activation_type'] == 'LayerGradCam':
        model = LayerGradCam(model,  model.module.lastconv, device_ids=[0, 1, 2, 3])
    elif params['activation_type'] == 'medcam':
        model = medcam.inject(model, backend='gcampp', layer='module.lastconv', return_attention=True) #, layer='module.lastconv'
    with torch.no_grad():
    # go through test set
        for image, label_true, label_val, name, aff_mat in test_loader:
            image = image.float().to(device)
            label_true = label_true.float().to(device)
            label_true = label_true.argmax(dim=1)
            # Set the requires_grad_ to the image for retrieving gradients
            image.requires_grad_().cpu()
            # Integrated Gradients
            if params['activation_type'] == 'GuidedGradCam':
                print(f"max. value of image {np.max(image.cpu().numpy())}, max. value of image {np.min(image.cpu().numpy())} and max. value of image {np.shape(image.cpu().numpy())}")
                attribution = model.attribute(image, target=label_true,interpolate_mode='trilinear')
            elif params['activation_type'] == 'IntegratedGradient':
                attribution = model.attribute(image, target=label_true, internal_batch_size=4)
            elif params['activation_type'] == 'NT_IntGrad':
                attribution = model.attribute(image, nt_type='smoothgrad', nt_samples=5, nt_samples_batch_size=1, internal_batch_size=6, target=label_true)
            elif params['activation_type'] == 'Occlusion':
                attribution = model.attribute(image, target=label_true, strides=(16,16,4),sliding_window_shapes=(1,32,32,8), show_progress=True)
            elif params['activation_type'] == 'LayerGradCam':
                attribution = model.attribute(image, target=label_true, relu_attributions=True)
                attribution = attr_.LayerAttribution.interpolate(attribution, (1, 1, 256,288,48))
            elif params['activation_type'] == 'medcam':
                attribution = model(image)
                attribution = attribution[1].squeeze(0).squeeze(0)
                attribution = attribution.cpu().numpy()
                print(np.any(np.isnan(attribution)))
            
            if params['activation_type'] != 'medcam':
                attribution = create_attr_map(attribution)
            
            attr_img = attribution
            #for i in range(len(label_true)):
            ni_img = nib.Nifti1Image(attr_img, affine=aff_mat[0].cpu().numpy())
            nib.save(ni_img, os.path.join(params['sal_maps'],params['activation_type'] + "_" + str(name.item())+"_"+params['iron_measure']+"_"+str(params['class_nb'])+"_classes"+".nii"))
            print(f"Finished f{str(name.item())}")

    return 0

# Took central concepts from captum.attr.visualization.visualize_image_attr
def create_attr_map(image, outlier_perc: Union[int, float] = 2):
    image = image.squeeze(0).squeeze(0)
    image = image.cpu().numpy()
    # _threshold = viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc)
    # img_norm = image / _threshold
    return viz._normalize_scale(image, viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc))
