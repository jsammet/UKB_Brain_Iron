"""
Training file of the Brain_Iron_NN.
Provides a training function, test function and fucntion to create saliency maps

Created by Joshua Sammet

Last edited: 03.01.2023
"""

import os
import warnings
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
from scipy.ndimage.interpolation import rotate


import pdb


def trainer(model, dataset, indices, params, optimizer, criterion, scheduler):
    '''
    Contains
    1. split of training set into train and validation
    2. Train the given model
    Returns
    1. Best performing model
    2. Arrays with train and validation loss
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
    val_sampler = RandomSampler(val_indices)
    train_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=params['num_workers'])
    val_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=params['num_workers'])

    # instantiate history dict and array for epoch timing
    history = {'train_loss': [], 'valid_loss': []}
    epoch_step_time = []

    for epoch in range(0, params['nb_epochs']):
        print("Epoch: {}".format(epoch))

        # Save models throughout the training process
        # save model checkpoint
        # if epoch % 20 == 0:
        #    torch.save(model.state_dict(),os.path.join(params['model_dir'], str(params['nb_epochs'])+ \
        #    "_"+str(params['lr'])+"_"+str(params['alpha'])+"_"+params['iron_measure']+'%04d.pt' % epoch))


        print("---------------------------------------------START TRAINING---------------------------------------------")
        step_start_time = time.time()
        model.train()
        # Initialize flip according to parameters
        dataset.flip = params['flip']
        # Initialize loss to 0
        train_loss=0.0

        for image, label_true, label_val, name, aff_mat in train_loader:
            # Read image and label, run model
            image = image.float().to(device)
            label_true = label_true.float().to(device)
            label_pred = model(image)
            # calculate total loss
            loss = criterion(label_pred.squeeze(1), label_true)
            # add loss to epoch loss
            train_loss += loss
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Print last validation results as example
        print(f'Result of last validation items (avg): \t\t True class: {label_true[-1]} \t Orig. value: {label_val[-1]} \t Prediction class: {torch.argmax(label_pred[-1]).item()}')
        print(f'Length of training set: \t\t { len(train_loader)}')
        # compute loss
        train_loss = train_loss / len(train_loader)
        print("train_loss: ", train_loss)
        history['train_loss'].append(train_loss.item())

        # Save time of epoch
        epoch_step_time.append(time.time() - step_start_time)

        ### VALIDATION
        print("---------------------------------------------START VALIDATION---------------------------------------------")
        model.eval()
        # Initialize flip to false, only used during training!
        dataset.flip = False
        # Initialize loss to 0
        valid_loss = 0.0

        with torch.no_grad():
        # go thorugh one batch
            for image, label_true, label_val, name, aff_mat in val_loader:
                # Read image and label, run model
                image = image.float().to(device)
                label_true = label_true.float().to(device)
                label_pred = model(image)
                # calculate loss
                loss = criterion(label_pred, label_true) 
                # add loss to epoch loss
                valid_loss += loss

            # Learning Rate Scheduler
            scheduler.step((valid_loss))

            # Print last validation results as example
            print(f'Result of last validation items (avg): \t\t True class: {label_true[-1]} \t Orig. value: {label_val[-1]} \t Prediction class: {label_pred[-1]}')
            # compute epoch loss
            valid_loss = valid_loss / len(val_loader)
            history['valid_loss'].append(valid_loss.item())

            # Print the train and validation loss and epoch info
            print(f'Epoch {epoch} \t\t Training - Cross-Entropy Loss: {train_loss} \t\t Validation - CE Loss: {valid_loss}')
            epoch_info = 'Epoch %d/%d' % (epoch + 1, params['nb_epochs'])
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            print(' - '.join((epoch_info, time_info)), flush=True)

    # save final model
    torch.save(model.state_dict(keep_vars=True),os.path.join(params['model_dir'], params['iron_measure'] + "_" + str(params['model'])+ "_model_" + str(params['flip']) + "_augment_" + \
    str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+ str(params['lr'])+"_lr_" +params['activation_type'] + 'final%04d.pt' % epoch))
    
    return model, history


def tester(model, dataset, indices, params,criterion):
    # Generate DataLoaders 
    test_sampler = RandomSampler(indices)
    test_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=params['num_workers'])
    # Put model onto devices - Probably unneccesary, but to be sure
    device = torch.device(params['device'])

    print("---------------------------------------------START TEST---------------------------------------------")
    model.eval()
    # Initialize flip to false, it is only used during training!
    dataset.flip = False
    # Initialize loss to 0
    valid_loss = 0.0
    # Create empty lists for additional info to be stored
    test_pred = []
    test_true = []
    test_val = []
    test_name = []

    with torch.no_grad():
        # go through test set
        for image, label_true, label_val, name, aff_mat in test_loader:
            # Read image and label, run model
            image = image.float().to(device)
            label_true = label_true.float().to(device)
            label_pred = model(image)

            # Store prediciton, correct class, original value and ID of subject
            for i in range(len(label_pred)):
                test_pred.append(torch.argmax(label_pred[i]).item())
                test_true.append(torch.argmax(label_true[i]).item())
                test_val.append(label_val[i].item())
                test_name.append(name[i].item())
            # calculate loss
            loss = criterion(label_pred, label_true) #loss = criterion(label_pred.squeeze(1), label_true)
            print(loss)
            # add loss to epoch loss
            test_loss += loss

    # Print last test results as example
    print(f'Result of last test items (avg): \t\t True class: {label_true} \t Prediction class: {torch.argmax(label_pred).item()}')
    # compute full test loss
    test_loss = test_loss / len(test_loader)
    print(f'Final test result: \t\t Test set loss - CE Loss: {test_loss}')

    # Save test results
    df = pd.DataFrame(data={"ID": list(range(len(test_pred))),"Name": test_name, "Orig. true val": test_val, "Prediction": test_pred, "True_Label": test_true})
    df.to_csv(params['test_file']+"_"+params['iron_measure'] + "_" + str(params['model'])+ "_model_" + str(params['flip']) + "_augment_" + \
    str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+ str(params['lr'])+"_lr_" +params['activation_type']+".csv", sep=',',index=False)

    return 0

def create_saliency(model, dataset, indices, params):
    """
    Function that creates attention maps, providing multiple options
    """
    # Put model into CUDA
    test_sampler = RandomSampler(indices)
    test_loader = data.DataLoader(dataset, batch_size=params['sal_batch'], sampler=test_sampler, num_workers=params['sal_workers'])
    device = torch.device(params['device'])

    print("---------------------------------------------START SALIENCY MAP AND TEST---------------------------------------------")
    model.eval()
    # Initialize flip to false, it is only used during training!
    dataset.flip = False
    # Initialize count to 0, helper variable for output
    cnt = 0

    # big if-elif to choose the activtion map mode 
    if params['activation_type'] == 'GuidedGradCam':
        print(model.module.down6)
        model = GuidedGradCam(model, model.module.down6[0], device_ids=[0, 1, 2, 3, 4])
    elif params['activation_type'] == 'IntegratedGradient':
        model = IntegratedGradients(model,multiply_by_inputs=True)
    # Integrated Gradient with NoiseTunnel, also additional parameter for additional flip
    elif params['activation_type'] == 'NT_IntGrad' or params['activation_type'] == 'flip_NT_IntGrad' or params['activation_type'] == 'NT_lowstd_IntGrad':
        model = IntegratedGradients(model,multiply_by_inputs=True)
        model = NoiseTunnel(model)
    elif params['activation_type'] == 'Occlusion':
        model = Occlusion(model)
    elif params['activation_type'] == 'LayerGradCam':
        model = LayerGradCam(model,  model.module.down6[0], device_ids=[0, 1, 2, 3,4])
    with torch.no_grad():
        # go through test set to create attention maps
        for image, label_true, label_val, name, aff_mat in test_loader:
            # load image
            image = image.float().to(device)
            # transform label from one-hot to numerical
            label_true = label_true.float().to(device)
            label_true = label_true.argmax(dim=1)
            # Set the requires_grad_ for the image for retrieving gradients
            image.requires_grad_().cpu()
            # if-elif to create the activtion map with the correct additional parameters 
            if params['activation_type'] == 'GuidedGradCam':
                attribution = model.attribute(image, target=label_true)
            elif params['activation_type'] == 'IntegratedGradient':
                attribution = model.attribute(image, target=label_true, internal_batch_size=4)
            # Integrated Gradient with NoiseTunnel. Additional 'flip_' for augmentation and easier use without altering output files
            elif params['activation_type'] == 'NT_IntGrad' or params['activation_type'] == 'flip_NT_IntGrad' or params['activation_type'] == 'NT_lowstd_IntGrad':
                attribution = model.attribute(image, nt_type='smoothgrad_sq', nt_samples=10, nt_samples_batch_size=1, internal_batch_size=6, target=label_true, stdevs=20.5)
            elif params['activation_type'] == 'Occlusion':
                attribution = model.attribute(image, target=label_true, strides=(16,16,4),sliding_window_shapes=(1,32,32,8), show_progress=True)
            elif params['activation_type'] == 'LayerGradCam':
                attribution = model.attribute(image, target=label_true, relu_attributions=True)
                attribution = attr_.LayerAttribution.interpolate(attribution, (256,288,48))
            
            # Normalize attention map and scale from 0 to 1
            print("Create attr map")
            attr_img = create_attr_map(attribution)

            # Save activation map as NIFTI
            ni_img = nib.Nifti1Image(attr_img, affine=aff_mat[0].cpu().numpy())
            nib.save(ni_img, os.path.join(params['sal_maps'],params['iron_measure'] + "_" + str(params['model'])+ "_model_" + str(params['flip']) + "_augment_" + \
                str(params['nb_epochs']) + "_eps_" + str(params['class_nb'])+'_class_'+params['activation_type']+".nii"))
            # Increase count and print progress
            cnt +=1
            print(f"Finished {str(name.item())}, image number {cnt}")

    return 0

# Took central concepts from captum.attr.visualization.visualize_image_attr
def create_attr_map(image, outlier_perc: Union[int, float] = 2):
    image = image.squeeze(0).squeeze(0)
    image = image.cpu().numpy()
    # _threshold = viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc)
    # img_norm = image / _threshold
    return normalize_scale(image, viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc))

# Took central concepts from captum.attr.visualization.normalize_scale
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