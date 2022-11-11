import os
from src.swi_dataset import swi_dataset
import torch
import numpy as np
import pandas as pd
import time
import nibabel as nib

from .loss import loss_func
from torch.utils.data.sampler import RandomSampler
from torch.utils import data
from captum.attr import GuidedGradCam, IntegratedGradients, Occlusion
from captum.attr._utils import visualization as viz

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
                loss = criterion(label_pred.squeeze(1), label_true)
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
    torch.save(model.state_dict(),os.path.join(params['model_dir'], str(params['nb_epochs'])+'_'+str(params['class_nb'])+'class_'+ \
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
            loss = criterion(label_pred.squeeze(1), label_true)
            print(loss)
            test_loss += loss

    # Print last test results as example
    # example_diff = torch.sum(torch.sub(label_pred,label_true)).item() / label_pred.size(dim=0)
    print(f'Result of last test items (avg): \t\t True class: {label_true} \t Prediction class: {torch.argmax(label_pred).item()}')
    #print test info
    test_loss = test_loss / len(test_loader)
    print(f'Final test result: \t\t Test set loss - CE Loss: {test_loss}')

    df = pd.DataFrame(data={"ID": list(range(len(test_pred))),"Name": test_name, "Orig. true val": test_val, "Prediction": test_pred, "True_Label": test_true})
    df.to_csv(params['test_file']+str(params['class_nb'])+'class_'+str(params['nb_epochs'])+ \
        "_"+str(params['lr'])+"_"+str(params['alpha'])+"_"+params['iron_measure']+".csv", sep=',',index=False)

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
    model.eval()
    if params['activation_type'] == 'GuidedGradCam':
        activity_model = GuidedGradCam(model, model.module.lastconv, device_ids=[0, 1, 2, 3])
    elif params['activation_type'] == 'IntegratedGradient':
        activity_model = IntegratedGradients(model)
    elif['activation_type'] == 'Occlusion':
        activity_model = Occlusion(model)
    with torch.no_grad():
    # go through test set
        for image, label_true, label_val, name, aff_mat in test_loader:
            image = image.float().to(device)
            label_true = label_true.float().to(device)
            label_true = label_true.argmax(dim=1)
            # Set the requires_grad_ to the image for retrieving gradients
            image.requires_grad_()
            # Integrated Gradients
            if params['activation_type'] == 'GuidedGradCam':
                attribution = activity_model.attribute(image, target=label_true)
            elif params['activation_type'] == 'IntegratedGradient':
                attribution = activity_model.attribute(image, target=label_true, internal_batch_size=4)
            elif['activation_type'] == 'Occlusion':
                attribution = activity_model.attribute(image, target=label_true, sliding_window_shapes=(3,3))

            attribution = create_attr_map(attribution)
            
            attr_img = attribution.cpu().numpy()
            ni_img = nib.Nifti1Image(attr_img, affine=aff_mat.cpu().numpy()[0])
            nib.save(ni_img, os.path.join(params['sal_maps'],params['activation_type'] + "_" + str(name.item())+"_"+params['iron_measure']+"_"+str(params['class_nb'])+"_classes"+".nii"))

    return 0

# Took central concepts from captum.attr.visualization.visualize_image_attr
def create_attr_map(image, outlier_perc=2):
    image = image.squeeze(0).squeeze(0)
    threshold = viz._cumulative_sum_threshold(np.abs(image), 100 - outlier_perc)
    img_norm = image / threshold
    return np.clip(img_norm, -1, 1)
