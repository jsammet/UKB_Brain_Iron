import os
from src.swi_dataset import swi_dataset
import torch
import numpy as np
import pandas as pd
import time
import nibabel as nib

from .loss import loss_func
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from torch.utils import data
from captum.attr import GuidedGradCam

import pdb


def trainer(model, indices, params, optimizer, criterion, scheduler, percentile_val):
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

    #Create the datasets for train and validation
    train_dataset = swi_dataset(train_indices, percentile_val, params)
    val_dataset = swi_dataset(val_indices, percentile_val, params)
    
    # determine weights for WEightedRandomSampler
    values, counts = np.unique(train_dataset.class_list, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in train_dataset.class_list]
    
    # generate DataLoaders
    train_sampler = RandomSampler(range(len(train_indices))) #WeightedRandomSampler(example_weights, len(train_dataset.class_list))
    train_loader = data.DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=params['num_workers'])
    val_sampler = RandomSampler(range(len(val_indices)))
    val_loader = data.DataLoader(val_dataset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=params['num_workers'])
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

        for image, label_true, label_val, name in train_loader:

            image = image.float().to(device)
            label_true = label_true.long().to(device)
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
        print(f'Result of last validation items (avg): \t\t True class: {label_true[-1]} \t Prediction class: {torch.argmax(label_pred[-1]).item()}')
        print(f'Length of training set: \t\t { len(train_loader)}')
        # get compute time
        train_loss = train_loss / len(train_loader)
        print("train_loss: ", train_loss)
        epoch_step_time.append(time.time() - step_start_time)
        history['train_loss'].append(train_loss)

        ### VALIDATION
        print("---------------------------------------------START VALIDATION---------------------------------------------")
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
        # go thorugh one batch
            for image, label_true, label_val, name in val_loader:

                image = image.float().to(device)
                label_true = label_true.long().to(device)
                label_pred = model(image)
                # calculate total loss
                loss = criterion(label_pred.squeeze(1), label_true)
                valid_loss += loss

            # Leraning Rate Scheduler
            scheduler.step((valid_loss))

            # Print last validation results as example
            #example_diff = torch.sum(torch.sub(label_pred,label_true)).item() / label_pred.size(dim=0)
            print(f'Result of last validation items (avg): \t\t True class: {label_true} \t Prediction class: {label_pred}')
            # print epoch info
            valid_loss = valid_loss / len(val_loader)
            history['valid_loss'].append(valid_loss)
            print(f'Epoch {epoch} \t\t Training - Cross-Entropy Loss: {train_loss} \t\t Validation - CE Loss: {valid_loss}')
            epoch_info = 'Epoch %d/%d' % (epoch + 1, params['nb_epochs'])
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            print(' - '.join((epoch_info, time_info)), flush=True)

      # final model save
    torch.save(model.state_dict(),os.path.join(params['model_dir'], str(params['nb_epochs'])+'_'+str(params['class_nb'])+'class_'+ \
        "_"+str(params['lr'])+"_"+str(params['alpha'])+"_"+params['iron_measure']+'final%04d.pt' % epoch))
    return model, history

def tester(model, indices, params,criterion, percentile_val):
    # Put model into CUDA
    test_sampler = RandomSampler(indices)
    test_dataset = swi_dataset(indices, percentile_val, params)
    test_loader = data.DataLoader(test_dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=params['num_workers'])
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
        for image, label_true, label_val, name in test_loader:

            image = image.float().to(device)
            label_true = label_true.long().to(device)

            label_pred = model(image)
            for i in range(len(label_pred)):
                test_pred.append(torch.argmax(label_pred[i]).item())
                test_true.append(label_true[i].item())
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

def test_saliency(model, indices, params,criterion, percentile_val):
    """
    Test function that incldues saliency aps based on grads
    """
    # Put model into CUDA
    test_sampler = RandomSampler(indices)
    test_dataset = swi_dataset(indices, percentile_val, sparams)
    test_loader = data.DataLoader(test_dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=params['num_workers'])
    device = torch.device(params['device'])

    print("---------------------------------------------START SALIENCY MAP AND TEST---------------------------------------------")
    test_loss = 0.0
    model.eval()
    test_pred = []
    test_true = []
    with torch.no_grad():
    # go through test set
        for image, label_true, label_val, name in test_loader:
            image = image.float().to(device)
            label_true = label_true.long().to(device)
            label_true = label_true.argmax(dim=1)
            # Set the requires_grad_ to the image for retrieving gradients
            image.requires_grad_()
            label_pred = model(image)
            
            # GuidedGradCAM
            # ToDo: get correct labels and check attention map of correct labeled cases
            class_
            guided_gc = GuidedGradCam(model, model.down6)
            attribution = guided_gc.attribute(input, class_)

            loss = criterion(label_pred.squeeze(1), label_true)
            print(loss)
            test_loss += loss

    # Print last test results as example
    example_diff = torch.sum(torch.sub(label_pred,label_true)).item() / label_pred.size(dim=0)
    print(f'Result of last test items (avg): \t\t {example_diff}')
    #print test info
    test_loss = test_loss / len(test_loader)
    print(f'Final test result: \t\t Test set loss - MSELoss: {test_loss}')

    #df = pd.DataFrame(data={"ID": list(range(len(test_pred))), "Prediction": test_pred, "True_Label": test_true})
    #df.to_csv("results/test_corrected_"+str(params['nb_epochs'])+params['iron_measure']+".csv", sep=',',index=False)

    return 0
