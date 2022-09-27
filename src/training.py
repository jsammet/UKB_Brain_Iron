import os
import torch
import numpy as np
import time

from .loss import loss_func
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils import data
from torchinfo import summary

def trainer(model, dataset, indices, params):
    '''
    Contains
    1. split of training set into train and validation
    2. Train the given model
    Returns
    1. Best performing model
    '''
    # Define parameters
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.KLDivLoss(reduction='sum')
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    if torch.cuda.is_available():
        model = model.cuda()
        model.to(torch.device(params['device']))
        criterion = criterion.cuda()

    # Print network model for good overview
    #print(summary(model, input_size=(params['batch_size'],1,256,288,48)))

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

    for epoch in range(0, params['nb_epochs']):
        print("Epoch: {}".format(epoch))
        # save model checkpoint
        if epoch % 20 == 0:
            torch.save(model.state_dict(),os.path.join(params['model_dir'], '%04d.pt' % epoch))

        print("---------------------------------------------START TRAINING---------------------------------------------")
        step_start_time = time.time()
        model.train()
        train_loss=0.0
        print('load images from DataLoader')
        # go through once
        for image, label_true in train_loader:

            image = image.float().cuda()
            label_true = label_true.float().cuda()
            print('Run model')
            label_pred = model(image)
            print('end model')

            # calculate total loss
            loss = criterion(label_pred, label_true)
            loss = loss.cuda()
            print(loss)
            train_loss += loss

            # backpropagate and optimize
            print('def optim')
            optimizer.zero_grad()
            print('torch autograd')
            #torch.autograd.grad(loss, image)
            loss.backward()
            print('oprim step')
            optimizer.step()

        # get compute time
        history_epoch_loss.append(loss)
        train_loss = train_loss / len(train_loader)
        epoch_step_time.append(time.time() - step_start_time)
        history['train_loss'].append(train_loss)

        ### VALIDATION
        print("---------------------------------------------START VALIDATION---------------------------------------------")
        valid_loss = 0.0
        model.eval()

        # go thorugh one batch
        for image, label_true in val_loader:

            image = image.float()
            label_true = label_true.float()
            label_pred = model(image)

            # calculate total loss
            loss = criterion(label_pred, label_true)
            valid_loss += loss

        # Leraning Rate Scheduler
        #scheduler.step((valid_loss)

        # print epoch info
        valid_loss = valid_loss / len(val_loader)
        history['valid_loss'].append(valid_loss)
        print(f'Epoch {epoch} \t\t Training - MSELoss: {train_loss} \t\t Validation - MSELoss: {valid_loss}')
        epoch_info = 'Epoch %d/%d' % (epoch + 1, nb_epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        print(' - '.join((epoch_info, time_info)), flush=True)

      # final model save
    torch.save(model.state_dict(),os.path.join(params['model_dir'], 'final%04d.pt' % epoch))
    return model

def tester(model, dataset, indices, params):
    test_sampler = RandomSampler(indices)
    test_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=params['num_workers'])

    print("---------------------------------------------START TEST---------------------------------------------")
    test_loss = 0.0
    model.eval()
    # go through test set
    for image, label_true in test_loader:

        image = image.to(device).float()
        label_true = label_true.to(device).float()
        label_pred = model(image)
        # calculate total loss
        loss = loss_func(label_pred, label_true)
        print(loss)
        test_loss += loss

    #print test info
    test_loss = test_loss / len(test_loader)
    print(f'Final test result: \t\t Test set loss - MSELoss: {test_loss}')

    return 0
