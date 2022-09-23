import torch
import numpy as np
from loss import loss_func
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils import data

def trainer(model, dataset, indices, params):
    '''
    Contains
    1. split of training set into train and validation
    2. training cycle for entwork

    Returns
    1. Best performing model
    '''
    torch.manual_seed(42)
    # Split of indices into train and validate
    dataset_size = len(indices)
    split = int(np.floor(params['val_percent'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # generate DataLoaders
    train_sampler = RandomSampler(train_indices)
    train_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    val_sampler = RandomSampler(val_indices)
    val_loader = data.DataLoader(dataset, batch_size=params['batch_size'], sampler=val_sampler)

    for epoch in range(0, params['nb_epochs']):
          print("Epoch: {}".format(epoch))
          # save model checkpoint
          if epoch % 20 == 0:
              torch.save(model.state_dict(),os.path.join(model_dir, '%04d.pt' % epoch))

          step_start_time = time.time()
          model.train()
          train_loss=0.0

          # go thorugh one batch
          for image, label_true in train_loader:
              
              image = y_src.to(device)
              label_pred = model(image)
              
              # calculate total loss
              loss = loss_func(label_pred, label_true)
              train_loss += loss

              # backpropagate and optimize
              optimizer.zero_grad()
              torch.autograd.backward(loss)
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
          for image, label_true in train_loader:

              image = y_src.to(device)
              label_pred = model(image)

              # calculate total loss
              loss = loss_func(label_pred, label_true)
              eval_loss += loss

          # Leraning Rate Scheduler
          scheduler.step((valid_loss[0]+valid_loss[1])/2)
          
          # print epoch info
          print(f'Epoch {epoch} \t\t Training - MSELoss: {train_loss} \t\t Validation - MSELoss: {valid_loss}')
          epoch_info = 'Epoch %d/%d' % (epoch + 1, nb_epochs)
          time_info = '%.4f sec/step' % np.mean(epoch_step_time)
          print(' - '.join((epoch_info, time_info)), flush=True)

      # final model save
      torch.save(model.state_dict(),os.path.join(model_dir, 'final%04d.pt' % epoch))
      return model
