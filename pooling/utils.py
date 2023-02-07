from torchvision import transforms, models
from sklearn.manifold import TSNE
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import random
import torch
import time
import copy
import os


def train_model(model, criterion, optimizer, scheduler, data, device, folder_name, num_epochs=25):

    epoch_num = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for batch in range(data.get_range(phase)):

#             for batch in range(5):

                inputs, labels = data.load_data(batch, phase)
                
#                 print(labels)
                
                
#                 agad
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.view(-1, 1))
                    
#                     print(outputs, loss)
# #                     
#                     aldgjla

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data.get_num(phase)
#             epoch_acc = running_corrects.double() / data.get_num(phase)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save each epoch that achieves a higher accuracy than the current best_acc in case the model crashes mid-training
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                        }, f"./{folder_name}/model_epoch{epoch}.torch")                
                
        epoch_num += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model