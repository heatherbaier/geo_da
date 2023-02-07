from torchvision import transforms, models
from sklearn.manifold import TSNE
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import argparse
import random
import torch
import time
import copy
import os

from dataloader import *
from utils import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type = str, required = True)
    args = parser.parse_args()
    
    records_dir = args.folder_name  
#     os.mkdir(records_dir)
#     os.mkdir(os.path.join(records_dir, "models"))        

    chlp = "../dimagery/tan/"
    crip = "../dimagery/phl/"
    ecup = "../dimagery/mex/"

    image_names = [chlp + _ for _ in os.listdir(chlp)] + [crip + _ for _ in os.listdir(crip)] + [ecup + _ for _ in os.listdir(ecup)]

    data = Dataloader(image_names, "../../da_gan/data/data_for_gan.csv", records_dir, batch_size = 32)

#     adgsa
    
    device = "cuda"
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)
    criterion = nn.L1Loss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    
    
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data, device, os.path.join(records_dir, "models"), num_epochs = 50)

    
    