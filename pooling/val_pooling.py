from sklearn.metrics import mean_absolute_error, r2_score
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
    
    
    

#     chlp = "./imagery/"
    with open("./pooled_model/valimages.txt", "r") as f:
        image_names = f.read().splitlines()
    image_names = [i for i in image_names if ".ipynb" not in i]
#     print(image_names)
    
    data = Valoader(image_names, "./pooled_data.csv", records_dir, batch_size = 32)

    print(data.load_data(2))
    
    weights = records_dir + "/models/" + os.listdir(records_dir + "/models/")[-1]
    weights = torch.load(weights)["model_state_dict"]
#     print(weights)
    
#     asdga
    
    device = "cuda"
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft.load_state_dict(weights)
    model_ft = model_ft.to(device)
    criterion = nn.L1Loss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    
    
#     dsagkla

    preds, trues, ids = [], [], []
        
    for idx in range(len(image_names)):
        
#     for idx in range(5):
        
        try:
        
            cur = data.load_data(idx)

            pred = model_ft(cur[0].to(device))
            
#             print(cur[1], cur[1].shape)

            trues.append(cur[1].item())

            preds.append(pred.item())
            ids.append(image_names[idx])


            print(pred)
            
        except Exception as e:
            
            print("ERROR: ", e)
            
            print(image_names[idx])
            
# print(preds)
# print(trues)
# print(ids)
        

df = pd.DataFrame()
df["school_id"] = ids
df["pred"] = preds
df["true"] = trues
print(df.head())

mae = mean_absolute_error(df["true"], df["pred"])
r2 = r2_score(df["true"], df["pred"])

print("MAE: ", mae)
print("R2: ", r2)

df.to_csv("./pooled_preds_v1.csv", index = False)
    
