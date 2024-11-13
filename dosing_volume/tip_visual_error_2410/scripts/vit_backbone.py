import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
import utils.arrays as arrays
from utils.utils import seed_everything, device_check, uniform_normalize_label, uniform_unnormalize_label
from utils.dataset import TipsDataset, load_data
from utils.resnet_helper import get_resnet, replace_bn_with_gn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from tqdm import tqdm
import wandb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import time
from datetime import datetime

from vit_pytorch.efficient import ViT
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision import transforms
import random
import numpy as np
import cv2
from linformer import Linformer

'''preparations'''
device_check()
seed = 0
mode = 'train' # 'train' or 'inference'
train_batch_size = 256
test_batch_size = 1

image_size=(120,120)
patch_size=10
dim=128
epochs = 1000
lr = 3e-5
gamma = 0.7

# seed_everything(1)
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'

if __name__ == '__main__':
    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    backbone = 'unet'
    rootpath = f'{TimeCode}_ViT_{mode}'
    save_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    '''load the dataset from npy file'''
    train_data = np.load('dosing_volume/tip_visual_error_2410/data/visual_error_diffusion_training.npy', allow_pickle=True)
    test_data = np.load('dosing_volume/tip_visual_error_2410/data/visual_error_diffusion_testing.npy', allow_pickle=True)

    train_list = train_data.tolist()
    test_list = test_data.tolist()

    train_data = TipsDataset(train_list)
    test_data = TipsDataset(test_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset = test_data, batch_size = test_batch_size, shuffle = False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    
    # --------------- Model Loading -----------------
    efficient_transformer = Linformer(
        dim=dim,
        seq_len=int(np.prod(image_size) / (patch_size ** 2)) + 1,  # mxn patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )
    device = 'cuda'

    model = ViT(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=61,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    pre_loss_tra = 100
    pre_loss_val = 100
    model_id = 0
    epoch_loss_list = []
    epoch_accuracy_list = []

    if mode == 'train':
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            tic = time.time()
            for data, label in tqdm(train_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            print("time cost:",time.time()-tic)
            print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
            epoch_loss_list.append(epoch_loss.cpu().detach().numpy())
            epoch_accuracy_list.append(epoch_accuracy.cpu().detach().numpy())
            model_id += 1

            if model_id % 100 == 0:
                PATH = f"{save_path}/model{model_id}"
                print(f"model been saved in: {PATH}")
                torch.save(model.state_dict(), PATH)
            
            pre_loss_tra = epoch_loss
    
    # elif mode == 'inference':
        
    # else:
    #     raise ValueError(f"Invalid mode: {mode}")