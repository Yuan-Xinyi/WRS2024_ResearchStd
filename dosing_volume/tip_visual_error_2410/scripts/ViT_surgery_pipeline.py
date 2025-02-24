import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
import utils.arrays as arrays
from utils.utils import seed_everything, device_check, uniform_normalize_label, uniform_unnormalize_label
from utils.dataset import MedicalDataset
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
import os

'''preparations'''
device_check()
seed = 0
dataset_name = 'surgery_bicameral'
mode = 'test' # 'train' or 'test'
train_batch_size = 256
test_batch_size = 1
save_freq = 50

dim=128
epochs = 1000
lr = 3e-5
gamma = 0.7

'''img parameters'''
image_size=(160,80)
patch_size=8
num_classes = 91

# seed_everything(1)
'''dataset dir list'''
parent_dir = 'dosing_volume/tip_visual_error_2410/data/surgery_datasets'
img2d_dataset_dir = os.path.join(parent_dir, '10degree3d')  # ['10degree2d', '10degree3d', '30degree3d']
print(f"Loading dataset from {img2d_dataset_dir}")


if __name__ == '__main__':
    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_ViT'
    save_path = f'dosing_volume/tip_visual_error_2410/results/surgery/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    file_list = glob.glob(os.path.join(img2d_dataset_dir, '*/*.jpg'))
    train_list, test_list = train_test_split(file_list, test_size=0.3, random_state=seed)

    train_data = MedicalDataset(train_list)
    test_data = MedicalDataset(test_list)

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
        num_classes=num_classes,
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
    model_id = 0
    epoch_loss_list = []
    epoch_accuracy_list = []

    if mode == 'train':
        wandb.init(project="surgery_vit", name=f"ViT_{TimeCode}")
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            tic = time.time()
            for data, label in tqdm(train_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)
                wandb.log({"loss": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)
                wandb.log({"accuracy": acc.item()})
                wandb.log({"epoch_loss": epoch_loss.item()})

            print("time cost:",time.time()-tic)
            print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
            epoch_loss_list.append(epoch_loss.cpu().detach().numpy())
            epoch_accuracy_list.append(epoch_accuracy.cpu().detach().numpy())
            model_id += 1

            if model_id % save_freq == 0:
                PATH = f"{save_path}/model{model_id}"
                print(f"model been saved in: {PATH}")
                torch.save(model.state_dict(), PATH)
            
            pre_loss_tra = epoch_loss
        wandb.finish()
    
    elif mode == 'test':
        inference_losses = []
        # PATH = f'dosing_volume/tip_visual_error_2410/results/surgery/0219_2024_ViT/model100'
        PATH = f'dosing_volume/tip_visual_error_2410/results/surgery/0219_2100_ViT/model100'
        model.load_state_dict(torch.load(PATH))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():  # Disable gradient computation for testing
            for data, label in (test_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                pred_label = torch.argmax(output, dim=1)
                loss = F.l1_loss(pred_label.float(), label.float())
                if loss.item() > 0:
                    print('gth label: ',label.item(),'pred_label:', pred_label.item())
                inference_losses.append(loss.item())

        loss_differences = np.array(inference_losses)
        avg_loss = np.mean(loss_differences)
        median_loss = np.median(loss_differences)
        std_loss = np.std(loss_differences)
        zero_ratio = np.sum(loss_differences == 0) / len(loss_differences)
        success_ratio = (np.sum(loss_differences == 0) + np.sum(loss_differences == 1)) / len(loss_differences)


        print(f"Test Set Median Loss: {median_loss:.4f}")
        print(f"Test Set Average Loss: {avg_loss:.4f}")
        print(f"Standard Deviation of Loss: {std_loss:.4f}")
        print(f"Proportion of Zero Losses: {zero_ratio * 100:.2f}%")
        print(f"Proportion of Successes (Zero and One Losses): {success_ratio * 100:.2f}%")

        # plt.figure(figsize=(10, 6))
        # plt.hist(loss_differences, bins=20, density=True, alpha=0.6, color='g', label="Histogram")

        # sns.kdeplot(loss_differences, color='b', label="KDE Curve")

        # plt.title("Probability Distribution of MSE Loss Differences")
        # plt.xlabel("MSE Loss Difference")
        # plt.ylabel("Density")
        # plt.legend()
        # plt.grid()
        # plt.savefig(save_path + f"loss_distribution.png")
        # # exit()

    else:
        raise ValueError(f"Invalid mode: {mode}")