# utils/data_loader.py

import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
from collections import namedtuple
import re

Batch = namedtuple('Batch', 'observations actions')
class TipsDataset(Dataset):
    def __init__(self, file_list):
        """Initialize the dataset and define data preprocessing transformations."""
        self.file_list = file_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Return data and label based on index."""
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_trans = self.transform(img)  # (3,120,120)
        label = int(img_path.split("/")[-1].split(".")[0])  # Extract the filename as the label
        
        batch = Batch(img_trans, label)
        return batch
    

def load_data(dataset_dir, seed, test_size=0.2):
    """
    Load the dataset, split it into training and validation sets, and return the data loaders.
    """
    # dataset_list = glob.glob(os.path.join(dataset_dir, '*/*.jpg'))
    dataset_list = glob.glob(os.path.join(dataset_dir, '*/*.png'))

    labels = [path.split("/")[-1].split(".")[0] for path in dataset_list]

    train_list, test_list = train_test_split(
                dataset_list, test_size=test_size, stratify=labels, random_state=seed)

    return train_list, test_list


class MedicalDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Return data and label based on index."""
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_trans = self.transform(img)  # (3,120,120)
        label = int(re.search(r'image(\d+)\.jpg', img_path).group(1))  # Extract the filename as the label
        
        batch = Batch(img_trans, label)
        return batch