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
        img_trans = self.transform(img)
        label = int(img_path.split("\\")[-1].split(".")[0])  # Extract the filename as the label
        return img_trans, label


def load_data(dataset_dir_old, dataset_dir_RIKEN, test_size=0.2, random_seed=42, batch_size=512):
    """
    Load the dataset, split it into training and validation sets, and return the data loaders.
    """
    # Get image paths
    dataset_list_0 = glob.glob(os.path.join(dataset_dir_RIKEN, '*/*.jpg'))
    dataset_list_1 = glob.glob(os.path.join(dataset_dir_old, '*/*.jpg'))

    labels_0 = [path.split("\\")[-1].split(".")[0] for path in dataset_list_0]
    labels_1 = [path.split("\\")[-1].split(".")[0] for path in dataset_list_1]

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    # Split the dataset into training and validation sets
    train_list_0, valid_list_0 = train_test_split(
        dataset_list_0, test_size=test_size, stratify=labels_0, random_state=random_seed
    )
    train_list_1, valid_list_1 = train_test_split(
        dataset_list_1, test_size=test_size, stratify=labels_1, random_state=random_seed
    )

    train_list = train_list_0 + train_list_1
    valid_list = valid_list_0 + valid_list_1

    # Create dataset objects
    train_data = TipsDataset(train_list)
    valid_data = TipsDataset(valid_list)

    # Create data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader
