import time

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

# Training settings
batch_size = 512
epochs = 1000
lr = 3e-5
gamma = 0.7
seed = 42


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)
print(seed)

class TipsDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_trans = self.transform(img)
        label = int(img_path.split("\\")[-1].split(".")[0])
        # label = int(img_path.split("\\")[-2])
        # print(label)
        return img_trans, label

dataset_dir_old = "../training_dataset/capture/mbp_D403/"
dataset_dir_RIKEN = "../training_dataset/capture/RIKEN_yokohama_tip_D403/img_2/"

dataset_list_0 = glob.glob(os.path.join(dataset_dir_RIKEN, '*/*.jpg'))
dataset_list_1 = glob.glob(os.path.join(dataset_dir_old, '*/*.jpg'))


print(len(dataset_list_0)+len(dataset_list_1))
# print(train_list[0])
# print(train_list[1000])
labels_0 = [path.split("\\")[-1].split(".")[0] for path in dataset_list_0]
labels_1 = [path.split("\\")[-1].split(".")[0] for path in dataset_list_1]
# labels = [path.split("\\")[-2] for path in train_list]
# print(labels[0])
# print(labels[1000])

image_size=(120,120)
patch_size=10
dim=128


train_list_0, valid_list_0 = train_test_split(dataset_list_0,
                                          test_size=0.2,
                                          stratify=labels_0,
                                          random_state=seed)

train_list_1, valid_list_1 = train_test_split(dataset_list_1,
                                          test_size=0.2,
                                          stratify=labels_1,
                                          random_state=seed)

train_list = train_list_0 + train_list_1
valid_list = valid_list_0 +valid_list_1



train_data = TipsDataset(train_list)
valid_data = TipsDataset(valid_list)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

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
epoch_val_loss_list = []
epoch_val_accuracy_list = []

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

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print("time cost:",time.time()-tic)
    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    epoch_loss_list.append(epoch_loss.cpu().detach().numpy())
    epoch_accuracy_list.append(epoch_accuracy.cpu().detach().numpy())
    epoch_val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
    epoch_val_accuracy_list.append(epoch_val_accuracy.cpu().detach().numpy())

    PATH = f"./data/vit_training/d405/vit_d403_{model_id}"
    torch.save(model.state_dict(), PATH)

    if epoch_loss > pre_loss_tra:
        print(f"saved: vit_d405_1000_{model_id}")
        model_id += 1
    pre_loss_tra = epoch_loss
    pre_loss_val = epoch_val_loss
np.savetxt("./data/vit_training/d405/epoch_loss.txt",epoch_loss_list)
np.savetxt("./data/vit_training/d405/epoch_accuracy.txt",epoch_accuracy_list)
np.savetxt("./data/vit_training/d405/epoch_val_loss.txt",epoch_val_loss_list)
np.savetxt("./data/vit_training/d405/epoch_val_accuracy.txt",epoch_val_accuracy_list)

# PATH = f"./models/vit_spiral_1000_last"
# torch.save(model.state_dict(), PATH)



