from vit_pytorch.efficient import ViT
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
import file_sys as fs
import time
import albumentations as A

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# Training settings
batch_size = 64
epochs = 3000
lr = 1e-4
gamma = 0.7
seed = 42

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model_name = 'vit_120'
date = time.strftime('%Y%m%d-%H%M%S')


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
        self.cached_data = {}

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = str(self.file_list[idx])
        if img_path in self.cached_data:
            img = self.cached_data[img_path]
        else:
            img = cv2.imread(img_path)
            self.cached_data[img_path] = img
        img_trans = self.transform(img)
        label = int(img_path.split("\\")[-1].split(".")[0])
        return img_trans, label


class TipsTrainDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.cached_data = {}
        aug = []
        aug.append(A.Blur(blur_limit=3, p=.5))
        aug.append(A.GaussNoise(p=.5))
        aug.append(A.RandomBrightnessContrast(p=.5, brightness_limit=.1, contrast_limit=.1))
        self.effect = A.Compose(aug, )

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = str(self.file_list[idx])
        if img_path in self.cached_data:
            img = self.cached_data[img_path]
        else:
            img = cv2.imread(img_path)
            self.cached_data[img_path] = img
        img = self.effect(image=img)['image']
        # print("RUN DES")
        # cv2.imshow("test", letterbox(img, new_shape=[360, 640], auto=False)[0])
        # cv2.waitKey(0)
        img_trans = self.transform(img)
        label = int(img_path.split("\\")[-1].split(".")[0])
        return img_trans, label


# train_dir = "./capture/spiral_t_hex/"
train_dir = fs.Path('.').joinpath("data", "capture", "feature_big")

train_list = list(train_dir.glob('**/*.jpg'))

print(len(train_list))
print(train_list[0])
print(train_list[1000])
labels = [path.name.split(".")[0] for path in train_list]
# labels = [path.split("\\")[-2] for path in train_list]
print(labels[0])
print(labels[1000])

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

train_data = TipsTrainDataset(train_list)
valid_data = TipsDataset(valid_list)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

efficient_transformer = Linformer(
    dim=256,
    seq_len=15 * 15 + 1,  # mxn patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
device = 'cuda'

model = ViT(
    dim=256,
    image_size=(120, 120),
    patch_size=8,
    num_classes=61,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=150, gamma=.1)

best_val_r = -999
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader, total=len(train_loader)):
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
            if epoch_val_accuracy > best_val_r:
                best_val_r = epoch_val_accuracy
                torch.save(model.state_dict(), f"trained_model/{model_name}_{date}_best")
    torch.save(model.state_dict(), f"trained_model/{model_name}_{date}_last")
    scheduler.step()
    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - lr: {scheduler.get_lr()[0]}\n"
    )
