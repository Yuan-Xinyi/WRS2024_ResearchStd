import torch
from vit_pytorch.efficient import ViT
from linformer import Linformer
import glob
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import fisheye_camera as fcam
import config_file as conf

batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42
dim = 128

img_size = (45,80)
patch_size = 5
num_classes = 61

efficient_transformer = Linformer(
    dim=dim,
    seq_len=int(img_size[0] / patch_size * img_size[1] / patch_size)+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
device = 'cuda'


new_model = ViT(
    dim=dim,
    image_size=img_size,
    patch_size=patch_size,
    num_classes=num_classes,
    transformer=efficient_transformer,
    channels=3,
).to(device)

model_path = "../models_learning/vit_new2"

pre_weights = torch.load(model_path, map_location=torch.device(device))
pre_weights.pop('to_patch_embedding.1.weight')
pre_weights.pop('to_patch_embedding.1.bias')
new_model.load_state_dict(pre_weights,strict=False,)
new_model.eval()
pic_transformer = transforms.Compose([transforms.ToTensor()])


get_frame = fcam.FisheyeCam(conf.calib_path).get_frame_cut_combine_row
pic = get_frame()
print(pic.shape)
_pic = Image.fromarray(pic)
pic_tensor = pic_transformer(_pic)
pic_tensor = pic_tensor.unsqueeze(0)
pic_tensor = pic_tensor.to(device)
with torch.no_grad():
    [val_output] = new_model(pic_tensor).detach().cpu().numpy()
print(val_output.argmax())



