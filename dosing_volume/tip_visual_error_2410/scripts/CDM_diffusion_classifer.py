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
import yaml
from datetime import datetime

'''CDM imports'''
from CDM.cdm import CDM
from CDM.models.diffusion import DDPM_Unet
from CDM.utils import plot_images
from accelerate import Accelerator
from CDM.CustomImageDataset import *

'''preparations'''
device_check()
seed = 0
# seed_everything(1)
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'


if __name__ == '__main__':
    config = yaml.load(open("dosing_volume/tip_visual_error_2410/CDM/config/config.yaml", "r"), Loader=yaml.FullLoader)

    #general config
    mode = config['general']['mode']
    # dataset = config['general']['dataset']
    # dataset_path = config['general']['dataset_path']
    model_arch = config['general']['model_arch']
    num_classes_cond = config['general']['num_classes_cond']  
    if num_classes_cond is not None:
        num_classes_cond += 1 #+1 for classifier free guidance
    beta_start = config['general']['beta_start']
    beta_end = config['general']['beta_end']
    noise_steps = config['general']['noise_steps']

    
    #Training config
    num_iterations = config['training']['num_iterations']
    lr = config['training']['lr']
    batch_size = config['training']['batch_size']
    val_freq = config['training']['val_freq']
    save_ckpt_freq = config['training']['save_ckpt_freq']
    sample_val_images = config['training']['sample_val_images']
    num_workers = config['training']['num_workers']
    ce_factor = config['training']['ce_factor']
    mse_factor = config['training']['mse_factor']
    ema_factor = config['training']['ema_factor']
    horizon = config['training']['horizon']

    #Sampling config  
    ckpt_folder = config['sampling']['ckpt_folder']
    ckpt_file = config['sampling']['ckpt_file']
    num_samples = config['sampling']['num_samples']
    num_sampling_steps = config['sampling']['num_sampling_steps']
    image_shape = config['sampling']['image_shape']
    sampler = config['sampling']['sampler']
    labels = config['sampling']['labels']
    w_cfg = config['sampling']['w_cfg']


    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{model_arch}_h{horizon}_CDM_steps{noise_steps}_{mode}'
    save_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    '''for the first time, save the dataset into npy file'''
    # train_list, test_list = load_data(dataset_dir, seed)
    # trainset_list = np.array(train_list)
    # testset_list = np.array(test_list)
    # np.save('visual_error_diffusion_training.npy', trainset_list)
    # np.save('visual_error_diffusion_testing.npy', testset_list)
    # exit() # exit after saving the dataset

    '''load the dataset from npy file'''
    train_data = np.load('dosing_volume/tip_visual_error_2410/data/visual_error_diffusion_training.npy', allow_pickle=True)
    test_data = np.load('dosing_volume/tip_visual_error_2410/data/visual_error_diffusion_testing.npy', allow_pickle=True)

    train_list = train_data.tolist()
    test_list = test_data.tolist()

    train_data = TipsDataset(train_list)
    test_data = TipsDataset(test_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    obs_dim = 256

    # --------------- Network Architecture -----------------
    in_channels =  3
    out_channels = noise_steps if (mode == 'FM_training' or mode == 'FM_sampling') else noise_steps + 2
    if model_arch == 'ddpm_unet_small':
        model = DDPM_Unet(in_channels=in_channels, out_channels=out_channels, channels=128, image_size=120, resamp_with_conv=True, ch_mult=[1, 2, 2, 2], num_res_blocks=2, attn_resolutions=[16,], dropout=0.1, num_classes=num_classes_cond)
    elif model_arch == 'ddpm_unet_large':
        model = DDPM_Unet(in_channels=in_channels, out_channels=out_channels, channels=128, image_size=120, resamp_with_conv=True, ch_mult=[1, 2, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16,], dropout=0.1, num_classes=num_classes_cond)
    else:
        raise NotImplementedError()

    print(f"===================================================================================")
    optimizer = torch.optim.Adam(model.parameters() ,lr=lr)

    accelerator = Accelerator(log_with="wandb")
    wandb.init(project="tip_visual")
    display_name = f'mode={mode}_model_arch={model_arch}_noise_steps={noise_steps}_num_classes_cond={num_classes_cond}'
    accelerator.init_trackers(
        project_name="CDM",
        config=config
    )

    
    cdm = CDM(accelerator=accelerator, model=model, optimizer=optimizer, ema_factor=ema_factor, batch_size=batch_size, num_workers=num_workers, 
                val_freq=val_freq, display_name=display_name, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, noise_steps=noise_steps,
                ce_factor=ce_factor, mse_factor=mse_factor, num_classes=num_classes_cond, num_sampling_steps=num_sampling_steps, beta_start=beta_start, beta_end=beta_end)
  
    if mode == 'training':
        cdm.simple_train(train_loader=train_loader, save_path = save_path, save_ckpt_freq=save_ckpt_freq, 
                         num_iterations=num_iterations, sampler=sampler)
        accelerator.end_training()

    elif mode == 'sampling':
        if num_classes_cond is not None and labels is not None:
            labels = torch.tensor([labels] * num_samples).to(accelerator.device)
        else:
            labels = None
        sampled_imgs = cdm.sample(num_samples=num_samples, image_shape=image_shape, labels=labels, 
                                  w_cfg=w_cfg, sampler=sampler, validation=False)
        plot_images(sampled_imgs, folder = save_path, figsize=(40,4))
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

    