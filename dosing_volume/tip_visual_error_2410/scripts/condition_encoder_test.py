import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
import utils.arrays as arrays
from utils.utils import seed_everything, device_check, uniform_normalize_label, uniform_unnormalize_label
from utils.utils import reshape_condition
from utils.dataset import TipsDataset, load_data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from tqdm import tqdm
import wandb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from linformer import Linformer

'''cleandiffuser imports'''
from datetime import datetime
from cleandiffuser.nn_condition import MultiImageObsCondition, EarlyConvViTMultiViewImageCondition
from cleandiffuser.nn_condition import FlexibleEarlyConvViTMultiViewImageCondition
from cleandiffuser.nn_condition import ViTImageCondition
from cleandiffuser.nn_diffusion import ChiTransformer, ChiUNet1d
from cleandiffuser.utils import report_parameters
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
device_check()
seed = 0
# seed_everything(1)
'''dataset dir list'''
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
# dataset_dir = '/home/lqin/wrs_2024/dosing_volume/tip_visual_error_2410/data/spiral_t_hex/'
# dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'

# diffuser parameters
dataset_name = 'spiral_visual_error_diffusion'  # ['spiral_visual_error_diffusion' for 2 cameras, 'visual_error_diffusion' for single camera]
backbone = 'unet' # ['transformer', 'unet', 'vit']
mode = 'train'  # ['train', 'inference', 'loop_inference']
condition_encoder = 'vit' # ['multi_image_obs', 'cnn_vit', 'vit']
train_batch_size = 64
test_batch_size = 1
solver = 'ddpm'
diffusion_steps = 20
obs_steps = 1
action_steps = 1
num_classes = 61
action_scale = 60.0
action_loss_weight = 1.0

# Training
device = 'cuda'
diffusion_gradient_steps = 300000
log_interval = 100
save_interval = 10000
lr = 0.00001
num_epochs = 1000

action_dim = 1
horizon = 4
obs_steps = 1
if dataset_name == 'spiral_visual_error_diffusion':
    shape_meta = {
        'obs': {
            'image': {
                'shape': (3, 45, 80),   # (channels, height, width) for image inputs
                'type': 'rgb'           # 'rgb'
            }
        }
    }
elif dataset_name == 'visual_error_diffusion':
    shape_meta = {
        'obs': {
            'image': {
                'shape': (3, 120, 120),   # (channels, height, width) for image inputs
                'type': 'rgb'           # 'rgb'
            }
        }
    }
else:
    raise ValueError(f"Invalid dataset name: {dataset_name}")

use_group_norm = True
ema_rate = 0.9999

# inference parameters
sampling_steps = 10
w_cg = 0.0001
temperature = 0.5
use_ema = False
rgb_model = 'resnet18' # ['resnet18', 'resnet50']


if __name__ == '__main__':
    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{backbone}_b{train_batch_size}_h{horizon}_{condition_encoder}_{mode}'
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
    train_data = np.load(f'dosing_volume/tip_visual_error_2410/data/{dataset_name}_training.npy', allow_pickle=True)
    test_data = np.load(f'dosing_volume/tip_visual_error_2410/data/{dataset_name}_testing.npy', allow_pickle=True)

    train_list = train_data.tolist()
    test_list = test_data.tolist()

    train_data = TipsDataset(train_list)
    test_data = TipsDataset(test_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset = test_data, batch_size = test_batch_size, shuffle = False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    obs_dim = 256

    # --------------- Network Architecture -----------------
    '''x max and x min'''
    if action_scale == 60.0:
        x_max = torch.ones((1, horizon, action_dim), device=device) * +action_scale
        x_min = torch.zeros((1, horizon, action_dim), device=device)
    else:
        x_max = torch.ones((1, horizon, action_dim), device=device) * +action_scale
        x_min = torch.ones((1, horizon, action_dim), device=device) * -action_scale


    # dropout=0.0 to use no CFG but serve as FiLM encoder
    if condition_encoder == 'multi_image_obs':
        nn_condition = MultiImageObsCondition(shape_meta=shape_meta, emb_dim=256, rgb_model_name=rgb_model, 
                                          use_group_norm=use_group_norm).to(device)
    elif condition_encoder == 'cnn_vit':
        nn_condition = EarlyConvViTMultiViewImageCondition(image_sz=(120,), in_channels=(3,), To=obs_steps, 
                                                           d_model=256, nhead=8, num_layers=12, patch_size=(16,)).to(device)
    elif condition_encoder == 'vit':
        dim, image_size, patch_size, channels = 256, shape_meta['obs']['image']['shape'][1:], 5, 3
        efficient_transformer = Linformer(
            dim=dim, seq_len=int(np.prod(image_size) / (patch_size ** 2)) + 1,  # mxn patches + 1 cls-token
            depth=12, heads=8, k=64)
        nn_condition = ViTImageCondition(image_size=image_size, patch_size=patch_size, dim=dim, 
                                         transformer=efficient_transformer, channels=channels).to(device)
    else:
        raise ValueError(f"Invalid condition encoder: {condition_encoder}")

    if backbone == 'transformer':
        nn_diffusion = ChiTransformer(
            action_dim, obs_dim, horizon, obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(device)
        
        agent = DDPM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=device,
                     diffusion_steps=sampling_steps, x_max=x_max, x_min=x_min, optim_params={"lr": lr})
    
    elif backbone == 'unet':
        nn_diffusion = ChiUNet1d(action_dim, 256, obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
                                 obs_as_global_cond=True, timestep_emb_type="positional").to(device)
        
        agent = DDPM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=device,
                     diffusion_steps=sampling_steps, x_max=x_max, x_min=x_min, optim_params={"lr": lr})
        
    else:
        raise ValueError(f"Invalid backbone: {backbone}")


    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")

    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=diffusion_gradient_steps)

    if mode == 'train':
        wandb.init(project="tip_visual")
        # ---------------------- Training ----------------------
        agent.train()
        n_gradient_step = 0
        log = {'avg_loss_diffusion': 0.}
        start_time = time.time()


        for batch in loop_dataloader(train_loader):
            img, action = batch[0].to(device).float(), batch[1].to(device).float()
            
            if condition_encoder == 'cnn_vit':
                img = reshape_condition(img, t = obs_steps)
                # print('Remark: we adjust the shape of conditional image for CNN-ViT')
            
            # process the image into observation
            if condition_encoder == 'vit':
                condition = img
            else:
                condition = {'image': img}
            
            # Normalize the label
            if action_scale == 60.0:
                action = action.unsqueeze(1).unsqueeze(2) # (batch, 1)
                naction = action.expand(-1, horizon, -1)  # (batch, expand dim, 1)
            else:
                naction = uniform_normalize_label(action, num_classes=num_classes, scale=action_scale)
                naction = naction.unsqueeze(1).unsqueeze(2) # (batch, 1)
                naction = naction.expand(-1, horizon, -1)  # (batch, expand dim, 1)

            diffusion_loss = agent.update(naction, condition)['loss']

            log['avg_loss_diffusion'] += diffusion_loss  # BaseDiffusionSDE.update
            # print(f'[t={n_gradient_step + 1}] diffusion loss = {current_loss}')
            diffusion_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % log_interval == 0:
                log['gradient_steps'] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= log_interval
                wandb.log(
                    {'step': log['gradient_steps'],
                    'avg_training_loss': log['avg_loss_diffusion'],
                    'total_time': time.time() - start_time}, commit = True)
                print(log)
                log = {"avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= diffusion_gradient_steps:
                break
        wandb.finish()
    
    elif mode == 'inference':
        # ---------------------- Testing ----------------------
        # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1115_1608_unet_h4_cnn_vit_train/diffusion_ckpt_latest.pt'
        # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1115_1739_unet_h4_vit_train/diffusion_ckpt_latest.pt'
        load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1127_1036_unet_b64_h4_vit_train/diffusion_ckpt_latest.pt'
        agent.load(load_path)
        agent.eval()
        agent.model.eval()
        agent.model_ema.eval()

        inference_losses = []
        if backbone == 'transformer':
            prior = torch.zeros((1, horizon, action_dim), device=device)
        elif backbone == 'unet':
            prior = torch.zeros((test_batch_size, horizon, action_dim), device=device)
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        with torch.no_grad():
            for batch in tqdm(test_loader):
                img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()
                if condition_encoder == 'cnn_vit':
                    img = reshape_condition(img, t = obs_steps)
                    # print('Remark: we adjust the shape of conditional image for CNN-ViT')
                
                # process the image into observation
                if condition_encoder == 'vit':
                    condition = img
                else:
                    condition = {'image': img}

                naction, _ = agent.sample(prior=prior, n_samples=1, sample_steps=sampling_steps,
                    solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)                    
                
                mean_action = naction.mean()
                if action_scale == 60.0:
                    pred_label = torch.round(mean_action)
                else:
                    pred_label = uniform_unnormalize_label(mean_action, num_classes=num_classes, scale=action_scale) 
                if gth_label.item() != pred_label:
                    print('gth_label:', gth_label.item(),'raw action', mean_action.item(), 'pred_label: ', pred_label.item())
                
                loss = F.l1_loss(torch.tensor([pred_label], device=device), gth_label)
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

        plt.figure(figsize=(10, 6))
        plt.hist(loss_differences, bins=20, density=True, alpha=0.6, color='g', label="Histogram")

        sns.kdeplot(loss_differences, color='b', label="KDE Curve")

        plt.title("Probability Distribution of MSE Loss Differences")
        plt.xlabel("MSE Loss Difference")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.savefig(save_path + f"loss_distribution.png")
        plt.show()
                   
    elif mode == 'loop_inference':
        checkpoints = [str(i) for i in range(20000, 220000, 20000)]
        checkpoints.append('latest')

        for checkpoint in checkpoints:
            # ---------------------- Testing ----------------------
            load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1115_1608_unet_h4_cnn_vit_train/diffusion_ckpt_latest.pt'
            # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1115_1739_unet_h4_vit_train/diffusion_ckpt_latest.pt'
            print(f"Loading checkpoint: {load_path}")
            
            agent.load(load_path)
            agent.eval()
            agent.model.eval()
            agent.model_ema.eval()

            inference_losses = []
            if backbone == 'transformer':
                prior = torch.zeros((1, horizon, action_dim), device=device)
            elif backbone == 'unet':
                prior = torch.zeros((test_batch_size, horizon, action_dim), device=device)
            else:
                raise ValueError(f"Invalid backbone: {backbone}")

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()
                    if condition_encoder == 'cnn_vit':
                        img = reshape_condition(img, t = obs_steps)
                        # print('Remark: we adjust the shape of conditional image for CNN-ViT')
                    
                    # process the image into observation
                    if condition_encoder == 'vit':
                        condition = img
                    else:
                        condition = {'image': img}

                    naction, _ = agent.sample(prior=prior, n_samples=1, sample_steps=sampling_steps,
                        solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)                    
                    
                    mean_action = naction.mean()
                    pred_label = uniform_unnormalize_label(mean_action, num_classes=num_classes, scale=action_scale) 
                    # print('gth_label:', gth_label.item(),'pred_label: ',pred_label)
                    
                    loss = F.l1_loss(torch.tensor([pred_label], device=device), gth_label)
                    inference_losses.append(loss.item())
            
            loss_differences = np.array(inference_losses)
            avg_loss = np.mean(loss_differences)
            median_loss = np.median(loss_differences)
            std_loss = np.std(loss_differences)
            zero_ratio = np.sum(loss_differences == 0) / len(loss_differences)
            success_ratio = (np.sum(loss_differences == 0) + np.sum(loss_differences == 1)) / len(loss_differences)

            print(f"Model Checkpoint: {checkpoint}")
            print(f"Test Set Median Loss: {median_loss:.4f}")
            print(f"Test Set Average Loss: {avg_loss:.4f}")
            print(f"Standard Deviation of Loss: {std_loss:.4f}")
            print(f"Proportion of Zero Losses: {zero_ratio * 100:.2f}%")
            print(f"Proportion of Successes (Zero and One Losses): {success_ratio * 100:.2f}%")

            
            plt.figure(figsize=(10, 6))
            plt.hist(loss_differences, bins=20, density=True, alpha=0.6, color='g', label="Histogram")

            sns.kdeplot(loss_differences, color='b', label="KDE Curve")

            plt.title("Probability Distribution of MSE Loss Differences")
            plt.xlabel("MSE Loss Difference")
            plt.ylabel("Density")
            plt.legend()
            plt.grid()
            plt.savefig(save_path + f"{checkpoint}_loss_distribution.png")
            # plt.show()
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

    