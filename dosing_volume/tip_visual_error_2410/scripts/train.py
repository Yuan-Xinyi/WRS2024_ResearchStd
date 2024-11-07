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

'''cleandiffuser imports'''
from datetime import datetime
from cleandiffuser.diffusion.diffusionsde import BaseDiffusionSDE, DiscreteDiffusionSDE, BaseDiffusionSDE
from cleandiffuser.nn_condition import MultiImageObsCondition
from cleandiffuser.nn_diffusion import ChiTransformer
from cleandiffuser.utils import report_parameters
from cleandiffuser.diffusion.ddpm import DDPM
# from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
device_check()
seed = 0
seed_everything(1)
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'

# diffuser parameters
solver = 'ddpm'
diffusion_steps = 20
predict_noise = True # [True, False]
obs_steps = 1
action_steps = 1
num_classes = 60
action_scale = 10.0

# Training
mode = 'train'  # ['train', 'inference', 'case_inference']
device = 'cuda'
diffusion_gradient_steps = 10000
batch_size = 8
log_interval = 100
save_interval = 1000
lr = 0.0001
num_epochs = 1

action_dim = 1
horizon = 1
obs_steps = 1
shape_meta = {
    'obs': {
        'image': {
            'shape': (3, 120, 120),   # (channels, height, width) for image inputs
            'type': 'rgb'           # 'rgb'
        }
    }
}
rgb_model = 'resnet50'
use_group_norm = True

# inference parameters
sampling_steps = 10
w_cg = 0.0001
temperature = 0.5
use_ema = True




if __name__ == '__main__':
    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{mode}'
    save_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    train_list, test_list = load_data(dataset_dir, seed)
    train_data = TipsDataset(train_list)
    test_data = TipsDataset(test_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle = False)

    # # construct ResNet18 encoder
    # # important: if you have multiple camera views, use seperate encoder weights for each view.
    # vision_encoder = get_resnet('resnet18')

    # # IMPORTANT!
    # # replace all BatchNorm with GroupNorm to work with EMA
    # # performance will tank if you forget to do this!
    # vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    obs_dim = vision_feature_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = ChiTransformer(
        action_dim, obs_dim, horizon, obs_steps, d_model=256, nhead=4, num_layers=4,
        timestep_emb_type="positional").to(device)
    
    # dropout=0.0 to use no CFG but serve as FiLM encoder
    nn_condition = MultiImageObsCondition(
        shape_meta=shape_meta, emb_dim=512, rgb_model_name=rgb_model, use_group_norm=use_group_norm).to(device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    # --------------- Diffusion Model --------------------
    '''x max and x min'''
    x_max = torch.ones((1, horizon, action_dim), device=device) * +action_scale  # （1，1，1）
    x_min = torch.zeros((1, horizon, action_dim), device=device) * -action_scale  # （1，1，1）

    agent = DDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=device,
        diffusion_steps=diffusion_steps, x_max=x_max, x_min=x_min,
        predict_noise=predict_noise, optim_params={"lr": lr})
    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, diffusion_gradient_steps)

    if mode == 'train':
        wandb.init(project="tip_visual")
        # ---------------------- Training ----------------------
        # agent.train()
        n_gradient_step = 0
        log = {'avg_loss_diffusion': 0.}
        start_time = time.time()

        for epoch in tqdm(range(num_epochs)):
            for batch in train_loader:
                nobs, action = batch[0].to(device).float(), batch[1].to(device).float() # [image, label] image size: (batch_size, 3, 120, 120), label size: (batch_size)
                
                '''normalize the label'''
                naction = uniform_normalize_label(action, num_classes=num_classes, scale=action_scale)  # (batch_size, 1)
                naction = naction.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, action_dim) (batch,1,1)
                '''don't need to normalize the label'''
                # naction = action.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, action_dim) (batch,1,1)
                
                condition = {'image': nobs}

                # ----------- Gradient Step ------------
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
        load_path = '/home/lqin/wrs_2024/dosing_volume/tip_visual_error_2410/results/diffuser/1106_1750_train/diffusion_ckpt_latest.pt'
        agent.load(load_path)
        agent.eval()
        inference_losses = []
        prior = torch.zeros((1, horizon, action_dim), device=device)

        with torch.no_grad():
            for batch in test_loader:
                nobs, gth_label = batch[0].to(device).float(), batch[1].to(device).float()  # [image, label]
                condition = {'image': nobs}
                
                naction, _ = agent.sample(prior=prior, n_samples=1, 
                                          sample_steps=sampling_steps, solver=solver, 
                                          condition_cfg=condition, w_cfg=1.0, use_ema=use_ema)   # (env_num, 64, 12)
                print('current action is:', naction)
                pred_label = uniform_unnormalize_label(naction, num_classes=num_classes, scale=action_scale) 
                loss = F.mse_loss(torch.tensor(pred_label).to(device='cuda'), gth_label)

                # print('pred_label:', pred_label, 'gth_label:', gth_label, 'loss:', loss.item())
                inference_losses.append(loss.item())
        
        loss_differences = np.array(inference_losses)
        avg_loss = np.mean(loss_differences)
        median_loss = np.median(loss_differences)
        print(f"Test Set Median Loss: {median_loss:.4f}", f"Test Set Average Loss: {avg_loss:.4f}")

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
                   
    elif mode == 'case_inference':
        # ---------------------- Case Test ----------------------
        load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1105_1556_transformer/diffusion_ckpt_5000.pt'
        agent.load(load_path)
        agent.eval()
        inference_losses = []
        prior = torch.zeros((1, horizon, action_dim), device=device)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
                nobs, gth_label = batch[0].to(device).float(), batch[1].to(device).float()
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

    