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
from cleandiffuser.nn_diffusion import ChiTransformer, JannerUNet1d, ChiUNet1d
from cleandiffuser.utils import report_parameters
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.dataset.dataset_utils import loop_dataloader
# from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
device_check()
seed = 0
# seed_everything(1)
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'

# diffuser parameters
backbone = 'transformer' # ['transformer', 'unet']
mode = 'loop_inference'  # ['train', 'inference', 'loop_inference']
train_batch_size = 16
test_batch_size = 1
solver = 'ddpm'
diffusion_steps = 20
predict_noise = False # [True, False]
obs_steps = 1
action_steps = 1
num_classes = 61
action_scale = 1.0
action_loss_weight = 1.0

# Training
device = 'cuda'
diffusion_gradient_steps = 200000
log_interval = 100
save_interval = 1000
lr = 0.00001
num_epochs = 1000

action_dim = 1
horizon = 4
obs_steps = 1
shape_meta = {
    'obs': {
        'image': {
            'shape': (3, 120, 120),   # (channels, height, width) for image inputs
            'type': 'rgb'           # 'rgb'
        }
    }
}
rgb_model = 'resnet50' # ['resnet18', 'resnet50']
use_group_norm = True
ema_rate = 0.9999

# inference parameters
sampling_steps = 10
w_cg = 0.0001
temperature = 0.5
use_ema = False


if __name__ == '__main__':
    # --------------- Data Loading -----------------
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{backbone}_{horizon}_{rgb_model}_{mode}'
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
    train_loader = DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset = test_data, batch_size = test_batch_size, shuffle = False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    obs_dim = 256

    # --------------- Network Architecture -----------------
    '''x max and x min'''
    if backbone == 'transformer':
        x_max = torch.ones((1, horizon, action_dim), device=device) * +action_scale  # （1，1，1）
        x_min = torch.ones((1, horizon, action_dim), device=device) * -action_scale  # （1，1，1）
    elif backbone == 'unet':
        x_max = torch.ones((1, horizon, action_dim), device=device) * +action_scale
        x_min = torch.ones((1, horizon, action_dim), device=device) * -action_scale
    else:
        raise ValueError(f"Invalid backbone: {backbone}")

    # dropout=0.0 to use no CFG but serve as FiLM encoder
    nn_condition = MultiImageObsCondition(shape_meta=shape_meta, emb_dim=256, rgb_model_name=rgb_model, 
                                          use_group_norm=use_group_norm).to(device)

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
            if backbone == 'transformer': 
                img, action = batch[0].to(device).float(), batch[1].to(device).float()
                
                # process the image into observation
                condition = {'image': img}
                
                # Normalize the label
                naction = uniform_normalize_label(action, num_classes=num_classes, scale=action_scale)
                naction = naction.unsqueeze(1).unsqueeze(2) # (batch, 1)
                naction = naction.expand(-1, horizon, -1)  # (batch, expand dim, 1)

                diffusion_loss = agent.update(naction, condition)['loss']
            
            elif backbone == 'unet':
                img, action = batch[0].to(device).float(), batch[1].to(device).float()
                
                # process the image into observation
                condition = {'image': img}
                
                # Normalize the label
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
        # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1108_1707_chiunet_train/diffusion_ckpt_latest.pt' # current best, though class = 60 wrongly
        
        # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1638_transformer_4_resnet18_train/diffusion_ckpt_latest.pt'
        # load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1639_transformer_1_resnet18_train/diffusion_ckpt_latest.pt'
        load_path = 'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1643_transformer_4_resnet50_train/diffusion_ckpt_latest.pt'

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
                if backbone == 'transformer':
                    img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()  # [image, label]
                    condition = {'image': img}
                    
                    naction, _ = agent.sample(prior=prior, n_samples=1, sample_steps=sampling_steps,
                        solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)   
                    mean_action = naction.mean()
                    pred_label = uniform_unnormalize_label(mean_action, num_classes=num_classes, scale=action_scale) 
                    
                elif backbone == 'unet':
                    img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()
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
            # load_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1638_transformer_4_resnet18_train/diffusion_ckpt_{checkpoint}.pt'
            # load_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1639_transformer_1_resnet18_train/diffusion_ckpt_{checkpoint}.pt'
            load_path = f'dosing_volume/tip_visual_error_2410/results/diffuser/1113_1643_transformer_4_resnet50_train/diffusion_ckpt_{checkpoint}.pt'
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
                    if backbone == 'transformer':
                        img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()  # [image, label]
                        condition = {'image': img}
                        
                        naction, _ = agent.sample(prior=prior, n_samples=1, sample_steps=sampling_steps,
                            solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)   
                        mean_action = naction.mean()
                        pred_label = uniform_unnormalize_label(mean_action, num_classes=num_classes, scale=action_scale) 
                        
                    elif backbone == 'unet':
                        img, gth_label = batch[0].to(device).float(), batch[1].to(device).float()
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
    

    