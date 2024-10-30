import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
import utils.arrays as arrays
from utils.utils import seed_everything, device_check
from utils.dataset import TipsDataset, load_data
from utils.resnet_helper import get_resnet, replace_bn_with_gn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

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
seed_everything(0)
# dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"
dataset_dir = 'dosing_volume/tip_visual_error_2410/data/mbp_D405/'


# image parameters
image_size=(120,120)
# dim=128

# diffuser parameters
solver = 'ddpm'
model_dim = 32
diffusion_steps = 20
predict_noise = False
action_loss_weight = 10.0
ema_rate = 0.9999
noise_level = 0.0
normalizer = 'GaussianNormalizer'  # [CustomizedNormalizer, GaussianNormalizer, SafeLimitsNormalizer]
obs_steps = 1
action_steps = 1

# Training
device = 'cuda'
n_train_steps = 5000
diffusion_gradient_steps = 20000
batch_size = 64
log_interval = 100
save_interval = 1000
sample_steps = 20
lr = 0.0001
epochs = 100



if __name__ == '__main__':
    train_list, valid_list = load_data(dataset_dir, seed)
    train_data = TipsDataset(train_list)
    valid_data = TipsDataset(valid_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle = True)

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2
    for batch in tqdm(train_loader):
        image, label = batch[0], batch[1] # image size: (batch_size, 3, 120, 120)
        print('finished')

    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_transformer'
    save_path = f'results/diffuser/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # # --------------- Network Architecture -----------------
    # nn_diffusion = ChiTransformer(
    #     action_dim, observation_dim, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
    #     timestep_emb_type="positional").to(device)
    # # dropout=0.0 to use no CFG but serve as FiLM encoder
    # nn_condition = MultiImageObsCondition(
    #     shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
    #     crop_shape=args.crop_shape, random_crop=args.random_crop, 
    #     use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(device)

    # # --------------- Diffusion Model --------------------
    # '''x max and x min'''
    # x_max = torch.ones((1, args.horizon, action_dim), device=device) * +10.0
    # x_min = torch.ones((1, args.horizon, action_dim), device=device) * -10.0

    # agent = DDPM(
    #     nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=device,
    #     diffusion_steps=args.sample_steps, optim_params={"lr": lr},
    #     ema_rate=args.ema_rate, predict_noise=args.predict_noise, x_max=x_max, x_min=x_min)

    # # ---------------------- Training ----------------------
    # diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    # agent.train()
    # n_gradient_step = 0
    # log = {"avg_loss_diffusion": 0.}

    # for batch in loop_dataloader(dataloader):

    #     x = batch.trajectories.to(device) # (batch_size, horizon, observation_dim + action_dim)
    #     nobs = x[:,:,action_dim:]
    #     naction = x[:,:,:action_dim]
    #     condition = nobs[:, :args.obs_steps, :] # (batch_size, obs_steps, observation_dim) (64,1,28)

    #     # ----------- Gradient Step ------------
    #     current_loss = agent.update(naction, condition)['loss'] # domain_rand loss
    #     log["avg_loss_diffusion"] += current_loss  # BaseDiffusionSDE.update
    #     # print(f'[t={n_gradient_step + 1}] diffusion loss = {current_loss}')
    #     diffusion_lr_scheduler.step()

    #     # ----------- Logging ------------
    #     if (n_gradient_step + 1) % args.log_interval == 0:
    #         log["gradient_steps"] = n_gradient_step + 1
    #         log["avg_loss_diffusion"] /= args.log_interval
    #         wandb.log({"avg_training_loss": log["avg_loss_diffusion"]}, commit = True)
    #         print(log)
    #         log = {"avg_loss_diffusion": 0.}

    #     # ----------- Saving ------------
    #     if (n_gradient_step + 1) % args.save_interval == 0:
    #         agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
    #         agent.save(save_path + f"diffusion_ckpt_latest.pt")

    #     n_gradient_step += 1
    #     if n_gradient_step >= args.diffusion_gradient_steps:
    #         break

    