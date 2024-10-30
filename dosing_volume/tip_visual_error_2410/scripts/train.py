import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
from utils.utils import seed_everything, device_check
from utils.dataset import TipsDataset, load_data

'''cleandiffuser imports'''
from cleandiffuser.diffusion.diffusionsde import BaseDiffusionSDE, DiscreteDiffusionSDE, BaseDiffusionSDE
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp, JannerUNet1d
from cleandiffuser.utils import report_parameters

'''preparations'''
device_check()
seed = 0
seed_everything(0)
dataset_dir_RIKEN = "dosing_volume/tip_visual_error_2410/data/RIKEN_yokohama_tip_D405/img_2/"

'''parameters'''
# training parameters
batch_size = 32
epochs = 100
lr = 3e-5
gamma = 0.7
device = 'cuda'

# image parameters
image_size=(120,120)
patch_size=10
dim=128


if __name__ == '__main__':
    train_list, valid_list = load_data(dataset_dir_RIKEN, seed)
    train_data = TipsDataset(train_list)
    valid_data = TipsDataset(valid_list)

    # Create data loaders
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle = True)