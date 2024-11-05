import random
import os
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def device_check():
    print(torch.__version__)
    print(torch.cuda.is_available())

def normalize_label(label, num_classes=64):
    return (label - 1) / (num_classes - 1)
