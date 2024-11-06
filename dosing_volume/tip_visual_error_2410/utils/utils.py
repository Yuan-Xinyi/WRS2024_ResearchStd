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

def unnormalize_label(normalized_label, num_classes=64):
    return normalized_label * (num_classes - 1) + 1


# seed_everything(1)
# action_label = 39
# action = normalize_label(action_label)
# print('the normalized a:', action)
# print('the unnormalized a:', unnormalize_label(action))


def uniform_normalize_label(label, num_classes=60):
    range_size = 1.0 / num_classes
    lower_bound = (label - 1) * range_size

    return lower_bound + random.uniform(0, range_size)


def uniform_unnormalize_label(normalized_label, num_classes=60):
    range_size = 1.0 / num_classes

    return int(normalized_label // range_size) + 1

seed_everything(1)
action_label = 39
action = normalize_label(action_label)
print('The normalized action:', action)
print('The unnormalized action:', unnormalize_label(action))