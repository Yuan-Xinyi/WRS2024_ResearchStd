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



# def uniform_normalize_label(label, num_classes=60):
#     range_size = 1.0 / num_classes
#     lower_bound = (label - 1) * range_size

#     return lower_bound + random.uniform(0, range_size)


# def uniform_unnormalize_label(normalized_label, num_classes=60):
#     range_size = 1.0 / num_classes

#     return int(normalized_label // range_size) + 1

import random

import random

def uniform_normalize_label(label, num_classes=60, scale=10):
    """
    Normalize the label to the range (-scale, scale) using uniform distribution.
    Args:
        label (int): The input label in the range [1, num_classes].
        num_classes (int): The total number of classes (default is 60).
        scale (float): The desired range for normalization (default is 10).
    Returns:
        float: The normalized label in the range (-scale, scale).
    """
    range_size = 2.0 * scale / num_classes  # Scale to the range (-scale, scale)
    lower_bound = (label - 1) * range_size - scale  # Normalized value in the range (-scale, scale)

    return lower_bound + random.uniform(0, range_size)


def uniform_unnormalize_label(normalized_label, num_classes=60, scale=10):
    """
    Unnormalize the normalized label back to the original label in the range [1, num_classes].
    Args:
        normalized_label (float): The normalized label in the range (-scale, scale).
        num_classes (int): The total number of classes (default is 60).
        scale (float): The desired range for unnormalization (default is 10).
    Returns:
        int: The unnormalized label in the range [1, num_classes].
    """
    range_size = 2.0 * scale / num_classes  # Scale to the range (-scale, scale)

    # Map the normalized label back to [0, num_classes-1], then to the original label range
    original_label = (normalized_label + scale) // range_size + 1
    return int(original_label)



def MinMaxNormalize(X):
    """
    Normalize the input data X to the range [-1, 1] using min-max scaling.
    Args:
        X (np.ndarray): Input data (2D array or higher).
    Returns:
        np.ndarray: Normalized data in the range [-1, 1].
    """
    X = X.reshape(-1, X.shape[-1]).astype(np.float32)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero, if min == max for a feature.
    normalized_data = (X - min_vals) / range_vals  # Normalize to [0, 1]
    normalized_data = normalized_data * 2 - 1  # Scale to [-1, 1]
    
    return normalized_data, min_vals, range_vals


def unMinMaxNormalize(X, min_vals, range_vals):
    """
    Reverse the normalization and return the original data.
    Args:
        X (np.ndarray): Normalized data (values in the range [-1, 1]).
        min_vals (np.ndarray): The minimum values of the original data.
        range_vals (np.ndarray): The range (max - min) of the original data.
    Returns:
        np.ndarray: The unnormalized data in its original range.
    """
    X = X.astype(np.float32)
    X = (X + 1) / 2  # Reverse scaling from [-1, 1] to [0, 1]
    unnormalized_data = X * range_vals + min_vals  # Reverse min-max scaling
    
    return unnormalized_data

def reshape_condition(img, view=1, t=1):
    """
    Reshape the input image tensor to the desired shape.
    Args:
        img (torch.Tensor): The input image tensor of shape (batch, 3, 120, 120).
        view (int): The number of views (default is 1).
        t (int): The number of frames per view (default is 1).
    Returns:
        torch.Tensor: The reshaped image tensor of shape (batch, view, time, 3, 120, 120).
    """
    batch_size, channels, height, width = img.shape
    return img.view(batch_size, view, t, channels, height, width)
