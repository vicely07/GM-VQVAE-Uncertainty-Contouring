"""
The following script contains utility functions and operations related to medical image processing 
and deep learning using PyTorch. It includes functions for data preprocessing, data splitting, 
CSV file manipulation, image postprocessing, and loading data for deep learning models.
"""

from typing import Tuple, List, Optional, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import numpy as np
#import nibabel as ni
import os, shutil
import time
import random
import pandas as pd
import numpy as np
import os
import cv2
import numpy as np
import os
import cv2
from scipy import ndimage
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Optional
from scipy.ndimage import center_of_mass

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_around_centroid(array: np.ndarray, dim1: int) -> np.ndarray:
    """
    Crop array around its centroid.

    Args:
        array (numpy.ndarray): Input array.
        dim1 (int): Dimension parameter.

    Returns:
        numpy.ndarray: Cropped array.
    """
    i, j = center_of_mass(array)
    i, j = int(i), int(j)
    w = int(dim1/2)
    imin = max(0, i-w)
    imax = min(array.shape[0], i+w+1)
    jmin = max(0, j-w)
    jmax = min(array.shape[1], j+w+1)
    crop = array[imin:imax, jmin:jmax]
    return crop

def standard_resize2d(image: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
    """
    Resize 2D image.

    Args:
        image (numpy.ndarray): Input image.
        dim (tuple): Target dimensions.

    Returns:
        numpy.ndarray: Resized image.
    """
    resize_x, resize_y = dim[0], dim[1]
    img_sm = cv2.resize(image, (resize_x, resize_y), interpolation=cv2.INTER_CUBIC)
    return img_sm

def split_train_test(directory: str, ratio_test: float = 0.15) -> None:
    """
    Split data into train and test sets.

    Args:
        directory (str): Directory containing data.
        ratio_test (float, optional): Ratio of data to use for testing. Defaults to 0.15.
    """
    if not os.path.exists(os.path.join(directory, "train")):
        os.mkdir(os.path.join(directory, "train"))
    if not os.path.exists(os.path.join(directory, "test")):
        os.mkdir(os.path.join(directory, "test"))

    images_list = [i for i in os.listdir(directory) if i.endswith(".nii")]

    random.shuffle(images_list)
    threshold = int(len(images_list)*ratio_test)
    train_list = images_list[:-threshold]
    test_list = images_list[-threshold:]

    for i in train_list:
        shutil.move(os.path.join(directory, i), os.path.join(directory, "train", i))
    for i in test_list:
        shutil.move(os.path.join(directory, i), os.path.join(directory, "test", i))

def save_data_to_csv(directory: str, z: np.ndarray) -> None:
    """
    Save data to CSV file.

    Args:
        directory (str): Directory to save the CSV file.
        z (numpy.ndarray): Data to be saved.
    """
    pd.DataFrame(z).to_csv(directory, header=None, index=False)

def postprocess_mask(mask: np.ndarray, s: float) -> np.ndarray:
    """
    Postprocess segmentation mask.

    Args:
        mask (numpy.ndarray): Input segmentation mask.
        s (float): Shape parameter.

    Returns:
        numpy.ndarray: Postprocessed mask.
    """
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = np.where(mask >= 0.3, 1.0, 0.0)
    if s <= 15:
        mask_blur = gaussian_filter(mask, sigma=7)
    elif 10 < s <= 15:
        mask_blur = gaussian_filter(mask, sigma=5)
    else:
        mask_blur = gaussian_filter(mask, sigma=3)

    return mask_blur

def load_data_images(path: str, batch_size: int) -> Iterator[torch.Tensor]:
    """
    Load data images and perform preprocessing.

    Args:
        path (str): Path to the directory containing images.
        batch_size (int): Batch size.

    Yields:
        torch.Tensor: Batch of processed images.
    """
    filenames = [i for i in os.listdir(path) if i.endswith(".npy")]
    random.shuffle(filenames)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                break

            image = np.load(os.path.join(path, filenames[i]), allow_pickle=True)
            image = np.where(image >= 1e-3, image, 0.0)
            image = crop_around_centroid(image, dim1=240)
            image = np.pad(image, ((1, 0), (1, 0)), "constant", constant_values=0)
            dim = (256, 256)
            image = torch.Tensor(standard_resize2d(image, dim))
            image = torch.reshape(image, (1, 1, 256, 256))
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            image = torch.where(image >= 0.1, image, 0.0)
            batch_image.append(image)

        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image