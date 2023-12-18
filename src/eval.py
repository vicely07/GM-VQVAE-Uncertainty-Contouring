
"""
This script evaluates the performance of GM-VQVAE models on different test sets.
It calculates Surface Dice Similarity Coefficient (DSC) and Hausdorff Distance metrics
to assess the accuracy of the models in segmenting regions of interest (ROIs) in 3D medical images.

Make sure to adapt paths and models according to your specific directory structure.

Note: This script assumes the existence of the following functions and classes:
    - GM_VQVAE, Encoder, ResBlock, Quantize, Decoder (from model.gm_vqvae)
    - standard_resize2d (from utils.dataloader)
    - calculate_surface_dsc, calculate_hausoff_dist (from utils.evaluation_metrics)

Author: Vi Ly
"""

import glob
import torch 
from model.gm_vqvae import VQVAE, Encoder, ResBlock, Quantize, Decoder
import numpy as np 
import pandas as pd 
from scipy import ndimage
from utils.dataloader import standard_resize2d
from utils.evaluation_metrics import calculate_surface_dsc, calculate_hausdorff_dist
import matplotlib.pyplot as plt
import cv2 
from typing import Tuple, List, Optional, Iterator


def evaluate_step(path_data_list: List[str], path_model_list: List[str], 
                   model_name_list: List[str], threshold: float, 
                   label_list: List[str]) -> pd.DataFrame:
    """
    Evaluate multiple models on a test set and calculate Surface DSC and Hausdorff Distance.

    Parameters:
    - path_data_list (list): List of paths to test data.
    - path_model_list (list): List of paths to model checkpoints.
    - model_name_list (list): List of model names.
    - threshold (float): Threshold for binarizing the predicted mask.
    - label_list (list): List of labels for specific ROI evaluation.

    Returns:
    - metric_tbl (DataFrame): DataFrame containing evaluation metrics for each model.
    """
    mean_dsc_list = []
    mean_hd_list = []
    for i in range(len(path_model_list)):
        mean_dsc, mean_hd = evaluate_testset(path_data_list[i], path_model_list[i], threshold, label_list[i])
        mean_dsc_list.append(mean_dsc)
        mean_hd_list.append(mean_hd)

    metric_tbl  = pd.DataFrame(list(zip(model_name_list, mean_dsc_list, mean_hd_list)), columns=['Model', 'Surface DSC', "Hausdroff Distance"])
    return metric_tbl


def evaluate_testset(path_data: str, path_model: str, threshold: float, 
                     label: str) -> Tuple[float, float]:
    """
    Evaluate a single model on a test set and calculate Surface DSC and Hausdorff Distance.

    Parameters:
    - path_data (str): Path to the test data.
    - path_model (str): Path to the model checkpoint.
    - threshold (float): Threshold for binarizing the predicted mask.
    - label (str): Label for specific ROI evaluation.

    Returns:
    - mean_dsc (float): Mean Surface DSC.
    - mean_hd (float): Mean Hausdorff Distance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model = torch.load(path_model, map_location=torch.device('cpu'))
    vae_model.to(device)

    dice_list = []
    hausdroff_list = []

    gt_list = glob.glob(path_data + "/*.npy")[0:1]

    for gt in gt_list:
        if label == "all":
            gt = np.load(gt)
        else:
            gt = np.load(gt)[1, ...]

        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        gt = np.where(gt >= 0.1, 1.0, 0.0)

        vae_model.eval()
        mask = pred_roi(gt, vae_model)
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = mask[1, ...]
        mask_t = np.where(mask >= threshold, 1.0, 0.0)

        sd = calculate_surface_dsc(mask_t.astype("bool"), gt.astype("bool"))
        hd = calculate_hausoff_dist(mask_t.astype("bool"), gt.astype("bool"))

        dice_list.append(sd)
        hausdroff_list.append(hd)

    return np.mean(dice_list), np.mean(hausdroff_list)

def pred_roi(gt: np.ndarray, vae_model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and resize predicted region of interest (ROI).

    Parameters:
    - gt (numpy.ndarray): Ground truth mask.
    - vae_model: Trained VAE model.

    Returns:
    - gt (numpy.ndarray): Ground truth mask.
    - pred (numpy.ndarray): Predicted mask.
    """
    gt = np.where(gt >= 0.2, gt, 0.0)
    pred = np.zeros(gt.shape)

    with torch.no_grad():
        for i in range(gt.shape[0]):
            if np.sum(gt[i, ...]) > 0:
                slice = gt.copy()[i, ...]
                crop, imin, imax, jmin, jmax = crop_around_centroid(slice, dim1=200)
                x = load_data_images(crop)
                y, latent_loss = vae_model(x.to(device), s=1)
                y = y.detach().cpu().numpy()
                dim = crop.shape
                y = standard_resize2d(y[0, 0, ...], dim)
                pred[i, imin:imax, jmin:jmax] = y.transpose(1, 0).copy()

    return gt.copy(), pred.copy()

def crop_around_centroid(array: np.ndarray, dim1: int) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Crop around the centroid of an array.

    Parameters:
    - array (numpy.ndarray): Input array.
    - dim1 (int): Dimension size.

    Returns:
    - crop (numpy.ndarray): Cropped array.
    - imin, imax, jmin, jmax (int): Crop indices.
    """
    i, j = ndimage.center_of_mass(array)
    i, j = int(i), int(j)
    w = int(dim1/2)
    imin = max(0,i-w)
    imax = min(array.shape[0],i+w+1)
    jmin = max(0,j-w)
    jmax = min(array.shape[1],j+w+1)
    crop =  array[imin:imax,jmin:jmax]

    return crop, imin, imax, jmin, jmax

def load_data_images(image: np.ndarray) -> torch.Tensor:
    """
    Load data images and perform standard preprocessing.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - image (torch.Tensor): Processed image.
    """
    image = np.pad(image, ((1,0), (1,0)), "constant", constant_values=0)
    dim = (256,256)
    image = torch.Tensor(standard_resize2d(image, dim))
    image = torch.reshape(image, (1,1, 256, 256))
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    return image

# Paths and models for evaluation
path_data = "../data/seg/3d_masks/Joint/test"
path_data_prostate = "../data/seg/3d_masks/Prostate/test/"
path_data_rectum = "../data/seg/3d_masks/Rectum/test/"
path_data_bladder = "../data/seg/3d_masks/Bladder/test/"

gm_vqvae = "../checkpoints/gmvqvae/best/model_gmvqvae_best_joint.pt"
gm_vqvae_prostate = "../checkpoints/gmvqvae/best/model_gmvqvae_best_prostate.pt"
gm_vqvae_rectum = "../checkpoints/gmvqvae/best/model_gmvqvae_best_rectum.pt"
gm_vqvae_bladder = "../checkpoints/gmvqvae/best/model_gmvqvae_best_bladder.pt"

threshold = 0.4
path_model_list = [gm_vqvae, gm_vqvae_prostate, gm_vqvae_rectum, gm_vqvae_bladder]
path_data_list = [path_data, path_data_prostate , path_data_rectum, path_data_bladder]
model_name_list = ["GM-VQVAE", "GM-VQVAE-Prostate", "GM-VQVAE-Rectum", "GM-VQVAE-Bladder"]
label_list = ["all", "prostate", "rectum", "bladder"]

metric_table = evaluate_step(path_data_list, path_model_list, model_name_list, threshold, label_list)
metric_table