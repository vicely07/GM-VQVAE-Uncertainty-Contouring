
"""
Medical Image Evaluation Metrics

This module provides a collection of functions for calculating surface and volume metrics to evaluate the similarity between predicted and ground truth binary masks in medical image segmentation tasks. The metrics include Surface Dice Similarity Coefficient (DSC) at various tolerances, Hausdorff distance, and mask volume calculation.

These functions utilize the 'surface_distance' library to compute surface distances and relevant metrics. The metrics can be useful in assessing the accuracy and agreement between predicted and ground truth segmentations, crucial for evaluating the performance of medical image segmentation algorithms.

Functions:
- `calculate_surface_dsc`: Calculate the Surface Dice Similarity Coefficient at 3mm tolerance.
- `calculate_surface_dsc_2mm`: Calculate the Surface Dice Similarity Coefficient at 2mm tolerance.
- `calculate_surface_dscmm`: Calculate the Surface Dice Similarity Coefficient at 1mm tolerance.
- `calculate_hausdorff_dist`: Calculate the Hausdorff distance between two binary masks.
- `calculate_volume`: Calculate the volume of a binary mask.

Note: Ensure that the 'surface_distance' library is installed before using these functions.

Author: Vi Ly
"""

import numpy as np
import os
import pandas as pd
import surface_distance
from typing import Tuple, List, Optional, Iterator
import numpy as np
import surface_distance

def calculate_surface_dsc(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calculate the Surface Dice Similarity Coefficient (DSC) between two binary masks.

    Args:
        mask_pred (np.ndarray): Predicted binary mask.
        mask_gt (np.ndarray): Ground truth binary mask.

    Returns:
        float: Surface DSC value.
    """
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 3)
    return surface_dsc

def calculate_surface_dsc_2mm(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calculate the Surface Dice Similarity Coefficient (DSC) at 2mm tolerance.

    Args:
        mask_pred (np.ndarray): Predicted binary mask.
        mask_gt (np.ndarray): Ground truth binary mask.

    Returns:
        float: Surface DSC value at 2mm tolerance.
    """
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 2)
    return surface_dsc

def calculate_surface_dscmm(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calculate the Surface Dice Similarity Coefficient (DSC) at 1mm tolerance.

    Args:
        mask_pred (np.ndarray): Predicted binary mask.
        mask_gt (np.ndarray): Ground truth binary mask.

    Returns:
        float: Surface DSC value at 1mm tolerance.
    """
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 1)
    return surface_dsc

def calculate_hausdorff_dist(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two binary masks.

    Args:
        mask_pred (np.ndarray): Predicted binary mask.
        mask_gt (np.ndarray): Ground truth binary mask.

    Returns:
        float: Hausdorff distance.
    """
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_robust_hausdorff(surface_distances, 100)
    return surface_dsc

def calculate_volume(voxel_array: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """
    Calculate the volume of a binary mask.

    Args:
        voxel_array (np.ndarray): Binary mask.
        voxel_size (Tuple[float, float, float]): Voxel size in millimeters.

    Returns:
        float: Volume of the binary mask.
    """
    voxel = np.sum(voxel_array) * np.array(voxel_size)
    return voxel[0] * voxel[1] * voxel[2]
