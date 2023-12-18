"""
Custom Loss Functions for Neural Network Training

This module defines a set of custom loss functions commonly used in neural network training for various tasks. The implemented loss functions include:

1. `KLDivergence`: KL divergence between the estimated normal distribution and a prior distribution. This loss is often utilized in variational autoencoders (VAEs) to ensure that the learned latent space is close to a predefined distribution.

2. `L1Loss`: L1 loss, measuring the Euclidean distance between prediction and ground truth using the L1 norm. This loss is suitable for regression tasks where absolute differences between corresponding elements are important.

3. `DiceLoss`: Dice loss for semantic segmentation tasks. It measures the overlap between predicted and ground truth masks, commonly used in medical image segmentation and other pixel-wise classification tasks.

These custom loss functions are implemented as PyTorch nn.Modules, making them compatible with PyTorch's automatic differentiation for seamless integration into neural network training pipelines.

Author: Vi Ly
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class KLDivergence(nn.Module):
    """
    KL divergence between the estimated normal distribution and a prior distribution.
    
    Attributes:
        N (int): The index N spans all dimensions of input (H x W x D).
    """
    def __init__(self):
        super(KLDivergence, self).__init__()
        self.N = 80 * 80

    def forward(self, z_mean: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Compute the KL divergence.

        Args:
            z_mean (Tensor): Mean of the distribution.
            z_sigma (Tensor): Standard deviation of the distribution.

        Returns:
            Tensor: KL divergence.
        """
        z_var = z_sigma * 2
        return 0.5 * ((z_mean**2 + z_var.exp() - z_var - 1).sum())

class L1Loss(nn.Module):
    """
    Measuring the Euclidean distance between prediction and ground truth using L1 Norm.
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the L1 loss.

        Args:
            x (Tensor): Predicted tensor.
            y (Tensor): Ground truth tensor.

        Returns:
            Tensor: L1 loss.
        """
        N = y.numel()
        return ((x - y).abs()).sum() / N

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation tasks.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
        """
        Compute the Dice loss.

        Args:
            inputs (Tensor): Predicted tensor.
            targets (Tensor): Ground truth tensor.
            smooth (float): Smoothing factor.

        Returns:
            Tensor: Dice loss.
        """
        # Comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
