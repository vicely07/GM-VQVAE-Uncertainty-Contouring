"""
VQ-VAE Training Script

This script trains a VQ-VAE model using PyTorch. It includes data loading,
model initialization, training loop, and validation. The script saves the
trained model at each epoch.

Make sure to replace placeholders in the code with your specific paths and
settings before running the script.

Author: Vi Ly
"""

# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
import time
import pandas as pd
import cv2
from scipy import ndimage
import torchvision.transforms.functional as TF
import glob
from tqdm import tqdm
from model.gm_vqvae import VQVAE, Encoder, ResBlock, Quantize, Decoder
from utils.dataloader import load_data_images
from utils.loss import *

# Setting hyperparameters
batch_size = 32
lrate = 0.001
epochs = 100
weight_decay = 1e-6  # Adjust as needed

# Paths and settings
train_path_data = "../data/seg/2d_slices/train/Joint/"
val_path_data = "../data/seg/2d_slices/val/Joint/"
path2save = "../checkpoints/gmvqvae/model_vae_epoch_{}.pt"

# Other settings
verbose = True
np.random.seed(10)
torch.manual_seed(10)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
train_no_images = len(glob.glob(train_path_data + "/*.npy"))
val_no_images = len(glob.glob(val_path_data + "/*.npy"))
print("Number of train fm images: ", train_no_images)
print("Number of val fm images: ", val_no_images)
criterion = nn.MSELoss()


def weights_init(m: nn.Module) -> None:
    """Initialize weights for Conv2d layers using Xavier uniform."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

if __name__ == "__main__":
    # Model initialization and setup
    vae_model = VQVAE()
    vae_model.to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lrate, weight_decay=weight_decay)
    vae_model.apply(weights_init)
    min_valid_loss = np.inf
    train_batch_avg_total_loss_list = []
    val_batch_avg_total_loss_list = []
    beta = 0.25

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Training phase
        train_loss_rec_epoch, train_loss_KL_epoch, train_total_loss_epoch = 0, 0, 0
        vae_model.train()
        for train_batch_images in tqdm(load_data_images(train_path_data, batch_size)):
            optimizer.zero_grad()
            train_batch_images = train_batch_images.to(device)
            train_out, train_latent_loss = vae_model(train_batch_images, s=1)
            train_recon_loss = criterion(train_out, train_batch_images)
            train_latent_loss = train_latent_loss.mean()
            train_total_loss_batch =  train_recon_loss + beta * train_latent_loss
            train_total_loss_batch.backward()
            optimizer.step()
            train_total_loss_epoch += train_total_loss_batch.item() * train_batch_images.shape[0]

        # Validation phase
        val_loss_rec_epoch, val_loss_KL_epoch, val_total_loss_epoch = 0, 0, 0
        vae_model.eval()
        for val_batch_images in tqdm(load_data_images(val_path_data, batch_size)):
            optimizer.zero_grad()
            val_batch_images = val_batch_images.to(device)
            val_out, val_latent_loss = vae_model(val_batch_images, s=1)
            val_recon_loss = criterion(val_out, val_batch_images)
            val_latent_loss = val_latent_loss.mean()
            val_total_loss_batch =  val_recon_loss + beta * val_latent_loss
            val_total_loss_batch.backward()
            optimizer.step()
            val_total_loss_epoch += val_total_loss_batch.item() * val_batch_images.shape[0]

        # Logging and saving
        train_log_info = (epoch + 1, epochs, train_loss_rec_epoch/train_no_images, train_loss_KL_epoch/train_no_images, train_total_loss_epoch/train_no_images)
        log('%d/%d  Train: Reconstruction Loss %.3f| KL Loss %.3f | Total Loss %.3f'% train_log_info)
        train_batch_avg_total_loss_list.append(train_total_loss_epoch/train_no_images)

        val_log_info = (epoch + 1, epochs,  val_loss_rec_epoch/val_no_images, val_loss_KL_epoch/val_no_images, val_total_loss_epoch/val_no_images)
        log('%d/%d  Validation: Reconstruction Loss %.3f| KL Loss %.3f | Total Loss %.3f'% val_log_info)
        val_batch_avg_total_loss_list.append(val_total_loss_epoch/val_no_images)
        epoch_log_df = pd.DataFrame({"train_loss": train_batch_avg_total_loss_list, "val_loss": val_batch_avg_total_loss_list})
        print(epoch_log_df)
        if min_valid_loss > val_total_loss_epoch/val_no_images:
          print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_total_loss_epoch/val_no_images:.6f}) \t Saving The Model')
          min_valid_loss = val_total_loss_epoch/val_no_images
          # Saving State Dict
          torch.save(vae_model, path2save.format(epoch+1))

  