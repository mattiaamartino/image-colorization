"""
metrics.py

This module provides functions for calculating various evaluation metrics for image colorization models,
including Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM),
and Fréchet Inception Distance (FID).

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import functional as TF
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

def _calculate_fid(real_features: np.ndarray, fake_features: np.ndarray):
    """
    Calculate the Fréchet Inception Distance (FID) between real and fake features.

    Args:
        real_features (numpy.ndarray): Features extracted from real images.
        fake_features (numpy.ndarray): Features extracted from fake (generated) images.

    Returns:
        float: The Fréchet Inception Distance (FID) score.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, device: str):
    """
    Evaluate the model on the given data loader and compute various metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        device (str): The device to use for computations.

    Returns:
        tuple: A tuple containing the following metrics:
            - avg_mse (float): Average Mean Squared Error.
            - std_mse (float): Standard Deviation of Mean Squared Error.
            - avg_psnr (float): Average Peak Signal-to-Noise Ratio.
            - std_psnr (float): Standard Deviation of Peak Signal-to-Noise Ratio.
            - avg_ssim (float): Average Structural Similarity Index Measure.
            - std_ssim (float): Standard Deviation of Structural Similarity Index Measure.
            - fid (float): Fréchet Inception Distance.
    """
    model.eval()
    # Initialize Inception v3 model
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception.eval()

    mse_list = []
    psnr_list = []
    ssim_list = []
    real_features = []
    fake_features = []

    with torch.no_grad():
        for _, l_channels, _, _, ab_channels in data_loader:
            l_channels = l_channels.to(device)
            ab_channels = ab_channels.to(device)
            ab_preds = model(l_channels)

            # Compute MSE
            mse = F.mse_loss(ab_preds, ab_channels)
            mse_list.append(mse.item())

            # Compute PSNR
            max_pixel_value = 1.0
            psnr = 20 * torch.log10(max_pixel_value**2 / mse)
            psnr_list.append(psnr.item())

            # Compute SSIM
            ab_preds_np = ab_preds.cpu().numpy()
            ab_channels_np = ab_channels.cpu().numpy()
            for pred, target in zip(ab_preds_np, ab_channels_np):
                ssim_value, _ = ssim(pred.transpose(1, 2, 0), target.transpose(1, 2, 0), data_range=1.0, channel_axis=2, win_size=11, full=True)
                ssim_list.append(ssim_value)

            # Prepare data for FID calculation
            # Expand L channel to 3 channels
            l_channels_expanded = torch.cat((l_channels, l_channels, l_channels), dim=1)
            # Convert grayscale to 3-channel
            # Combine L channel with AB channels to form LAB images
            real_images_lab = torch.cat((l_channels, ab_channels), dim=1)
            fake_images_lab = torch.cat((l_channels, ab_preds), dim=1)

            # Resize images to 299x299 and normalize to [-1, 1] for Inception model
            real_images_resized = TF.resize(real_images_lab, [299, 299])
            fake_images_resized = TF.resize(fake_images_lab, [299, 299])

            # Convert images to range [-1, 1]
            real_images_resized = (real_images_resized - 0.5) * 2
            fake_images_resized = (fake_images_resized - 0.5) * 2

            real_features_batch = inception(real_images_resized).detach().cpu().numpy()
            fake_features_batch = inception(fake_images_resized).detach().cpu().numpy()

            real_features.append(real_features_batch)
            fake_features.append(fake_features_batch)

    # Compute average and std metrics
    mse_array = np.array(mse_list)
    psnr_array = np.array(psnr_list)
    ssim_array = np.array(ssim_list)

    avg_mse = mse_array.mean()
    std_mse = mse_array.std()

    avg_psnr = psnr_array.mean()
    std_psnr = psnr_array.std()

    avg_ssim = ssim_array.mean()
    std_ssim = ssim_array.std()

    # Compute FID
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    fid = _calculate_fid(real_features, fake_features)

    return avg_mse, std_mse, avg_psnr, std_psnr, avg_ssim, std_ssim, fid