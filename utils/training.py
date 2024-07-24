"""
train.py

This module provides functions for training the models for image colorization
and additional auxiliary functions.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils.models import save_model
import os
from pathlib import Path
from typing import Optional, List
import numpy as np

###########
### CNN ###
###########

def train_cnn(epochs: int, model: torch.nn.Module, criterion: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,
              train_loader: DataLoader, test_loader: DataLoader,
              device: Optional[str] = "cuda",
              save_losses: Optional[bool] = False, save_checkpoints: Optional[bool] = False, file_name: Optional[str] = ""):
    """
    Function to train the CNN model for image colorization. It also works for the UNet training.

    Args:
        epochs (int): Number of training epochs.
        model (torch.nn.Module): PyTorch model to be trained.
        criterion (torch.nn.modules.loss._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for the training data.
        test_loader (torch.utils.data.dataloader.DataLoader): DataLoader for the validation data.
        device (Optional[str]): Device to use for training (e.g., "cuda" or "cpu"). Default is "cuda".
        save_losses (Optional[bool]): Whether to save the training and validation losses to a file. Default is False.
        save_checkpoints (Optional[bool]): Whether to save the model checkpoints during training. Default is False.
        file_name (Optional[str]): Base name for saving model checkpoints and losses file. Default is an empty string.

    Returns:
        tuple: A tuple containing two lists:
            - train_losses (List[float]): List of training losses for each epoch.
            - test_losses (List[float]): List of validation losses for each epoch.
    """
    train_losses = []
    test_losses = []
    
    if save_checkpoints and (epochs < 10):
        print("Checkpoints will not be saved. Epochs must be greater than or equal to 10 for saving checkpoints.")

    for epoch in range(epochs):
        running_loss = 0.0
        test_loss = 0.0
        model.train()

        for _, l_channels, _, _, ab_channels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True):
            l_channels, ab_channels = l_channels.to(device), ab_channels.to(device)
            optimizer.zero_grad()
            outputs = model(l_channels)
            loss = criterion(outputs, ab_channels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            for _, l_channels, _, _, ab_channels in tqdm(test_loader, desc='Testing', leave=True):
                l_channels, ab_channels = l_channels.to(device), ab_channels.to(device)
                outputs = model(l_channels)
                loss = criterion(outputs, ab_channels)
                test_loss += loss.item()

        test_losses.append(test_loss / len(test_loader))
        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Validation Loss: {test_losses[-1]}")

        if save_checkpoints and (epochs > 10):
            if epoch in [int(epochs / 3) - 1, 2 * int(epochs / 3) - 1, epochs - 1]:
                # save three checkpoints
                save_model(model, f"{file_name}_{epoch + 1}")            

    if save_losses:
        _save_losses_cnn(train_losses, test_losses, file_name)

    print('Finished Training')
    return train_losses, test_losses

def _save_losses_cnn(train_losses: List[float], test_losses: List[float], file_name: str, save_dir: str = "losses"):
    """
    Save the training and validation losses to a txt file in the specified directory.

    Args:
        train_losses (List[float]): List of training losses for each epoch.
        test_losses (List[float]): List of validation losses for each epoch.
        file_name (str): Name of the txt file.
        save_dir (str, optional): Directory where the losses will be saved. Default is "losses".
    """
    # Create the save directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = save_dir_path / f"{file_name}.txt"

    with open(file_path, "w") as f:
        f.write("Epoch,Train Loss,Validation Loss\n")
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), start=1):
            f.write(f"{epoch},{train_loss},{test_loss}\n")

    print(f"Losses saved to {file_path}")

###########
### GAN ###
###########

def _save_losses_gan(d_losses: List[float], g_losses: List[float], file_name: str, save_dir: str = "losses"):
    """
    Save the discriminator and generator losses to a txt file in the specified directory.

    Args:
        d_losses (List[float]): List of discriminator losses for each epoch.
        g_losses (List[float]): List of generator losses for each epoch.
        file_name (str): Name of the txt file.
        save_dir (str, optional): Directory where the losses will be saved. Default is "losses".
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = save_dir_path / f"{file_name}_losses.txt"

    with open(file_path, "w") as f:
        f.write("Epoch,Discriminator Loss,Generator Loss\n")
        for epoch, (d_loss, g_loss) in enumerate(zip(d_losses, g_losses), start=1):
            f.write(f"{epoch},{d_loss},{g_loss}\n")

    print(f"Losses saved to {file_path}")

def train_gan(epochs: int, discriminator: torch.nn.Module, generator: torch.nn.Module,
              disc_opt: torch.optim.Optimizer, gen_opt: torch.optim.Optimizer,
              criterion: torch.nn.modules.loss._Loss, train_loader: DataLoader,
              device: Optional[str] = "cuda", l1_lambda: float = 0.0, label_smoothing: Optional[bool] = False,
              add_noise: Optional[bool] = False, noise_std: float = 0.1,
              save_losses: Optional[bool] = False, save_checkpoints: Optional[bool] = False, file_name: Optional[str] = ""):
    """
    Function to train the GAN model with optional label smoothing, 
    optional L1 regularization for generator and optional noise to discriminator inputs.

    Args:
        epochs (int): Number of training epochs.
        discriminator (torch.nn.Module): Discriminator model.
        generator (torch.nn.Module): Generator model.
        disc_opt (torch.optim.Optimizer): Optimizer for the discriminator.
        gen_opt (torch.optim.Optimizer): Optimizer for the generator.
        criterion (torch.nn.modules.loss._Loss): Loss function (e.g., BCELoss).
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for the training data.
        device (Optional[str]): Device to use for training (e.g., "cuda" or "cpu"). Default is "cuda".
        l1_lambda (float): Weight for the L1 loss component. Default is 0.0 (no regularization).
        label_smoothing (Optional[bool]): Applies label smoothing to true labels to reduce overconfidence. Smoothing value is in [0.9, 0.95].
        add_noise (Optional[bool]): Adds noise to the discriminator inputs to prevent vanishing gradients. Default is False.
        noise_std (float): Standard deviation of the noise to add to the discriminator inputs. Default is 0.1.
        save_losses (Optional[bool]): Whether to save the training losses to a file. Default is False.
        save_checkpoints (Optional[bool]): Whether to save the generator checkpoints during training. Default is False.
        file_name (Optional[str]): Base name for saving model checkpoints and losses file. Default is an empty string.

    Returns:
        tuple: A tuple containing two lists:
            - d_losses (List[float]): List of discriminator losses for each epoch.
            - g_losses (List[float]): List of generator losses for each epoch.
    """
    d_losses = []
    g_losses = []

    if save_checkpoints and (epochs < 10):
        print("Checkpoints will not be saved. Epochs must be greater than or equal to 10 for saving checkpoints.")

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        discriminator.train()
        generator.train()

        loop = tqdm(train_loader, leave=True, desc=f'Epoch {epoch + 1}/{epochs}')

        for _, l, _, _, ab in loop:
            l, ab = l.to(device), ab.to(device)

            # Train Discriminator
            disc_opt.zero_grad()
            fake_ab = generator(l).detach()
            fake_lab = torch.cat((l, fake_ab), dim=1)
            real_lab = torch.cat((l, ab), dim=1)
            
            # Optionally add noise to discriminator inputs
            if add_noise:
                noise = torch.randn_like(fake_lab, device=device) * noise_std
                fake_lab += noise
                real_lab += noise

            pred_fake = discriminator(fake_lab)
            pred_real = discriminator(real_lab)
            
            if label_smoothing:
                # Apply label smoothing only to real labels
                smooth_real_labels = torch.rand_like(pred_real, device=device) * 0.05 + 0.90  # Smooth real labels between 0.9 and 0.95
                loss_real = criterion(pred_real, smooth_real_labels)
                loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake, device=device))
            else:
                loss_real = criterion(pred_real, torch.ones_like(pred_real, device=device))
                loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake, device=device))

            d_loss = (loss_fake + loss_real) / 2
            d_loss.backward()
            disc_opt.step()
            epoch_d_loss += d_loss.item()

            # Train Generator
            gen_opt.zero_grad()
            fake_ab = generator(l)
            fake_lab = torch.cat((l, fake_ab), dim=1)
            pred_fake_gen = discriminator(fake_lab)
            g_loss_adv = criterion(pred_fake_gen, torch.ones_like(pred_fake_gen, device=device))
            if l1_lambda > 0.0:
                g_loss_l1 = F.l1_loss(fake_ab, ab)  # L1 loss component
                g_loss = g_loss_adv + l1_lambda * g_loss_l1  # Combined GAN loss with L1 regularization
            else:
                g_loss = g_loss_adv  # Standard GAN loss
            g_loss.backward()
            gen_opt.step()
            epoch_g_loss += g_loss.item()

            loop.set_postfix(d_loss=epoch_d_loss / (loop.n + 1), g_loss=epoch_g_loss / (loop.n + 1))            

        if save_checkpoints and (epochs > 10):
            if epoch in [int(epochs / 4) - 1, 2 * int(epochs / 4) - 1, 3 * int(epochs / 4) - 1, epochs - 1]:
                # save four checkpoints
                save_model(generator, f"{file_name}_{epoch + 1}")

        d_losses.append(epoch_d_loss / len(train_loader))
        g_losses.append(epoch_g_loss / len(train_loader))

    if save_losses:
        _save_losses_gan(d_losses, g_losses, file_name)

    print('Finished Training')
    return d_losses, g_losses

###############
#### EXTRA ####
###############

def load_losses(file_path: str):
    """
    Load training and validation losses from a file into NumPy arrays.

    Args:
        file_path (str): Path to the file containing the losses.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - train_losses (np.ndarray): Array of training losses.
            - val_losses (np.ndarray): Array of validation losses.
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # skipping the header row

    train_losses = data[:, 1]
    val_losses = data[:, 2]

    return train_losses, val_losses