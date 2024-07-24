"""
models.py

This module provides functions and classes of the models.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

class CNN(nn.Module):
    """
    A deep Convolutional Neural Network (CNN) model.

    This model consists of nine of convolutional layers, batch normalization,
    and ReLU activations followed by two fully connected layers. The final output
    is reshaped to the expected dimensions.

    Attributes:
        height (int): Height of the input image.
        width (int): Width of the input image.
        features (nn.Sequential): Sequential container of convolutional layers, batch normalization, and ReLU activations.
        flatten (nn.Flatten): Layer to flatten the output of the convolutional layers.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer that outputs the final prediction.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """

    def __init__(self, height: int, width: int):
        """
        Initializes the CNN model with specified height and width of the input image.

        Args:
            height (int): Height of the input image.
            width (int): Width of the input image.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 2 * self.height * self.width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output tensor reshaped to (batch_size, 2, height, width).
        """
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 2, self.height, self.width)
        return x
    
class UNet(nn.Module):
    """
    A U-Net model.

    This model consists of a contracting path (encoder) and a symmetric expanding path (decoder).
    It uses convolutional layers with ReLU activations and skip connections from the contracting path to the upsampled output.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer in the encoder.
        bn1 (nn.BatchNorm2d): Batch normalization layer for conv1.
        conv2 (nn.Conv2d): Second convolutional layer in the encoder.
        bn2 (nn.BatchNorm2d): Batch normalization layer for conv2.
        conv3 (nn.Conv2d): Third convolutional layer in the encoder.
        bn3 (nn.BatchNorm2d): Batch normalization layer for conv3.
        conv4 (nn.Conv2d): Fourth convolutional layer in the encoder.
        bn4 (nn.BatchNorm2d): Batch normalization layer for conv4.
        conv5 (nn.Conv2d): Fifth convolutional layer in the bottleneck with dilation rate 2.
        bn5 (nn.BatchNorm2d): Batch normalization layer for conv5.
        t_conv1 (nn.ConvTranspose2d): First transposed convolutional layer in the decoder.
        t_bn1 (nn.BatchNorm2d): Batch normalization layer for t_conv1.
        t_conv2 (nn.ConvTranspose2d): Second transposed convolutional layer in the decoder.
        t_bn2 (nn.BatchNorm2d): Batch normalization layer for t_conv2.
        t_conv3 (nn.ConvTranspose2d): Third transposed convolutional layer in the decoder.
        t_bn3 (nn.BatchNorm2d): Batch normalization layer for t_conv3.
        t_conv4 (nn.ConvTranspose2d): Fourth transposed convolutional layer in the decoder.
        output (nn.Conv2d): Output convolutional layer with 2 output channels.
        sigmoid (nn.Sigmoid): Sigmoid activation function for the output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_bn1 = nn.BatchNorm2d(128)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_bn2 = nn.BatchNorm2d(64)
        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_bn3 = nn.BatchNorm2d(32)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Predicted A and B channel tensors of shape (batch_size, 2, height, width).
        """
        x_1 = F.relu(self.bn1(self.conv1(x)))
        x_2 = F.relu(self.bn2(self.conv2(x_1)))
        x_3 = F.relu(self.bn3(self.conv3(x_2)))
        x_4 = F.relu(self.bn4(self.conv4(x_3)))

        x_5 = F.relu(self.bn5(self.conv5(x_4)))

        x_6 = F.relu(self.t_bn1(self.t_conv1(x_5)))
        x_6 = torch.cat((x_6, x_3), 1)
        x_7 = F.relu(self.t_bn2(self.t_conv2(x_6)))
        x_7 = torch.cat((x_7, x_2), 1)
        x_8 = F.relu(self.t_bn3(self.t_conv3(x_7)))
        x_8 = torch.cat((x_8, x_1), 1)
        x_9 = F.relu(self.t_conv4(x_8))
        x_9 = torch.cat((x_9, x), 1)
        
        x = self.output(x_9)
        x = self.sigmoid(x)
        return x
    
class EncoderDecoderGenerator(nn.Module): # that of the paper
    """
    An encoder-decoder generator model for GANs.

    The architecture consists of an encoder network that downsamples the input grayscal image,
    and a decoder network that upsamples the encoded representation to produce the predicted AB channels.

    Attributes:
        encoder (nn.Sequential): A sequential container of convolutional layers, batch normalization,
                                 and LeakyReLU activations for the encoder network.
        decoder (nn.Sequential): A sequential container of transposed convolutional layers, batch normalization,
                                 and LeakyReLU activations for the decoder network.
        sigmoid (nn.Sigmoid): A sigmoid activation function applied to the final output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """
    def __init__(self):
        """
        Initializes the EncoderDecoderGenerator.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),  # image size: 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ) 

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),  # 256x256
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Predicted A and B channel tensors of shape (batch_size, 2, height, width).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x
    
class PatchGAN(nn.Module):
    """
    A PatchGAN discriminator model for GANs.

    This model takes a 3-channel image as input and outputs a single channel tensor,
    representing the probability that the input image is real or generated for each path.
    The architecture consists of several convolutional layers with LeakyReLU activations,
    batch normalization, and a final sigmoid activation function.
    The final output is of size 16x16.
    
    Attributes:
        conv (nn.Sequential): A sequential container of convolutional layers, batch normalization,
                              and LeakyReLU activations.
        sigmoid (nn.Sigmoid): A sigmoid activation function applied to the final output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """
    def __init__(self):
        """
        Initializes the PatchGAN model.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # image size: 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1), # 16x16
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Predicted probability map of shape (batch_size, 1, 16, 16).
        """
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class BaselineCNN(nn.Module):
    """
    A baseline CNN for image colorization. Used privately to test functions and other internal things before scaling things up.

    This model takes a grayscale (single-channel) image as input and outputs a two-channel tensor
    representing the predicted A and B channels of the LAB color space. The architecture consists
    of several convolutional layers with ReLU activations, and the final output is passed through a
    sigmoid activation function to ensure the output values are between 0 and 1.

    Attributes:
        layers (nn.Sequential): A sequential container of convolutional layers and ReLU activations.
        sigmoid (nn.Sigmoid): A sigmoid activation function applied to the final output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initializes the baseline model.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Predicted A and B channel tensors of shape (batch_size, 2, height, width).
        """
        x = self.layers(x)
        x = self.sigmoid(x)
        return x

def save_model(model: torch.nn.Module, model_name: str, model_dir: Optional[str] = "models"):
    """
    Save a PyTorch model to a pth file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the model file (e.g., "baseline"). No extension needed.
        model_dir (Optional[str]): The directory where the model file will be saved. Default is "models".
    """
    # create the model directory if it doesn't exist
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    # construct the full path for the model file
    model_name = f"{model_name}.pth"
    model_path = model_dir_path / model_name

    torch.save(model.state_dict(), model_path) # save the model state dict
    print(f"Model saved to {model_path} successfully!")
    
def load_model(model: torch.nn.Module, model_path: str):
   """
   Load a PyTorch model from a file.

   Args:
       model (torch.nn.Module): The PyTorch model object to load the weights into.
       model_path (str): The path to the model file (e.g., "models/baseline.pth").

   Returns:
       torch.nn.Module: The model object with the loaded weights.
   """
   model.load_state_dict(torch.load(model_path))
   print(f"{model._get_name()} model loaded successfully!")
   return model

