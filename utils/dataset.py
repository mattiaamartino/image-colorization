"""
dataset.py

This module defines the CocoDataset class for loading images from the COCO dataset.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze

"""

import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage.color import rgb2lab
from typing import Optional

class CocoDataset(Dataset):
    """
    A dataset class for loading images from the COCO dataset.

    This dataset loads images from a directory containing COCO dataset images and provides access to the RGB images along with their corresponding L, A, and B channels in LAB color space.

    Args:
        root (str): Path to the directory containing the image files.
        transform (Optional[torchvision.transforms.Compose]): Transformation to apply to the images. Defaults to None.

    Attributes:
        root (str): The root directory containing the image files.
        transform (Optional[torchvision.transforms.Compose]): The transformation applied to the images.
        image_paths (List[str]): List of paths to the image files in the dataset.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index: int): Retrieves a sample from the dataset.

    Raises:
        ValueError: If no images are found in the specified directory.

    """
    
    def __init__(self, root: str, transform: Optional[torchvision.transforms.Compose] = None):
        """
        Initialize the CocoDataset.

        Args:
            root (str): Path to the directory containing the image files.
            transform (Optional[torchvision.transforms.Compose]): Transformation to apply to the images. Defaults to None.

        """
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]
        if not self.image_paths:
            raise ValueError("No images found in the directory. Check the directory path.")
        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset. It is assumed that transform object containts ToTensor()

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            [RGB image tensor: Size([3, H, W]),
            L channel tensor: Size([1, H, W]),
            A channel tensor: Size([H, W]),
            B channel tensor: Size([H, W]),
            AB channels tensor: Size([2, H, W])]
            All tensors are in the range [0,1]
        """
        try:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert("RGB") # image now is in [0,255]
            if self.transform:
                image = self.transform(image)
            image = image.permute(1, 2, 0)  # Bring to HxWxC for conversion after transform
            lab_image = rgb2lab(image)  # rgb2lab needs inputs in [0, 1] --> out in L [0-100], a [-128, 127], b [-128-127]
            l_channel = torch.from_numpy(lab_image[:, :, 0] / 100).unsqueeze(0)  # unsqueeze gives us shape [1, H, W]
            ab_channels = torch.from_numpy((lab_image[:, :, 1:] + 128) / 255).permute(2, 0, 1)  # torch.Size([2, H, W])
            a_channel = ab_channels[0, :, :]  # Extract A channel
            b_channel = ab_channels[1, :, :]  # Extract B channel
            image = image.permute(2, 0, 1)  # Convert RGB image back to CxHxW
            return image, l_channel, a_channel, b_channel, ab_channels  # All are in [0, 1]
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            raise
