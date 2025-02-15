"""
DataLoader: Data Processing Pipeline for Satellite Image Datasets

This script provides a data preprocessing pipeline for satellite image datasets.
It includes functions for loading images and annotations, applying transformations (such as normalization and noise addition),
and creating PyTorch datasets and dataloaders for model training.

Key Features:
- Normalization of input images
- Gaussian noise augmentation for training data
- Custom dataset class for handling image, annotation, and mask data
- Functionality to expand date-based image lists
- Data partitioning into training, validation, and test sets
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.ndimage import rotate

### Class Definitions
class Compose(object):
    """ Combines multiple transformations into a single pipeline """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno):
        for t in self.transforms:
            img, anno = t(img, anno)
        return img, anno

class Normalize(object):
    """ Converts images and annotations to PyTorch tensors and normalizes the image """
    def __init__(self):
        pass

    def __call__(self, image, anno_img):
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert image to tensor
        anno_img = torch.from_numpy(anno_img)  # Convert annotation to tensor
        return image, anno_img

class AddGaussianNoise(object):
    """ Adds Gaussian noise to the image and ensures annotation shape consistency """
    def __init__(self, mu=0.0, sigma=0.005):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img, anno_img):
        if img.dim() != 3:
            raise ValueError(f"Expected img to have 3 dimensions (C, H, W), but got {img.shape}")
        
        C, H, W = img.shape  # Extract dimensions
        noise = torch.randn_like(img) * self.sigma + self.mu  # Generate noise
        img = img + noise  # Apply noise to image

        anno_img = anno_img.squeeze(-1)  # Ensure annotation has correct shape
        return img, anno_img

class DataTransform(object):
    """ Applies specified transformations for different dataset phases (train/valid/test) """
    def __init__(self, input_size):
        self.data_transform = {
            'train': Compose([Normalize()]),
            'valid': Compose([Normalize()]),
            'test': Compose([Normalize()])
        }

    def __call__(self, phase, img, anno_img):
        return self.data_transform[phase](img, anno_img)

class SatelliteDataset(data.Dataset):
    """ Custom dataset class for handling satellite image datasets """
    def __init__(self, img_list, anno_img_list, mask_list, dates, phase, transform):
        self.img_list = img_list
        self.anno_img_list = anno_img_list
        self.mask_list = mask_list
        self.dates = dates
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_img, mask_img = self.pull_item(index)
        return img.float(), anno_img.float(), mask_img.float(), self.dates[index]

    def pull_item(self, index):
        """ Loads image, annotation, and mask from file paths and applies transformations """
        img = np.load(self.img_list[index])
        anno_img = np.load(self.anno_img_list[index])
        mask_img = np.load(self.mask_list[index])
        img, anno_img = self.transform(self.phase, img, anno_img)
        mask_img = torch.from_numpy(mask_img)
        return img, anno_img, mask_img

### Data Processing Functions
def expand_dates(data_list, date_list):
    """ Expands and verifies date lists for dataset entries """
    expanded_dates = []
    for data_path in data_list:
        date = data_path.split('/')[-2]
        if date in date_list:
            expanded_dates.append(date)
        else:
            raise ValueError(f"Date {date} not found in date_list.")
    return expanded_dates

def augment_training_data(img_list, anno_list, mask_list, date_list, noise_mu=0.0, noise_sigma=0.01, max_augment=0.4):
    """ Performs data augmentation by adding Gaussian noise to training images """
    augmented_img_list = img_list.copy()
    augmented_anno_list = anno_list.copy()
    augmented_mask_list = mask_list.copy()
    augmented_date_list = date_list.copy()
    
    max_augment_count = int(len(img_list) * max_augment)
    augment_count = 0
    for img_path, anno_path, mask_path, date in zip(img_list, anno_list, mask_list, date_list):
        if augment_count >= max_augment_count:
            break
        img = np.load(img_path)
        anno = np.load(anno_path)
        mask = np.load(mask_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        anno_tensor = torch.from_numpy(anno).float()
        noise_transform = AddGaussianNoise(mu=noise_mu, sigma=noise_sigma)
        img_tensor_noise, anno_tensor_noise = noise_transform(img_tensor.clone(), anno_tensor.clone())
        np.save(img_path.replace('.npy', '_noise.npy'), img_tensor_noise.permute(1, 2, 0).numpy())
        np.save(anno_path.replace('.npy', '_noise.npy'), anno_tensor_noise.numpy().reshape(100, 160, 1))
        augmented_img_list.append(img_path.replace('.npy', '_noise.npy'))
        augmented_anno_list.append(anno_path.replace('.npy', '_noise.npy'))
        augmented_mask_list.append(mask_path)
        augmented_date_list.append(date)
        augment_count += 1
    return augmented_img_list, augmented_anno_list, augmented_mask_list, augmented_date_list
