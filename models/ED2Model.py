"""
ED2Model: A Lightweight Encoder-Decoder Model for Satellite Image Processing

This model implements an encoder-decoder neural network designed for satellite image processing. 
It takes an 88-channel input and applies multiple convolutional layers with batch normalization and ReLU activation 
to extract hierarchical features. The encoder progressively downsamples and compresses information, while the decoder 
upsamples and reconstructs the output using skip connections to retain spatial information.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import make_datapath_list, DataTransform, SatelliteDataset, augment_training_data
import torch.nn.functional as F
import os
from matplotlib.colors import Normalize, ListedColormap
import pandas as pd
from datetime import datetime
from collections import defaultdict
import time
from torchvision import models

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Encoder block
        self.skip1 = nn.Sequential(
            nn.Conv2d(88, 36, kernel_size=1),  # Skip connection from input
            nn.BatchNorm2d(36)
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(88, 48, kernel_size=1),
            nn.BatchNorm2d(48), 
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48), 
            nn.ReLU(),            
            nn.Conv2d(48, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=1, stride=2)  # Skip connection with downsampling
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(100, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),            
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        # Decoder block
        self.dec2 = nn.Sequential(
            nn.Conv2d(356, 128, kernel_size=3, padding=1),  # Concatenating enc2_out with skip connection
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(164, 64, kernel_size=3, padding=1),  # Concatenating dec2_out with enc1_out
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 1, kernel_size=1)  # Final output layer
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc1_out = torch.cat([torch.relu(self.skip1(x)), torch.relu(enc1_out)], dim=1)
        enc2_out = self.enc2(enc1_out)
        enc2_out = torch.cat([torch.relu(enc2_out), torch.relu(self.skip2(enc1_out))], dim=1)

        # Decoder
        dec2_out = self.dec2(enc2_out)
        dec2_out = torch.cat([dec2_out, enc1_out], dim=1)

        dec1_out = self.dec1(dec2_out)        
        return dec1_out

def train_model(model, dataloaders_dict, optimizer, criterion_train, criterion_eval, num_epochs, device):
    """
    Train the model using the provided dataloaders and optimization settings.
    """
    print("Device in use:", device)
    model.to(device)

    # Enable cuDNN benchmark mode for performance optimization
    torch.backends.cudnn.benchmark = True

    # Initialize logging
    logs = []
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # Iteration counter
    iteration = 1

    for epoch in range(num_epochs):
        print('-------------')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-------------')

        t_epoch_start = time.time()

        # Loop over training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print('(train)')
                criterion = criterion_train  # Loss function for training
            else:
                model.eval()
                print('(val)')
                criterion = criterion_eval  # Loss function for validation

            running_loss = 0.0
            total_pixel_count = 0

            for input_data, GroundTruth, _, _ in dataloaders_dict[phase]:
                input_data = input_data.to(device)
                GroundTruth = GroundTruth.to(device)
                mask_data = (~torch.isnan(GroundTruth)).to(device)  # Mask for valid pixels

                # Zero gradients for training
                if phase == 'train':
                    optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_data)  # Model prediction

                    GroundTruth = GroundTruth.permute(0, 3, 1, 2)
                    mask_data = mask_data.permute(0, 3, 1, 2)

                    # Compute loss
                    loss = criterion(output[mask_data], GroundTruth[mask_data])

                    # Backpropagation and optimization step for training
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)  # Gradient clipping
                        optimizer.step()

                        if iteration % 10 == 0:  # Log every 10 iterations
                            print(f"Iteration {iteration} || Loss: {loss.item():.4f}")
                        iteration += 1

                    running_loss += loss.item()
                    total_pixel_count += mask_data.sum().item()

            # Compute loss for each phase
            if phase == 'train':
                epoch_train_loss = running_loss / total_pixel_count
            else:
                epoch_val_loss = torch.sqrt(torch.tensor(running_loss / total_pixel_count))

        scheduler.step(epoch_val_loss)  # Adjust learning rate based on validation loss

        # Log epoch results
        t_epoch_finish = time.time()
        print(f'epoch {epoch + 1} || Epoch_TRAIN_Loss:{epoch_train_loss:.4f} || Epoch_VAL_Loss:{epoch_val_loss:.4f}')
        print(f'timer: {t_epoch_finish - t_epoch_start:.4f} sec.')

        # Save logs
        logs.append({'epoch': epoch + 1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss})
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv", index=False)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = "weights2_2nn"  # Directory to save model weights
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'weights2_2nn/model_epoch_{epoch + 1}.pth')

    # Plot training and validation loss
    train_losses = [log['train_loss'] for log in logs]
    val_losses = [log['val_loss'] for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == "__main__":
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    rootpath = '/home/maika/anaconda3-lab/project/sotsuken/images/africa'  # Root path for data
    input_size = (100, 160)  # Image input size

    # Generate dataset file paths
    train_img_list, train_anno_list, train_mask_list, expanded_train_list, \
    val_img_list, val_anno_list, val_mask_list, expanded_val_list, \
    test_img_list, test_anno_list, test_mask_list, expanded_test_list = make_datapath_list(rootpath)

    # **Remove existing augmented data**
    for img_path in train_img_list:
        for suffix in ['_noise.npy', '_rot.npy']:  # File names of augmented data
            img_aug_path = img_path.replace('.npy', suffix)
            if os.path.exists(img_aug_path):
                os.remove(img_aug_path)

    # **Perform data augmentation**
    augmented_img_list, augmented_anno_list, augmented_mask_list, augmented_date_list = augment_training_data(
        train_img_list, train_anno_list, train_mask_list, expanded_train_list,
        noise_mu=0.0, noise_sigma=0.5, max_augment=0.0  # Removed `rotation_degrees`
    )

    # **Merge original and augmented data**
    train_img_list.extend(augmented_img_list)
    train_anno_list.extend(augmented_anno_list)
    train_mask_list.extend(augmented_mask_list)
    expanded_train_list.extend(augmented_date_list)  # Merge date list as well

    # **Apply data transformation**
    transformer = DataTransform(input_size)

    # **Create dataset**
    train_dataset = SatelliteDataset(train_img_list, train_anno_list, train_mask_list, expanded_train_list, phase='train', transform=transformer)
    val_dataset = SatelliteDataset(val_img_list, val_anno_list, val_mask_list, expanded_val_list, phase='valid', transform=transformer)

    # **Create dataloaders**
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=min(4, os.cpu_count() // 2), pin_memory=True)

    # **Define loss functions**
    criterion_train = nn.SmoothL1Loss(reduction='sum', beta=1.0)  # Training loss function
    criterion_eval = nn.MSELoss(reduction='sum')  # Evaluation loss function

    dataloaders_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    # **Train the model**
    train_model(model, dataloaders_dict, optimizer, criterion_train, criterion_eval, num_epochs=100, device=device)
