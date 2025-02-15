"""
InceptionModel: A convolutional neural network incorporating Inception-Residual blocks

This model is designed for satellite image processing. It takes an 88-channel input and progressively extracts features through convolutional layers with BatchNorm, ReLU activations, and Dropout. 
It integrates two Inception-Residual blocks to enhance feature extraction by applying multiple kernel sizes in parallel and concatenating the outputs. The final layer outputs a single-channel image prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from chat.CNNdataloader2 import make_datapath_list, DataTransform, SatelliteDataset, augment_training_data
import torch.nn.functional as F
import os
from matplotlib.colors import Normalize, ListedColormap
import pandas as pd
from datetime import datetime
from collections import defaultdict
import time
from torchvision import models

# Define the network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional layer block
        self.layer1 = nn.Sequential(
            nn.Conv2d(88, 96, kernel_size=5, padding=2),  # First convolutional layer
            nn.BatchNorm2d(96),  # Batch normalization
            nn.ReLU(),  # Activation function
            nn.Conv2d(96, 96, kernel_size=3, padding=1),  # Second convolutional layer
            nn.ReLU()
        )

        # First Inception-Residual Block
        self.inception1_1 = nn.Sequential(
            nn.Conv2d(88, 16, kernel_size=1),  # 1x1 convolution
            nn.Conv2d(16, 16, kernel_size=5, padding=2),  # 5x5 convolution
        )
        self.inception1_2 = nn.Sequential(
            nn.Conv2d(88, 16, kernel_size=1),  # 1x1 convolution
            nn.Conv2d(16, 16, kernel_size=7, padding=3),  # 7x7 convolution
        )
        self.inception1_3 = nn.Sequential(
            nn.Conv2d(88, 16, kernel_size=1),  # 1x1 convolution
        )

        # Second convolutional layer block
        self.layer2 = nn.Sequential(
            nn.Conv2d(144, 160, kernel_size=5, padding=2),  # Adjusted for concatenated inputs
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Second Inception-Residual Block
        self.inception2_1 = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=1, padding=0),  # 1x1 convolution
            nn.Conv2d(32, 32, kernel_size=5, padding=2),  # 5x5 convolution
        )
        self.inception2_2 = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=1, padding=0),  # 1x1 convolution
            nn.Conv2d(32, 32, kernel_size=7, padding=3),  # 7x7 convolution
        )
        self.inception2_3 = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=1, padding=0),  # 1x1 convolution
        )

        # Third convolutional layer block
        self.layer3 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=5, padding=2),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fourth convolutional layer block
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 160, kernel_size=5, padding=2),  # Input size adjusted for concatenated features
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fifth convolutional layer block
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final output layer
        self.final_layer = nn.Conv2d(32, 1, kernel_size=1, padding=0)  # Single-channel output

    def forward(self, x):
        # Apply first inception-residual block
        inception_out1_1 = self.inception1_1(x)
        inception_out1_2 = self.inception1_2(x)
        inception_out1_3 = self.inception1_3(x)
        x = self.layer1(x)
        x = torch.cat([x, inception_out1_1, inception_out1_2, inception_out1_3], dim=1)

        # Apply second convolutional block
        x = self.layer2(x)

        # Apply second inception-residual block
        inception_out2_1 = self.inception2_1(x)
        inception_out2_2 = self.inception2_2(x)
        inception_out2_3 = self.inception2_3(x)
        x = self.layer3(x)
        x = torch.cat([x, inception_out2_1, inception_out2_2, inception_out2_3], dim=1)

        # Apply subsequent layers
        x = self.layer4(x)
        x = self.layer5(x)

        # Apply final layer
        x = self.final_layer(x)
        return x

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
