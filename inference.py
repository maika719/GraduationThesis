"""
inference.py: A script for performing inference using a trained neural network on satellite image data.

This script loads a trained model, processes test images, and generates predictions. 
It saves the results as images and CSV files, and computes error metrics such as RMSE. 
It also creates boxplots for visualizing prediction errors by MODIS data range.

Key Features:
- Loads a trained model from a specified path
- Performs inference on test satellite images
- Saves input, ground truth, and predicted images
- Computes and saves RMSE and error metrics
- Generates daily and monthly error reports
- Saves difference images between ground truth and predictions
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import make_datapath_list, DataTransform, SatelliteDataset
from InceptionModel import Net #Change the model as needed
from torch.utils.data import DataLoader
from matplotlib.colors import Normalize, ListedColormap
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import mean_squared_error
from matplotlib.colors import CenteredNorm, ListedColormap
import math
import csv

def inference(model, dataloader, device, output_dir):
    """ Runs inference on the test dataset and saves predictions and error metrics. """
    print("Starting inference...")
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)  # Create output directory

    all_differences = defaultdict(list)  # Dictionary to store prediction errors
    modis_values = defaultdict(list)  # Dictionary to store MODIS values
    total_rmse = []  # List to store RMSE values for each batch

    with torch.no_grad():
        for batch_idx, (input_data, GroundTruth, _, dates) in enumerate(dataloader):
            input_data = input_data.to(device)
            GroundTruth = GroundTruth.to(device)
            mask_data = (~torch.isnan(GroundTruth)).to(device)

            # Model inference
            output = model(input_data)
            
            # Adjust dimensions
            GroundTruth = GroundTruth.permute(0, 3, 1, 2)
            mask_data = mask_data.permute(0, 3, 1, 2)
            output_normalized = output.clone()

            # Compute differences per date
            for i, date in enumerate(dates):
                date = date.strip()
                diff_for_sample = (GroundTruth[i] - output_normalized[i])[mask_data[i]].cpu().numpy()
                all_differences[date].extend(diff_for_sample.flatten())
                
                # Store MODIS values, excluding NaNs
                modis_for_sample = GroundTruth[i].cpu().numpy().flatten()
                valid_modis_values = modis_for_sample[~np.isnan(modis_for_sample)]
                modis_values[date].extend(valid_modis_values)

            # Compute RMSE for batch
            true_values = GroundTruth[mask_data].cpu().numpy()
            pred_values = output_normalized[mask_data].cpu().numpy()
            batch_rmse = math.sqrt(mean_squared_error(true_values, pred_values))
            total_rmse.append(batch_rmse)

            # Save images and results
            save_images_and_csv(input_data, GroundTruth, output_normalized, dates, output_dir)

    # Compute overall RMSE
    overall_rmse = sum(total_rmse) / len(total_rmse)
    print(f"\nOverall RMSE: {overall_rmse:.4f}")

    # Save boxplot data
    save_monthly_boxplot_data(all_differences, modis_values, "output/ED2")
    print(f"Inference complete. Results saved in {output_dir}.")"}

def save_images_and_csv(input_data, GroundTruth, predicted, dates, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output folder

    # Dictionary to store all differences by date
    all_differences_pred = {}
    all_differences_input = {}

    for i, date in enumerate(dates):
        date = date.strip()  # Remove extra spaces from date string
        date_folder = os.path.join(output_dir, date)
        os.makedirs(date_folder, exist_ok=True)  # Create subfolder for each date

        # Save input image
        input_image = input_data[i, 0, :, :].cpu().numpy()
        plt.imshow(input_image, cmap='YlOrBr')
        plt.title(f"Input Data ({date})")
        plt.colorbar()
        plt.savefig(os.path.join(date_folder, f"input_data_{date}.png"))
        plt.close()

        # Save GroundTruth image
        GroundTruth_image = GroundTruth[i].squeeze(0).cpu().numpy()
        plt.imshow(GroundTruth_image, cmap='YlOrBr')
        plt.title(f"GroundTruth ({date})")
        plt.colorbar()
        plt.savefig(os.path.join(date_folder, f"GroundTruth_{date}.png"))
        plt.close()

        # Save predicted image
        predicted_image = predicted[i].squeeze(0).cpu().numpy()
        plt.imshow(predicted_image, cmap='YlOrBr')
        plt.title(f"Predicted AOD ({date})")
        plt.colorbar()
        plt.savefig(os.path.join(date_folder, f"predicted_{date}.png"))
        plt.close()

        # Compute difference images (GroundTruth - Predicted)
        difference_image_pred = GroundTruth_image - predicted_image

        # Compute difference images (GroundTruth - Input)
        difference_image_input = GroundTruth_image - input_image

        # Compute statistics ignoring NaNs (GT - Prediction)
        rmse_pred = np.sqrt(np.nanmean(difference_image_pred ** 2))
        mean_pred = np.nanmean(difference_image_pred)
        std_pred = np.nanstd(difference_image_pred)

        # Compute statistics ignoring NaNs (GT - Input)
        rmse_input = np.sqrt(np.nanmean(difference_image_input ** 2))
        mean_input = np.nanmean(difference_image_input)
        std_input = np.nanstd(difference_image_input)

        # Store difference data for boxplot
        all_differences_pred[date] = difference_image_pred[~np.isnan(difference_image_pred)].flatten()
        all_differences_input[date] = difference_image_input[~np.isnan(difference_image_input)].flatten()

        # Set custom colormap
        cmap = plt.cm.seismic  # Blue to white to red
        cmap_with_nan = cmap(np.arange(cmap.N))  # Retrieve original colormap array
        cmap_with_nan[:, -1] = 1.0  # Set alpha values (fully opaque)
        cmap_with_nan = ListedColormap(cmap_with_nan)
        cmap_with_nan.set_bad(color='#FFFF00')  # Set NaN areas to yellow

        # Handle NaNs (GroundTruth - Predicted)
        nan_mask_pred = np.isnan(difference_image_pred)
        difference_image_pred[nan_mask_pred] = np.nan

        # Handle NaNs (GroundTruth - Input)
        nan_mask_input = np.isnan(difference_image_input)
        difference_image_input[nan_mask_input] = np.nan

        # Get max absolute value ignoring NaNs
        max_abs_pred = max(abs(np.nanmin(difference_image_pred)), abs(np.nanmax(difference_image_pred)))
        max_abs_input = max(abs(np.nanmin(difference_image_input)), abs(np.nanmax(difference_image_input)))
        max_abs = max(max_abs_pred, max_abs_input)

        # Plot and save difference image (GroundTruth - Predicted)
        plt.figure(figsize=(8, 6))
        plt.imshow(difference_image_pred, cmap=cmap_with_nan, vmin=-max_abs, vmax=max_abs)
        plt.colorbar(label="Difference (GroundTruth - Predicted)")
        plt.title(f"Difference (GT - Pred) | RMSE: {rmse_pred:.4f} | Mean: {mean_pred:.4f} | Std: {std_pred:.4f}",fontsize=10)
        plt.savefig(os.path.join(date_folder, f"difference_with_nan_{date}.png"))
        plt.close()

        # Plot and save difference image (GroundTruth - Input)
        plt.figure(figsize=(8, 6))
        plt.imshow(difference_image_input, cmap=cmap_with_nan, vmin=-max_abs, vmax=max_abs)
        plt.colorbar(label="Difference (GroundTruth - Input)")
        plt.title(f"Difference (GT - Input) | RMSE: {rmse_input:.4f} | Mean: {mean_input:.4f} | Std: {std_input:.4f}",fontsize=10)
        plt.savefig(os.path.join(date_folder, f"difference_input_{date}.png"))
        plt.close()

        # Save data in CSV format
        save_to_csv(GroundTruth_image, predicted_image, date, os.path.join(date_folder, f"predicted_{date}.csv"))

    # Generate boxplots for differences
    save_difference_boxplots(all_differences_pred, all_differences_input, output_dir)

def save_difference_boxplots(all_differences_pred, all_differences_input, output_dir):
    """
    Save boxplots for the difference data (GT - Prediction, GT - Input) for each date.
    """

    # Sort the dates in chronological order
    sorted_dates = sorted(all_differences_pred.keys())

    for date in sorted_dates:
        pred_differences = all_differences_pred[date]
        input_differences = all_differences_input[date]

        # Create a folder for each date
        date_folder = os.path.join(output_dir, date)
        os.makedirs(date_folder, exist_ok=True)

        # Create a boxplot for GT - Prediction & GT - Input
        plt.figure(figsize=(8, 6))

        # Draw boxplot
        box = plt.boxplot(
            [pred_differences, input_differences], 
            vert=True, 
            patch_artist=True,  # Fill the boxes
            widths=0.5, 
            medianprops=dict(color="black"),  # Color of the median line
            flierprops=dict(marker='o', markersize=3, linestyle='none', markeredgecolor='black')
        )

        # Set box colors
        for patch, color in zip(box['boxes'], ["red", "blue"]):
            patch.set_facecolor(color)

        # Set labels
        plt.title(f"Difference BoxPlot ({date})", fontsize=16)  # Increase title font size
        plt.ylabel("Error", fontsize=14)  # Increase y-axis label font size
        plt.xlabel("Category", fontsize=14)  # Increase x-axis label font size
        plt.xticks([1, 2], ["GT - Predicted", "GT - CHIMERE"], fontsize=14)  # Modify x-axis labels

        # Add a horizontal red line at y=0
        plt.axhline(y=0, color="red", linestyle="--", linewidth=2)

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(date_folder, "boxplot_difference_combined.png"))
        plt.close()

def save_to_csv(GroundTruth, Prediction, date, output_file):
    """
    Save GroundTruth and Prediction data to a CSV file with pixel coordinates.
    """
    height, width = GroundTruth.shape
    data = {
        "x": [],  # X-coordinate of the pixel
        "y": [],  # Y-coordinate of the pixel
        "date": [],  # Date corresponding to the data
        "MODIS": [],  # GroundTruth values (MODIS data)
        "Prediction": []  # Predicted values from the model
    }

    # Add data for each pixel
    for y in range(height):
        for x in range(width):
            data["x"].append(x)
            data["y"].append(y)
            data["date"].append(date)
            data["MODIS"].append(GroundTruth[y, x])
            data["Prediction"].append(Prediction[y, x])

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def save_monthly_boxplot_data(all_differences, modis_values, output_dir):
    """
    Save daily error data and monthly RMSE data in CSV files based on MODIS value ranges.
    """
    modis_ranges = {
        "0.00-0.80": (0.00, 0.80),  # Range 0.00 to 0.80
        "0.80-3.00": (0.80, 3.00),  # Range 0.80 to 3.00
        "3.00-5.00": (3.00, 5.00)   # Range 3.00 to 5.00
    }

    # Data storage
    daily_error_records = []  # List to store daily error data
    monthly_rmse_records = defaultdict(list)  # Dictionary to store monthly RMSE data

    for date, differences in all_differences.items():
        try:
            date_obj = datetime.strptime(date, "%Y%m%d")  # Convert date string to datetime object
            month_key = date_obj.strftime("%Y%m")  # Extract year and month

            for range_key, (modis_min, modis_max) in modis_ranges.items():
                # Filter errors based on MODIS value range
                filtered_errors = [
                    diff for diff, modis in zip(differences, modis_values[date]) 
                    if modis_min <= modis < modis_max
                ]

                if filtered_errors:
                    # Store daily error data
                    for error in filtered_errors:
                        daily_error_records.append([date, range_key, error])

                    # Compute RMSE and add to monthly list
                    rmse = math.sqrt(mean_squared_error([0] * len(filtered_errors), filtered_errors))
                    monthly_rmse_records[month_key].append((range_key, rmse))

        except ValueError:
            print(f"Invalid date format: {date}")  # Handle invalid date format
            continue

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save daily error data to CSV
    daily_error_csv_path = os.path.join(output_dir, "daily_error_data.csv")
    with open(daily_error_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "MODIS Range", "Error"])
        writer.writerows(daily_error_records)

    # Save monthly RMSE data to CSV
    monthly_rmse_csv_path = os.path.join(output_dir, "monthly_rmse_data.csv")
    with open(monthly_rmse_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Month", "MODIS Range", "RMSE"])
        for month, values in monthly_rmse_records.items():
            for range_key, rmse in values:
                writer.writerow([month, range_key, rmse])

    print(f"CSV files saved:\n  {daily_error_csv_path}\n  {monthly_rmse_csv_path}")


if __name__ == "__main__":
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Net()

    # Load trained model
    weights_path = "weights2_2/model_epoch_90.pth"  # Path to trained model
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"Model {weights_path} loaded successfully.")

    rootpath = ''  # Root path for data
    input_size = (100, 160)  # Image input size

    # Generate test dataset file paths
    _, _, _, _, _, _, _, _, \
    test_img_list, test_anno_list, test_mask_list, expanded_test_list = make_datapath_list(rootpath)

    # Data transformation
    transformer = DataTransform(input_size)

    # Create test dataset
    test_dataset = SatelliteDataset(
        test_img_list, test_anno_list, test_mask_list, expanded_test_list, phase='test', transform=transformer
    )
    
    # Create test dataloader (no shuffling for evaluation)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Run inference
    output_dir = "inference_results"  # Directory to save inference results
    inference(model, test_dataloader, device, output_dir)
