"""
MODISAfricaValidationDataProcessor

This script processes CSV files containing MODIS satellite data for Africa (validation data),
converts them into images by mapping values to pixel grids,
and saves the resulting images as NumPy (.npy) files.
The script iterates through a specified date range and processes data accordingly,
specifically for validation data preparation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import calendar

# Define base paths for CSV files and result storage
csv_base_path = "/home/maika/anaconda3-lab/project/sotsuken/data/Satellite Data(2024_ZhangMaoquan)/africa data/valid/split_by_date"
result_base_folder = './images/africa/0130/valid/MODIS/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Function to load CSV data
def load_data(csv_file_path):
    print(csv_file_path + " is being loaded...")
    df = pd.read_csv(csv_file_path, header=0)
    print("Data loading completed!")
    return df

# Function to convert data into images and save as NumPy files
def create_images_and_save_npy(df, result_folder_path):
    # Create a folder for saving results
    os.makedirs(result_folder_path, exist_ok=True)
    print(result_folder_path + " folder has been created!")

    # Set image resolution
    img_width, img_height = 160, 100

    # Initialize an empty array to store image data
    num_channels = 1  # Set number of channels to 1
    image_data = np.full((img_height, img_width, num_channels), np.nan)

    # Create a mask array to track assigned pixels
    assigned_mask = np.zeros((img_height, img_width), dtype=bool)
    
    # Assign values to pixels
    for i in range(len(df)):
        x = int(round(df['lon'].iloc[i] * (img_width - 1)))  # Round to increase precision
        y = int(round(df['lat'].iloc[i] * (img_height - 1)))  # Round to increase precision
        
        # Assign non-NaN values only
        modis_value = df['MODIS'].iloc[i]
        if not np.isnan(modis_value):
            image_data[y, x, 0] = modis_value
            assigned_mask[y, x] = True
        else:
            assigned_mask[y, x] = True

    # Remove unassigned rows and columns using the mask
    assigned_rows = np.any(assigned_mask, axis=1)
    assigned_cols = np.any(assigned_mask, axis=0)

    # Apply row and column filtering
    image_data = image_data[assigned_rows, :, :]
    image_data = image_data[:, assigned_cols, :]

    # Save as NumPy (.npy) file
    np.save(f"{result_folder_path}/data_image.npy", image_data)
    print(f"MODIS channel data saved in npy format -> {result_folder_path}/data_image.npy")

# Function to process files within a specified date range
def process_files_in_date_range(start_month, end_month, start_day):
    for month in range(start_month, end_month + 1):
        # Get the last day of the month
        _, last_day = calendar.monthrange(2021, month)

        # Iterate through the date range
        for day in range(start_day, last_day + 1):
            date_str = f"{month:02}{day:02}"
            csv_file_path = os.path.join(csv_base_path, f"data_2021{date_str}.csv")

            if not os.path.exists(csv_file_path):
                print(f"File does not exist: {csv_file_path}")
                continue

            # Create a folder for storing results
            result_folder_path = os.path.join(result_base_folder, f"2021{date_str}")

            # Process the data
            df = load_data(csv_file_path)
            create_images_and_save_npy(df, result_folder_path)

# Main function
def main():
    # Process CSV files from January 21 to the end of December
    process_files_in_date_range(1, 12, 21)

# Execute script
if __name__ == "__main__":
    main()
