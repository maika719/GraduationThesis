"""
AfricaValidationDataProcessor

This script processes CSV files containing satellite data for Africa (validation data),
converts them into images by mapping values to pixel grids,
and saves the resulting images as NumPy (.npy) files.
The script iterates through a specified date range and processes data accordingly,
without applying normalization to the input features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import calendar

# Base folder path settings
csv_base_path = "/home/maika/anaconda3-lab/project/sotsuken/data/Satellite Data(2024_ZhangMaoquan)/africa data/valid/split_by_date/"
result_base_folder = './images/africa/sonomama/valid/' + datetime.now().strftime("ok%Y-%m-%d_%H-%M-%S")

# Function to load data
def load_data(csv_file_path):
    print(csv_file_path + " is being loaded...")
    df = pd.read_csv(csv_file_path, header=0)
    print("Data loading completed!")
    return df

# Function to retain raw feature values without normalization
def normalize_variables(df):
    # Exclude 'lon', 'lat', 'date', and 'MODIS' columns
    columns_to_normalize = df.drop(columns=['lon', 'lat', 'date', 'MODIS'])
    
    normalized_df = pd.DataFrame()  # Dataframe to retain original values
    
    # Retain feature values without applying normalization
    for col in columns_to_normalize.columns:
        normalized_df[col] = columns_to_normalize[col]
    
    print("Feature processing (no normalization) completed!")
    return normalized_df, columns_to_normalize.columns

# Function to save data in npy while converting data to images
def create_images_and_save_npy(df, normalized_data, channel_names, result_folder_path):
    # Create a folder for saving results
    os.makedirs(result_folder_path, exist_ok=True)
    print(result_folder_path + " folder has been created!")

    # Set image resolution
    img_width, img_height = 160, 100

    # Initialize an empty array to store image data
    num_channels = normalized_data.shape[1]
    image_data = np.full((img_height, img_width, num_channels), np.nan)
   
    # Initialize mask array to track assignments
    assigned_mask = np.zeros((img_height, img_width), dtype=bool)
    
    # Assign values to pixels
    for i in range(len(df)):
        x = int(round(df['lon'].iloc[i] * (img_width - 1)))  # Round for precision
        y = int(round(df['lat'].iloc[i] * (img_height - 1)))  # Round for precision

        # Assign values only if they are not NaN
        if not np.any(np.isnan(normalized_data.iloc[i].to_numpy())):
            image_data[y, x, :] = normalized_data.iloc[i].to_numpy()
            assigned_mask[y, x] = True
        else: 
            assigned_mask[y, x] = True

    # Remove rows and columns where no assignments were made
    assigned_rows = np.any(assigned_mask, axis=1)
    assigned_cols = np.any(assigned_mask, axis=0)

    # Apply mask to remove unassigned rows and columns
    image_data = image_data[assigned_rows, :, :]
    image_data = image_data[:, assigned_cols, :]

    # Save as NumPy file
    np.save(f"{result_folder_path}/data_image.npy", image_data)
    print(f"All channel data saved in npy format -> {result_folder_path}/data_image.npy")

# Function to process files in date range
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

            # Create folder
            result_folder_path = os.path.join(result_base_folder, f"2021{date_str}")

            # Process data
            df = load_data(csv_file_path)
            normalized_data, channel_names = normalize_variables(df)
            create_images_and_save_npy(df, normalized_data, channel_names, result_folder_path)

# Main function
def main():
    # Process CSV files from January 21 to the end of December
    process_files_in_date_range(1, 12, 21)

# Execute script
if __name__ == "__main__":
    main()
