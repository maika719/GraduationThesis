"""
LatLonGridNormalizer

This script reads a CSV file containing latitude, longitude, and date data,
and normalizes the latitude and longitude values based on a grid with a specified resolution.
It then saves the processed data back to a new CSV file.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

class Conversion:
    def __init__(self, df, lat_col, lon_col, date_col, grid_resolution):
        self.df = df
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col
        self.grid_resolution = grid_resolution

        # Get the range of latitude and longitude
        min_lat, max_lat = self.df[self.lat_col].min(), self.df[self.lat_col].max()
        min_lon, max_lon = self.df[self.lon_col].min(), self.df[self.lon_col].max()

        # Set up the grid resolution
        self.lat_grid = np.arange(min_lat, max_lat + grid_resolution, grid_resolution)
        self.lon_grid = np.arange(min_lon, max_lon + grid_resolution, grid_resolution)

        self.lat_size = len(self.lat_grid)
        self.lon_size = len(self.lon_grid)

        # Generate mesh grid
        self.grid_lon, self.grid_lat = np.meshgrid(self.lon_grid, self.lat_grid)

        # Group by date column
        if self.date_col in self.df.columns:
            self.grouped = self.df.groupby(self.date_col)
        else:
            raise ValueError(f"'{self.date_col}' column does not exist in the DataFrame.")

        # Normalize longitude and latitude values
        self.norm_lon = (self.grid_lon - self.grid_lon.min()) / (self.grid_lon.max() - self.grid_lon.min())
        self.norm_lat = (self.grid_lat - self.grid_lat.min()) / (self.grid_lat.max() - self.grid_lat.min())

    def save_to_csv(self, output_path):
        # Overwrite latitude and longitude columns with normalized values
        self.df[self.lon_col] = (self.df[self.lon_col] - self.grid_lon.min()) / (self.grid_lon.max() - self.grid_lon.min())
        self.df[self.lat_col] = (self.df[self.lat_col] - self.grid_lat.min()) / (self.grid_lat.max() - self.grid_lat.min())

        # Save data to CSV file with progress bar
        with tqdm(total=len(self.df), desc="Saving CSV") as pbar:
            self.df.to_csv(output_path, index=False)
            pbar.update(len(self.df))  # Update progress bar

        print(f"Data with normalized lat/lon saved to {output_path}")

# Define input CSV file path
csv_file_path = "/home/maika/anaconda3-lab/project/sotsuken/data/Satellite Data(2024_ZhangMaoquan)/asia data/chim_valid.csv"
data = pd.read_csv(csv_file_path)

# Initialize the Conversion class
Conversion = Conversion(
    df=data,
    lon_col="lon",  # Specify longitude column name
    lat_col="lat",  # Specify latitude column name
    date_col="date",  # Specify date column name
    grid_resolution=0.1  # Set grid resolution
)

# Define output CSV file path
output_csv_path = ""

# Save the normalized data along with the original data
Conversion.save_to_csv(output_csv_path)
