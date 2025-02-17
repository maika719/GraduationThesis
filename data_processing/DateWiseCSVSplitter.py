"""
DateWiseCSVSplitter

This script reads a CSV file containing satellite data, groups the data by the 'date' column,
and saves each date's data into a separate CSV file within a specified output directory.
"""

import pandas as pd
import os
from tqdm import tqdm

# Define file paths and parameters
csv_file_path = ""
output_dir = ""

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Convert 'date' column to string to ensure proper grouping
# Since the 'date' column is already an integer, converting it to a string avoids unintended formatting issues
data['date'] = data['date'].astype(str)

# Group data by the 'date' column
grouped = data.groupby('date')

# Iterate over each group and save it as a separate CSV file
# tqdm is used to display a progress bar
for date, group in tqdm(grouped, desc="Saving date-wise CSVs"):
    # Construct the output file path
    output_path = os.path.join(output_dir, f"data_{date}.csv")
    
    # Save each date's data as a new CSV file
    group.to_csv(output_path, index=False)

print(f"CSV files have been saved in {output_dir}")
