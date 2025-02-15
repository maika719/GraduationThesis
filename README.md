# Project Overview

This repository provides a pipeline for processing and analyzing satellite data, including preprocessing, training, and inference using deep learning models.

## Execution Workflow

The execution follows these steps:

### 1️⃣ Latitude and Longitude Normalization
Run `LatLonGridNormalizer.py` to normalize latitude and longitude values.
```bash
python data_processing/LatLonGridNormalizer.py
```

### 2️⃣ CSV Splitting by Date
Use `DateWiseCSVSplitter.py` to split CSV files into separate files based on date.
```bash
python data_processing/DateWiseCSVSplitter.py
```

### 3️⃣ Convert Data to Numpy Format
Convert raw data into numpy (.npy) format using the following scripts:
- `AfricaTrainingDataProcessor.py` (Processes training data for Africa)
- `AfricaValidationDataProcessor.py` (Processes validation data for Africa)
- `MODISAfricaTrainingDataProcessor.py` (Processes MODIS training data)
- `MODISAfricaValidationDataProcessor.py` (Processes MODIS validation data)

Run these scripts as needed to generate preprocessed `.npy` files.
```bash
python data_processing/AfricaTrainingDataProcessor.py
python data_processing/AfricaValidationDataProcessor.py
python data_processing/MODISAfricaTrainingDataProcessor.py
python data_processing/MODISAfricaValidationDataProcessor.py
```

### 4️⃣ Train the Model
Train a deep learning model using one of the scripts in the `models/` directory.
For example, to train using `ED2Model.py`:
```bash
python models/ED2Model.py
```
Modify hyperparameters and configurations as needed within the script.

### 5️⃣ Perform Inference
Run `inference.py` to perform inference using the trained model.
```bash
python inference/inference.py
```
Results will be saved in the `inference_results/` directory.

## Requirements
Make sure to install the required dependencies before running the scripts:
```bash
pip install -r requirements.txt
```

## Directory Structure
```
/project-root
│── /models                  # Model definitions
│── /data_processing         # Data preprocessing scripts
│── /dataloaders             # DataLoader scripts
│── /inference               # Inference scripts
│── main.py                  # Main execution script
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── .gitignore               # Files to exclude from Git tracking
```

## Notes
- **Ensure that data files are correctly formatted before running the scripts.**
- **Modify model configurations as needed to optimize performance.**
- **Inference results will be saved automatically in `inference_results/`.**

