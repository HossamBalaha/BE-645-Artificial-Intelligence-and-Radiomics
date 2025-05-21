'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 3rd, 2025
# Last Modification Date: Mar 3rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file and directory operations.
import warnings  # For warning control.
import tqdm  # For progress bar functionality.
import time  # For time-related functions.
import json  # For JSON file operations.
import numpy as np  # For numerical operations and array manipulations.
import pandas as pd  # For data manipulation and analysis.
from HMB_Helpers import *
from medmnist import AdrenalMNIST3D

# Ignore warnings for cleaner output.
warnings.filterwarnings("ignore")


def MedMnistLoaderIterator(split="train"):
  # Load the dataset for the specified split and download if necessary.
  dataset = AdrenalMNIST3D(split=split, download=True)
  # Print dataset information for reference.
  for info in dataset.info:
    print(f"{info}: {dataset.info[info]}")
  # Extract the label dictionary for mapping indices to categories.
  categoriesDict = dataset.info["label"]
  # Extract the total number of records for the specified split.
  totalRecords = dataset.info["n_samples"][split]
  # Iterate through the dataset with a progress bar.
  for i, record in tqdm.tqdm(
    enumerate(dataset),
    total=totalRecords,
    desc=f"Processing {split} dataset",
    unit="Volume",
  ):
    # Extract the volume data from the record.
    volume = record[0][0]
    # Normalize the volume to the range [0, 1].
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    # Scale the normalized volume to the range [0, 255].
    volume = volume * 255
    # Convert the volume to an unsigned 8-bit integer format.
    volume = volume.astype(np.uint8)
    # Extract the category index from the record.
    categoryIdx = record[1][0]
    # Map the category index to its corresponding label.
    category = categoriesDict[str(categoryIdx)]
    # Yield the preprocessed volume, category, and index.
    yield volume, category, i


# Configuration flags and parameters for feature extraction.
APPLY_NORMALIZATION = True  # Flag to apply normalization during feature extraction.
IGNORE_ZEROS = True  # Flag to ignore zero values during feature extraction.
GLCM_THETAS = [0]  # List of angles (in degrees) for GLCM feature extraction.
GLCM_DISTANCES = [1]  # List of distances for GLCM feature extraction.
GLRLM_THETAS = [0]  # List of angles (in degrees) for GLRLM feature extraction.
SPLIT = "train"  # Specify the dataset split (e.g., "train", "test").

# Generate a timestamp and create a storage directory for the extracted features.
CURRENT_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp for unique directory naming.
STORAGE_DIR = rf"Data/3D_Features_{CURRENT_TIMESTAMP}"  # Define the storage directory name.
os.makedirs(STORAGE_DIR, exist_ok=True)  # Create the storage directory if it does not already exist.

# Initialize the dataset iterator for the specified split.
iterator = MedMnistLoaderIterator(split=SPLIT)

# Lists to store extracted features for each type of feature set.
fosList = []  # List to store First-Order Statistics (FOS) features.
glcmList = []  # List to store Gray-Level Co-occurrence Matrix (GLCM) features.
glrlmList = []  # List to store Gray-Level Run-Length Matrix (GLRLM) features.
shapeList = []  # List to store shape-based features.

# Iterate through the dataset and extract features for each volume.
for volumeCropped, cat, file in iterator:
  # Extract First-Order Statistics (FOS) features.
  fos = FirstOrderFeatures3D(
    volumeCropped,
    isNorm=APPLY_NORMALIZATION,
    ignoreZeros=IGNORE_ZEROS,
  )
  # Add the category label to the FOS features.
  fos["Category"] = cat
  # Add the file index to the FOS features.
  fos["File"] = file
  # Append the FOS features to the list.
  fosList.append(fos)

  # Extract Gray-Level Co-occurrence Matrix (GLCM) features.
  glcmRecord = {}
  for d in GLCM_DISTANCES:
    for theta in GLCM_THETAS:
      # Calculate the GLCM co-occurrence matrix.
      coMatrix = CalculateGLCM3DCooccuranceMatrix(
        volumeCropped,
        d,
        theta,
        isSymmetric=False,
        isNorm=True,
        ignoreZeros=True,
      )
      # Calculate GLCM features from the matrix.
      glcmFeatures = CalculateGLCMFeatures3D(coMatrix)
      # Rename the keys to include the distance and angle for uniqueness.
      newDict = {}
      for key in glcmFeatures.keys():
        newKey = f"{key}_{d}_{theta}"
        newDict[newKey] = glcmFeatures[key]
      # Update the GLCM record with the renamed features.
      glcmRecord.update(newDict)
  # Add the category label to the GLCM features.
  glcmRecord["Category"] = cat
  # Add the file index to the GLCM features.
  glcmRecord["File"] = file
  # Append the GLCM features to the list.
  glcmList.append(glcmRecord)

  # Extract Gray-Level Run-Length Matrix (GLRLM) features.
  glrlmRecord = {}
  for theta in GLRLM_THETAS:
    # Calculate the GLRLM run-length matrix.
    rlMatrix = CalculateGLRLM3DRunLengthMatrix(
      volumeCropped,
      theta,
      isNorm=APPLY_NORMALIZATION,
      ignoreZeros=IGNORE_ZEROS,
    )
    # Calculate GLRLM features from the matrix.
    glrlmFeatures = CalculateGLRLMFeatures3D(rlMatrix, volumeCropped)
    # Rename the keys to include the angle for uniqueness.
    newDict = {}
    for key in glrlmFeatures.keys():
      newKey = f"{key}_{theta}"
      newDict[newKey] = glrlmFeatures[key]
    # Update the GLRLM record with the renamed features.
    glrlmRecord.update(newDict)
  # Add the category label to the GLRLM features.
  glrlmRecord["Category"] = cat
  # Add the file index to the GLRLM features.
  glrlmRecord["File"] = file
  # Append the GLRLM features to the list.
  glrlmList.append(glrlmRecord)

  # Extract shape-based features.
  shape = ShapeFeatures3D(volumeCropped)  # Calculate shape-based features.
  # Add the category label to the shape features.
  shape["Category"] = cat
  # Add the file index to the shape features.
  shape["File"] = file
  # Append the shape features to the list.
  shapeList.append(shape)

# Convert the lists of dictionaries to Pandas DataFrames for easier manipulation and analysis.
fosDF = pd.DataFrame(fosList)
glcmDF = pd.DataFrame(glcmList)
glrlmDF = pd.DataFrame(glrlmList)
shapeDF = pd.DataFrame(shapeList)

# Save the DataFrames to CSV files for persistence.
fosDF.to_csv(os.path.join(STORAGE_DIR, f"FOS.csv"), index=False)
glcmDF.to_csv(os.path.join(STORAGE_DIR, f"GLCM.csv"), index=False)
glrlmDF.to_csv(os.path.join(STORAGE_DIR, f"GLRLM.csv"), index=False)
shapeDF.to_csv(os.path.join(STORAGE_DIR, f"Shape.csv"), index=False)

# Merge all DataFrames into a single DataFrame for comprehensive analysis.
allTogether = pd.merge(fosDF, glcmDF, on=["Category", "File"])
allTogether = pd.merge(allTogether, glrlmDF, on=["Category", "File"])
allTogether = pd.merge(allTogether, shapeDF, on=["Category", "File"])
# Reorder columns to have "Category" at the end.
columns = allTogether.columns
columns.remove("Category")
columns.append("Category")
allTogether.columns = columns
allTogether.to_csv(os.path.join(STORAGE_DIR, f"Merged.csv"), index=False)

# Save filtered DataFrames to CSV files by removing the "File" column.
fosDF.drop(columns=["File"], inplace=True)
glcmDF.drop(columns=["File"], inplace=True)
glrlmDF.drop(columns=["File"], inplace=True)
shapeDF.drop(columns=["File"], inplace=True)
allTogether.drop(columns=["File"], inplace=True)

fosDF.to_csv(os.path.join(STORAGE_DIR, f"FOS_Filtered.csv"), index=False)
glcmDF.to_csv(os.path.join(STORAGE_DIR, f"GLCM_Filtered.csv"), index=False)
glrlmDF.to_csv(os.path.join(STORAGE_DIR, f"GLRLM_Filtered.csv"), index=False)
shapeDF.to_csv(os.path.join(STORAGE_DIR, f"Shape_Filtered.csv"), index=False)
allTogether.to_csv(os.path.join(STORAGE_DIR, f"Merged_Filtered.csv"), index=False)

# Save the configuration settings to a JSON file for reproducibility.
configs = {
  "APPLY_NORMALIZATION": APPLY_NORMALIZATION,
  "IGNORE_ZEROS"       : IGNORE_ZEROS,
  "GLCM_THETAS"        : GLCM_THETAS,
  "GLCM_DISTANCES"     : GLCM_DISTANCES,
  "GLRLM_THETAS"       : GLRLM_THETAS,
  "CURRENT_TIMESTAMP"  : CURRENT_TIMESTAMP,
  "STORAGE_DIR"        : STORAGE_DIR,
  "SPLIT"              : SPLIT,
}
# Write the configurations to a JSON file with indentation for readability.
with open(os.path.join(STORAGE_DIR, "config.json"), "w") as configFile:
  json.dump(configs, configFile, indent=4)
