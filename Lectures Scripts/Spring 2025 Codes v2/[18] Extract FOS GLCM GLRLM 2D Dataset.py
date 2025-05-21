'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 29th, 2024
# Last Modification Date: Feb 9th, 2025
# Permissions and Citation: Refer to the README file.
'''

import os  # Import operating system interface module for file path handling.
import tqdm  # Import progress bar module for processing visualization.
import cv2  # Import OpenCV library for image processing.
import numpy as np  # Import numerical computing library.
import pandas as pd  # Import data analysis library.
from HMB_Helpers import *  # Import custom helper functions from local module.

# Set path to medical imaging dataset (user needs to modify this path).
# datasetPath = r"..\..\Datasets\COVID-19 Radiography Database"
# # Define list of medical condition categories to process.
# classes = [
#   "COVID",
#   "Normal",
#   "Viral Pneumonia",
#   "Lung_Opacity",
# ]
# STORAGE_DIR = r"Data/COVID-19 Radiography Database 2D"  # Directory to store extracted features.
# MASK_POSTFIX = ""  # Postfix for mask files.
# ALLOWED_REGIONS = 2  # Number of allowed regions in each image.
# ADD_ROI_IF_NOT_FOUND = False  # Flag to add ROI if not found.

# Set path to medical imaging dataset (user needs to modify this path).
datasetPath = r"..\..\Datasets\Dataset_BUSI_with_GT"
# Define list of medical condition categories to process.
classes = [
  "normal",
  "benign",
  "malignant",
]
STORAGE_DIR = r"Data/BUSI 2D"  # Directory to store extracted features.
MASK_POSTFIX = "_mask"  # Postfix for mask files.
ALLOWED_REGIONS = 1  # Number of allowed regions in each image.
ADD_ROI_IF_NOT_FOUND = True  # Flag to add ROI if not found.

# Configure image processing parameters and feature extraction settings.
# THETAS_DEGREE = [0, 90, 45, 135]  # Angles for matrix calculations.
# DISTANCES = [1, 2, 3]  # Pixel distances for co-occurrence matrix calculations.
THETAS_DEGREE = [0]  # Angles for matrix calculations.
DISTANCES = [1]  # Pixel distances for co-occurrence matrix calculations.
IGNORE_ZEROS = True  # Flag to exclude zero-valued pixels from calculations.
APPLY_NORMALIZATION = True  # Flag to enable data normalization.
# Maximum files to process per category (for testing).
# Set to a high value (e.g., 99999) to process all files in the dataset.
MAX_FILES_PER_CATEGORY = 9999
DO_RESIZE = False  # Flag to enable image resizing.
TARGET_SIZE = (256, 256)  # Target size for image resizing.
CONTOUR_AREA_THRESHOLD = 0  # Minimum area threshold for contour detection.

# Check if the angles list is empty or contains invalid values.
if (len(THETAS_DEGREE) == 0) or (not all(isinstance(x, int) for x in THETAS_DEGREE)):
  raise ValueError("[THETAS_DEGREE] list is empty or contains invalid values.")

# Check if the distances list is empty or contains invalid values.
if (len(DISTANCES) == 0) or (not all(isinstance(x, int) for x in DISTANCES)):
  raise ValueError("[DISTANCES] list is empty or contains invalid values.")

# Create the storage directory if it does not exist.
os.makedirs(STORAGE_DIR, exist_ok=True)

# Convert theta from degrees to radians for mathematical operations.
thetasRad = np.radians(THETAS_DEGREE)

# Initialize list to store feature extraction records.
records = []

# Main processing loop through each medical condition category.
for cls in classes:
  print(f"Processing class: {cls}.")
  # Construct full path to current class directory.
  clsPath = os.path.join(datasetPath, cls)
  # List all image files in class directory.
  files = os.listdir(clsPath + "/images")
  # Process files with progress bar visualization.
  for file in tqdm.tqdm(files[:MAX_FILES_PER_CATEGORY]):
    try:
      # Construct paths to image and segmentation mask files.
      caseImgPath = os.path.join(clsPath, "images", file)
      if (len(MASK_POSTFIX) > 0):
        ext = file.split(".")[-1]
        file = file.replace(f".{ext}", f"{MASK_POSTFIX}.{ext}")
      caseSegPath = os.path.join(clsPath, "masks", file)

      # Validate existence of required files before processing.
      if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
        raise FileNotFoundError("One or more files were not found. Please check the file paths.")

      # Load grayscale images from file paths.
      caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
      caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

      if (DO_RESIZE):
        # Resize images to target dimensions using cubic interpolation.
        caseImg = cv2.resize(caseImg, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        caseSeg = cv2.resize(caseSeg, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

      # Binarize segmentation mask by thresholding.
      caseSeg[caseSeg > 0] = 255

      if (ADD_ROI_IF_NOT_FOUND):
        # Check if the segmentation mask is empty.
        if (np.sum(caseSeg) == 0):
          # Get the shape of the input image.
          inputSize = caseImg.shape
          # Set the mask to be a rectangle in the middle.
          caseSeg = np.zeros(inputSize[:2], dtype=np.uint8)
          x = inputSize[1] // 4  # 1/4 of the width.
          y = inputSize[0] // 4  # 1/4 of the height.
          w = inputSize[1] // 2  # 1/2 of the width
          h = inputSize[0] // 2  # 1/2 of the height
          caseSeg[y:y + h, x:x + w] = 255

      # Extract regions of interest from medical images.
      regions = ExtractMultipleObjects(
        caseImg,  # Image to extract regions from.
        caseSeg,  # Segmentation mask for region extraction.
        cntAreaThreshold=CONTOUR_AREA_THRESHOLD,  # Minimum contour area threshold.
      )

      # Validate number of detected regions.
      if (len(regions) != ALLOWED_REGIONS):
        print(f"File: {file} contains {len(regions)} regions. Skipping...")
        continue

      # Initialize dictionary to store region features.
      regionsFeatures = {}

      # Process each extracted region for feature extraction.
      for i, cropped in enumerate(regions):
        # Calculate first-order statistical features.
        firstOrder = FirstOrderFeatures(
          cropped,  # Image to extract features from.
          isNorm=APPLY_NORMALIZATION,  # Flag to enable data normalization.
          ignoreZeros=IGNORE_ZEROS,  # Flag to exclude zero-valued pixels.
        )
        # Append the first-order features to the region-specific dictionary.
        regionsFeatures.update(
          {
            a + f"_R{i + 1}": b
            for a, b in firstOrder.items()
          }
        )

        for j, distance in enumerate(DISTANCES):
          for k, theta in enumerate(thetasRad):
            # Calculate Gray-Level Co-occurrence Matrix (GLCM).
            glcm = CalculateGLCMCooccuranceMatrix(
              cropped, distance, theta,
              isNorm=APPLY_NORMALIZATION,  # Flag to enable data normalization.
              ignoreZeros=IGNORE_ZEROS,  # Flag to exclude zero-valued pixels.
            )

            # Extract texture features from GLCM.
            glcmFeatures = CalculateGLCMFeatures(glcm)

            # Append GLCM features to the region-specific dictionary.
            regionsFeatures.update(
              {
                a + f"_R{i + 1}_D{j + 1}_T{k + 1}": b
                for a, b in glcmFeatures.items()
              }
            )

        for k, theta in enumerate(thetasRad):
          # Calculate Gray-Level Run-Length Matrix (GLRLM).
          rlMatrix = CalculateGLRLMRunLengthMatrix(
            cropped, theta,
            isNorm=APPLY_NORMALIZATION,  # Flag to enable data normalization.
            ignoreZeros=IGNORE_ZEROS,  # Flag to exclude zero-valued pixels.
          )

          # Extract features from Run-Length Matrix.
          rlFeatures = CalculateGLRLMFeatures(rlMatrix, cropped)

          # Append GLRLM features to the region-specific dictionary.
          regionsFeatures.update(
            {
              a + f"R_{i + 1}_T{k + 1}": b
              for a, b in rlFeatures.items()
            }
          )

      # Create final record with class label and all features.
      record = {
        **regionsFeatures,
        "Class": cls,
      }
      # Add complete record to main list.
      records.append(record)

    # Handle exceptions and continue processing other files.
    except Exception as e:
      print(f"Error in file: {file}.\nError is: {e}")
      continue

# Convert collected records to pandas DataFrame.
df = pd.DataFrame(records)
# Identify numeric columns (all except last 'Class' column).
numericCols = df.columns[:-1]
# Convert numeric columns to float32 for precision.
convertedData = df[numericCols].astype(np.float32)
# Find columns with only one unique value (constant columns).
cols2Drop = convertedData.columns[convertedData.nunique() == 1]
# Remove constant columns from DataFrame.
df = df.drop(columns=cols2Drop)

# Save processed DataFrame to CSV file.
df.to_csv(
  rf"{STORAGE_DIR}/Records_{MAX_FILES_PER_CATEGORY}_{THETAS_DEGREE}_"
  rf"{DISTANCES}_{IGNORE_ZEROS}_{APPLY_NORMALIZATION}.csv",
  index=False,
)
# Print confirmation and preview of final DataFrame.
print("Features are extracted successfully.")
print(df.head())
