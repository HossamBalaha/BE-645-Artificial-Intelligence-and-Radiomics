'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 13th, 2024
# Last Modification Date: Jun 13th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
from HMB_Summer_2025_Helpers import *

# Example file paths for medical imaging data.
caseImgPaths = [
  r"Data/Volume Slices/Volume Slice 65.bmp",
  r"Data/Volume Slices/Volume Slice 66.bmp",
  r"Data/Volume Slices/Volume Slice 67.bmp",
]
caseSegPaths = [
  r"Data/Segmentation Slices/Segmentation Slice 65.bmp",
  r"Data/Segmentation Slices/Segmentation Slice 66.bmp",
  r"Data/Segmentation Slices/Segmentation Slice 67.bmp",
]

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Set the connectivity type (6 or 26).
C = 6

# Calculate the Size-Zone Matrix.
szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix3D(
  volumeCropped,
  connectivity=C,
  isNorm=True,
  ignoreZeros=True,
)

# Compute features from Size-Zone Matrix.
features = CalculateGLSZMFeatures(szMatrix, volumeCropped, N, Z)

# Print the connectivity value.
print(f"At connectivity = {C}:")
for feature, value in features.items():
  # Print feature name with value rounded to 4 decimal places.
  print(f"{feature} : {np.round(value, 4)}")
