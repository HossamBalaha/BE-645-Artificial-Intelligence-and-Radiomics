'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 6th, 2024
# Last Modification Date: Jun 5th, 2025
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

# Set analysis angle to 0 degrees (horizontal direction).
theta = 0
# Convert angle to radians for trigonometric functions.
theta = np.radians(theta)

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Compute GLRLM with normalized probabilities and zero exclusion.
rlMatrix = CalculateGLRLM3DRunLengthMatrix(
  volumeCropped, theta,
  isNorm=True, ignoreZeros=True
)

# Extract texture features from computed GLRLM.
features = CalculateGLRLMFeatures(rlMatrix, volumeCropped)

# Display computed features with formatted output.
print(f"At angle {theta} degrees:")
for feature, value in features.items():
  # Print feature name with value rounded to 4 decimal places.
  print(f"{feature} : {np.round(value, 4)}")
