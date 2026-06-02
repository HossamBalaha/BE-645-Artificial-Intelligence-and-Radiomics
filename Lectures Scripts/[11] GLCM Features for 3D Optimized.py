'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
from HMB_Spring_2026_Helpers import (
  ReadVolume, CalculateGLCMCooccuranceMatrix3D,
  CalculateGLCMFeaturesOptimized
)

# Define the paths to the volume slices and segmentation masks.
caseImgPaths = [
  r"Data/Volume Slices/Volume Slice 65.bmp",  # Path to the first slice.
  r"Data/Volume Slices/Volume Slice 66.bmp",  # Path to the second slice.
  r"Data/Volume Slices/Volume Slice 67.bmp",  # Path to the third slice.
]
caseSegPaths = [
  r"Data/Segmentation Slices/Segmentation Slice 65.bmp",  # Path to the first segmentation mask.
  r"Data/Segmentation Slices/Segmentation Slice 66.bmp",  # Path to the second segmentation mask.
  r"Data/Segmentation Slices/Segmentation Slice 67.bmp",  # Path to the third segmentation mask.
]

# Define parameters for the GLCM calculation.
d = 1  # Distance between voxel pairs.
theta = 0  # Angle (in degrees) for the direction of voxel pairs.
theta = np.radians(theta)  # Convert theta to radians.
# Keep it False unless you are sure that the GLCM can be transposed.
isSymmetric = False  # Whether to make the GLCM symmetric.
isNorm = True  # Whether to normalize the GLCM.
ignoreZeros = True  # Whether to ignore zero-valued pixels.

# Read and preprocess the 3D volume.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Calculate the GLCM using the defined function.
coMatrix = CalculateGLCMCooccuranceMatrix3D(
  volumeCropped,  # 3D volume.
  d,  # Distance between voxel pairs.
  theta,  # Angle for the direction of voxel pairs
  isSymmetric=isSymmetric,  # Whether to make the GLCM symmetric.
  isNorm=isNorm,  # Whether to normalize the GLCM.
  ignoreZeros=ignoreZeros,  # Whether to ignore zero-valued pixels.
)

# Calculate texture features from the GLCM.
features = CalculateGLCMFeaturesOptimized(coMatrix)

# Print the GLCM features.
for key in features:
  print(f"{key}:", np.round(features[key], 4))  # Print each feature and its value.
