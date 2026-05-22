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
from HMB_Spring_2026_Helpers import *

# Define the range of slices to be processed.
contentRng = (45, 73)

caseImgPaths = [
  rf"Data/Volume Slices/Volume Slice {i}.bmp"
  for i in range(contentRng[0], contentRng[1] + 1)
]
caseSegPaths = [
  rf"Data/Segmentation Slices/Segmentation Slice {i}.bmp"
  for i in range(contentRng[0], contentRng[1] + 1)
]

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(
  caseImgPaths,  # Paths to image slices.
  caseSegPaths,  # Paths to segmentation slices.
  raiseErrors=False,  # Do not raise errors.
)

# Extract shape features from the preprocessed volume.
shapeFeatures = ShapeFeatures3D(volumeCropped)

# Print the shape features.
# Print the calculated shape features.
print("Shape Features:")
for feature, value in shapeFeatures.items():
  print(f"{feature}: {value:0.4f}")
