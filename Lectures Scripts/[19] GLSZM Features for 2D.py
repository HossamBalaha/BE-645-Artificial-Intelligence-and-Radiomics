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
from HMB_Spring_2026_Helpers import *

# Define path to sample liver image file.
caseImgPath = r"Data/Sample Liver Image.bmp"
# Define path to corresponding liver segmentation mask file.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"

# Verify both image and mask files exist before attempting to load.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  # Raise error with descriptive message if any file is missing.
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load input image as grayscale (single channel intensity values).
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
# Load segmentation mask as grayscale (binary or labeled format).
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Extract Region of Interest (ROI) by masking input image with segmentation.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Calculate bounding box coordinates of non-zero region in ROI.
x, y, w, h = cv2.boundingRect(roi)
# Crop ROI to tight bounding box around segmented area.
cropped = roi[y:y + h, x:x + w]

# Validate cropped image contains non-zero pixels to prevent empty processing.
if (np.sum(cropped) <= 0):
  # Raise error if cropped image is completely black/empty.
  raise ValueError("The cropped image is empty. Please check the segmentation mask.")

# Set the connectivity type (4 or 8).
C = 8

# Calculate the Size-Zone Matrix.
szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(
  cropped,
  connectivity=C,
  isNorm=True,
  ignoreZeros=True
)

# Compute features from Size-Zone Matrix.
features = CalculateGLSZMFeatures(szMatrix, cropped, N, Z)

# Print the connectivity value.
print(f"At connectivity = {C}:")
# Iterate through computed features and print formatted values.
for key in features:
  # Print feature name with value rounded to 4 decimal places.
  print(key, ":", np.round(features[key], 4))
