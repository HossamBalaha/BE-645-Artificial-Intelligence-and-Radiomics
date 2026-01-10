'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 20th, 2024
# Last Modification Date: Jul 6th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing and reading files.
import numpy as np  # For numerical operations.
from HMB_Summer_2025_Helpers import *

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

# Set the parameters for the LBP computation.
distanceLBP = 1  # Distance from the center pixel to the surrounding pixels.
isClockwiseLBP = False  # Direction of LBP computation (True for clockwise, False for counterclockwise).
thetaLBP = 135  # Start from the top-left corner.
normalizeLBP = True  # Flag to normalize LBP values.

# Define parameters for the GLCM calculation.
distanceGLCM = 1  # Distance between pixel pairs.
thetaGLCM = 0  # Angle (in degrees) for the direction of pixel pairs.
thetaGLCM = np.radians(thetaGLCM)  # Convert theta to radians.
isSymmetricGLCM = False  # Whether to make the GLCM symmetric.
isNormGLCM = True  # Whether to normalize the GLCM.
ignoreZerosGLCM = True  # Whether to ignore zero-valued pixels.

# Calculate the GLCM co-occurrence matrix for the cropped image.
coMatrixCropped = CalculateGLCMCooccuranceMatrix(
  cropped,  # Input image for GLCM calculation.
  d=distanceGLCM,  # Distance between pixel pairs.
  theta=thetaGLCM,  # Angle in radians for the direction of pixel pairs.
  isSymmetric=isSymmetricGLCM,  # Whether to make the GLCM symmetric.
  isNorm=isNormGLCM,  # Whether to normalize the GLCM.
  ignoreZeros=ignoreZerosGLCM,  # Whether to ignore zero-valued pixels.
)
# Calculate the GLCM features for the cropped image.
featuresGLCM = CalculateGLCMFeaturesOptimized(coMatrixCropped)

# Calculate the GLCM features for the LBP image.
lbpMatrix = LocalBinaryPattern2D(
  cropped, distance=distanceLBP, theta=thetaLBP,
  isClockwise=isClockwiseLBP, normalizeLBP=normalizeLBP,
)
# Invert the LBP matrix to match the GLCM calculation.
lbpMatrix = 255 - lbpMatrix
# Calculate the GLCM co-occurrence matrix for the LBP image.
coMatrixLBP = CalculateGLCMCooccuranceMatrix(
  lbpMatrix, d=distanceGLCM, theta=thetaGLCM,
  isSymmetric=isSymmetricGLCM, isNorm=isNormGLCM,
  ignoreZeros=ignoreZerosGLCM,
)
# Calculate the GLCM features for the LBP image.
featuresLBP = CalculateGLCMFeaturesOptimized(coMatrixLBP)

print("-" * 50)
print(f"%15s: %12s \t %12s" % ("Feature", "Image GLCM", "GLCM from LBP"))
print("-" * 50)
for k in featuresGLCM.keys():
  print(f"%15s: {featuresGLCM[k]:12.4f} \t {featuresLBP[k]:12.4f}" % k)
print("-" * 50)
