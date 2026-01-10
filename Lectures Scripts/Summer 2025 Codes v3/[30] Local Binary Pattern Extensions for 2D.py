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
import matplotlib.pyplot as plt  # For plotting and visualization.
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

# Compute the LBP matrix and uniform LBP matrix for the cropped image.
lbpMatrix = LocalBinaryPattern2D(
  cropped,
  distance=distanceLBP,
  theta=thetaLBP,
  isClockwise=isClockwiseLBP,
  normalizeLBP=normalizeLBP,
)
lbpMatrixUniform = UniformLocalBinaryPattern2D(
  cropped,
  distance=distanceLBP,
  theta=thetaLBP,
  isClockwise=isClockwiseLBP,
  normalizeLBP=normalizeLBP,
)

# Display the original image, LBP image, and uniform LBP image side by side.
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 3, 2)
plt.imshow(lbpMatrix, cmap="gray")
plt.title("LBP Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 3, 3)
plt.imshow(lbpMatrixUniform, cmap="gray")
plt.title("Uniform LBP Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.savefig("Data/LBP_Images.png", dpi=720, bbox_inches="tight")  # Save the plot as an image.
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
