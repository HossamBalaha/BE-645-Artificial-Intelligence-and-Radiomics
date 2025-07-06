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
distance = 1  # Distance from the center pixel to the surrounding pixels.
isClockwise = False  # Direction of LBP computation (True for clockwise, False for counterclockwise).
theta = 135  # Start from the top-left corner.
normalizeLBP = True  # Flag to normalize LBP values.

# Compute the LBP matrix for the cropped ROI using the defined parameters.
lbpMatrix = LocalBinaryPattern2D(
  cropped,
  distance=distance,
  theta=theta,
  isClockwise=isClockwise,
  normalizeLBP=normalizeLBP,
)

# Calculate the histogram of the image to analyze frequency distribution.
hist2D = []
# Find the minimum and maximum pixel values in the cropped image.
min = int(np.min(cropped))
max = int(np.max(cropped))
# Iterate through the range of pixel values to compute the histogram.
for i in range(min, max + 1):
  # Count the occurrences of each unique pixel value in the image.
  hist2D.append(np.count_nonzero(cropped == i))
hist2D = np.array(hist2D)

# Calculate the histogram of the LBP matrix to analyze frequency distribution.
hist2DLBP = []
# Find the minimum and maximum LBP values in the LBP matrix.
minLBP = int(np.min(lbpMatrix))
maxLBP = int(np.max(lbpMatrix))
# Iterate through the range of LBP values to compute the histogram.
for i in range(minLBP, maxLBP + 1):
  # Count the occurrences of each unique LBP value in the matrix.
  hist2DLBP.append(np.count_nonzero(lbpMatrix == i))
hist2DLBP = np.array(hist2DLBP)

# Calculate percentiles of the histogram to summarize the distribution.
quantiles = [10, 25, 50, 75, 90]
percentiles = np.percentile(hist2D, quantiles)
percentilesLBP = np.percentile(hist2DLBP, quantiles)

print("Percentiles:")
print("Histogram Percentiles:", percentiles)
print("LBP Histogram Percentiles:", percentilesLBP)

# Visualize the original image and the computed LBP image side by side.
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 2, 2)
plt.imshow(lbpMatrix, cmap="gray")
plt.title("LBP Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.savefig("Data/LBP_Image.png", dpi=720, bbox_inches="tight")
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.

# Plot the histogram of the LBP matrix to visualize the frequency distribution of LBP values.
plt.figure()
plt.subplot(2, 1, 1)
plt.bar(np.arange(min, max + 1), hist2D)
plt.title("Histogram of Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(2, 1, 2)
plt.bar(np.arange(minLBP + 1, maxLBP), hist2DLBP[1:-1])
plt.title("2D Histogram of LBP")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.savefig("Data/LBP_Histogram.png", dpi=720, bbox_inches="tight")
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
