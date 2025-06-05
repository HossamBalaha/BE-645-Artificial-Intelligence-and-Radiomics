'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 29th, 2024
# Last Modification Date: Jun 5th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.
from skimage.feature import graycomatrix, graycoprops  # For GLCM and texture feature calculations.

# Define the paths to the input image and segmentation mask.
caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the liver image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the liver segmentation mask.

# Check if the files exist.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load the images in grayscale mode.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the liver image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Extract the Region of Interest (ROI) using the segmentation mask.
roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

# Crop the ROI to remove unnecessary background.
x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

if (np.sum(cropped) <= 0):
  raise ValueError("The cropped image is empty. Please check the segmentation mask.")

# Define parameters for the GLCM calculation.
d = 1  # Distance between pixel pairs.
theta = 0  # Angle (in degrees) for the direction of pixel pairs.
isSymmetric = False  # Whether to make the GLCM symmetric.
isNorm = True  # Whether to normalize the GLCM.

N = np.max(cropped) + 1  # Number of unique intensity levels.

# Calculate the Gray-Level Co-occurrence Matrix (GLCM) using `skimage`.
coMatrix = graycomatrix(
  cropped,  # Input image.
  distances=[d],  # List of distances between pixel pairs.
  angles=[theta],  # List of angles (in radians) for the direction of pixel pairs.
  levels=N,  # Number of gray levels.
  symmetric=isSymmetric,  # Whether to make the GLCM symmetric.
  normed=isNorm,  # Whether to normalize the GLCM.
)

# Extract GLCM features using `skimage`.
contrast = graycoprops(coMatrix[1:, 1:, :, :], "contrast")[0, 0]  # Contrast feature.
correlation = graycoprops(coMatrix[1:, 1:, :, :], "correlation")[0, 0]  # Correlation feature.
energy = graycoprops(coMatrix[1:, 1:, :, :], "energy")[0, 0]  # Energy feature.
homogeneity = graycoprops(coMatrix[1:, 1:, :, :], "homogeneity")[0, 0]  # Homogeneity feature.
ASM = graycoprops(coMatrix[1:, 1:, :, :], "ASM")[0, 0]  # Angular Second Moment (ASM) feature.

# Print the GLCM features.
print("Contrast:", np.round(contrast, 4))  # Print the contrast feature.
print("Correlation:", np.round(correlation, 4))  # Print the correlation feature.
print("Energy:", np.round(energy, 4))  # Print the energy feature.
print("Homogeneity:", np.round(homogeneity, 4))  # Print the homogeneity feature.
print("ASM:", np.round(ASM, 4))  # Print the ASM feature.

# Display the cropped image and the co-occurrence matrix.
plt.figure()  # Create a new figure.
plt.subplot(1, 2, 1)  # Create a subplot in the first position.
plt.imshow(cropped, cmap="gray")  # Display the cropped image in grayscale.
plt.title("Cropped Image")  # Set the title of the subplot.
plt.axis("off")  # Hide the axes.
plt.colorbar()  # Add a color bar to show intensity values.
plt.tight_layout()  # Adjust the layout for better visualization.

plt.subplot(1, 2, 2)  # Create a subplot in the second position.
plt.imshow(coMatrix[1:, 1:, 0, 0], cmap="gray")  # Display the GLCM in grayscale.
plt.title("Co-occurrence Matrix")  # Set the title of the subplot.
plt.colorbar()  # Add a color bar to show intensity values.
plt.tight_layout()  # Adjust the layout for better visualization.

plt.show()  # Display the histogram plot.
plt.close()  # Close the figure.
plt.clf()  # Clear the current figure.
