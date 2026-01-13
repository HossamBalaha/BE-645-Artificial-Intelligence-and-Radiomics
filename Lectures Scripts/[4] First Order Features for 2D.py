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
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.
from HMB_Spring_2026_Helpers import *  # Import custom helper functions.

# Define the paths to the input image and segmentation mask.
caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the liver image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the liver segmentation mask.

# Load the images in grayscale mode.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the liver image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Extract the Region of Interest (ROI) using the mask.
roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

# Check if the ROI is empty (background-only).
if (np.sum(roi) == 0):
  raise ValueError("The segmentation mask is empty. Please provide a valid mask.")

# Crop the ROI to remove unnecessary background.
x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

# Ensure the cropped region is not empty.
if (np.sum(cropped) == 0):
  raise ValueError("The cropped region is empty. Please check the segmentation mask.")

# Calculate first-order features for 2D images.
# With normalization and ignoring zeros (background).
fos, hist2D = FirstOrderFeatures2D(cropped, isNorm=True, ignoreZeros=True)
# With normalization and without ignoring zeros (background).
# fos, hist2D = FirstOrderFeatures2D(cropped, isNorm=True, ignoreZeros=False)

print("First Order Features:")  # Print the header for the features.
for key, value in fos.items():  # Iterate through the features' dictionary.
  print(f"{key}: {np.round(value, 4)}")  # Print each feature with its value formatted to 4 decimal places.

# Plot the histogram.
min = int(fos["Min"])  # Get the minimum pixel value from the features.
max = int(fos["Max"])  # Get the maximum pixel value from the features.
plt.figure()  # Create a new figure for the plot.
plt.bar(np.arange(min, max + 1), hist2D)  # Plot the histogram as a bar chart.
plt.title("2D Histogram")  # Set the title of the plot.
plt.xlabel("Pixel Value")  # Label the x-axis.
plt.ylabel("Frequency")  # Label the y-axis.
plt.tight_layout()  # Adjust the layout for better visualization.

# Save the histogram plot as an image file.
ext = caseImgPath.split(".")[-1]  # Get the file extension of the input image.
plt.savefig(
  caseImgPath.replace(f".{ext}", f" Histogram.jpg"),  # Replace the file extension and save the plot.
  dpi=720,  # Set the resolution of the saved image.
  bbox_inches="tight",  # Ensure the entire plot is saved without cropping.
)

plt.show()  # Display the histogram plot.
plt.close()  # Close the figure.
plt.clf()  # Clear the current figure.
