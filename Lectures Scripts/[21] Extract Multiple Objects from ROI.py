'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 29th, 2024
# Last Modification Date: Jun 19th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # Provides functions for interacting with the operating system.
import cv2  # OpenCV library for computer vision tasks.
import numpy as np  # Numerical computing library for array operations.
import matplotlib.pyplot as plt  # Visualization library for plotting.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Define file paths for input images using raw strings to handle Windows paths.
caseImgPath = r"Data/Sample Normal Chest Image.png"  # Path to the medical scan image.
caseSegPath = r"Data/Sample Normal Chest Mask.png"  # Path to the segmentation mask.

# Another example with different paths for testing purposes.
# caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the medical scan image.
# caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the segmentation mask.

# Set target dimensions for image resizing to ensure consistent processing size.
targetSize = (256, 256)  # Tuple specifying width and height in pixels.
# Define minimum contour area threshold for filtering small regions (0 means keep all).
cntAreaThreshold = 0  # Measured in square pixels.

# Check if both input files exist before proceeding with processing.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  # Raise explicit error if any file is missing to prevent runtime failures.
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load medical images from disk using appropriate OpenCV flags.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Read mask as single-channel.

# Check if images were loaded correctly, raise error if not.
if ((caseImg is None) or (caseSeg is None)):
  # Raise error with descriptive message if image loading fails.
  raise ValueError("Error reading the image or mask. Please check the file paths and formats.")

# Call the function to extract multiple objects from the ROI in the medical image.
regions = ExtractMultipleObjectsFromROI(
  caseImg,  # The original medical image.
  caseSeg,  # The segmentation mask indicating regions of interest.
  targetSize=targetSize,  # Resize images to this target size.
  cntAreaThreshold=cntAreaThreshold  # Minimum contour area to consider for extraction.
)

# Output the total number of valid regions found in the segmentation mask.
print(f"Number of Regions: {len(regions)}.")

plt.figure(figsize=(12, 6))

plt.subplot(1, len(regions) + 2, 1)
plt.imshow(caseImg, cmap="gray")  # Display original image in grayscale.
plt.title("Original Image")  # Set the title of the plot.
plt.axis("off")  # Turn off the axis.
plt.tight_layout()  # Apply tight layout.

plt.subplot(1, len(regions) + 2, 2)
plt.imshow(caseSeg, cmap="gray")  # Display segmentation mask in grayscale.
plt.title("Segmentation Mask")  # Set the title of the plot.
plt.axis("off")  # Turn off the axis.
plt.tight_layout()  # Apply tight layout.

# Configure matplotlib subplots for multi-image visualization.
for i in range(len(regions)):
  # Create subplot grid with one row and N columns for N regions.
  plt.subplot(1, len(regions) + 2, i + 3)
  # Display current region in grayscale with proper colormap.
  plt.imshow(regions[i], cmap="gray")
  # Add descriptive title to each subplot showing region number.
  plt.title(f"Region {i + 1}")
  # Remove axis ticks and labels for cleaner visualization.
  plt.axis("off")
  # Apply tight layout to ensure proper spacing between subplots.
  plt.tight_layout()

# Save the final figure as a PNG image with high resolution and tight bounding box.
plt.savefig(
  "Data/Extract Multiple Objects from ROI.png",
  dpi=300,
  bbox_inches="tight",
)
# Render the matplotlib figure with all extracted regions.
plt.show()
# Close the figure to free memory after user interaction.
plt.close()
