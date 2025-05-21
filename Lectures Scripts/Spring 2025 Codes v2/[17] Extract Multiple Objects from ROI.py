'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 29th, 2024
# Last Modification Date: Feb 3rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import standard Python libraries for system operations and data processing.
import os  # Provides functions for interacting with the operating system.
import cv2  # OpenCV library for computer vision tasks.
import numpy as np  # Numerical computing library for array operations.
import matplotlib.pyplot as plt  # Visualization library for plotting.

# Define file paths for input images using raw strings to handle Windows paths.
caseImgPath = r"Data/Sample Normal Chest Image.png"  # Path to the medical scan image.
caseSegPath = r"Data/Sample Normal Chest Mask.png"  # Path to the segmentation mask.

# Set target dimensions for image resizing to ensure consistent processing size.
targetSize = (256, 256)  # Tuple specifying width and height in pixels.
# Define minimum contour area threshold for filtering small regions (0 means keep all).
cntAreaThreshold = 0  # Measured in square pixels.

# Check if both input files exist before proceeding with processing.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  # Raise explicit error if any file is missing to prevent runtime failures.
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load medical images from disk using appropriate OpenCV flags.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Read CT scan as grayscale.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Read mask as single-channel.

# Resize images to standard dimensions using cubic interpolation for quality.
caseImg = cv2.resize(caseImg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize scan image.
caseSeg = cv2.resize(caseSeg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize mask image.

# Binarize segmentation mask by thresholding to ensure only 0/255 values.
caseSeg[caseSeg > 0] = 255  # Convert any positive values to pure white.

# Perform sanity check on the segmentation mask to ensure valid content.
if (np.sum(caseSeg) <= 0):  # Calculate sum of all pixel values in mask.
  # Raise error if mask contains no white pixels to prevent empty processing.
  raise ValueError("The mask is completely black/empty. Please check the segmentation mask.")

# Detect contours in the segmentation mask using simple approximation method.
contours = cv2.findContours(caseSeg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Sort detected contours from left-to-right based on their x-coordinate.
contours = sorted(contours[0], key=lambda x: cv2.boundingRect(x)[0], reverse=False)

# Initialize empty list to store extracted region-of-interest (ROI) images.
regions = []

# Process each detected contour to extract individual anatomical structures.
for i in range(len(contours)):
  # Calculate the area of the current contour for size filtering.
  cntArea = cv2.contourArea(contours[i])
  # Skip contours smaller than threshold to ignore noise/artifacts.
  if (cntArea <= cntAreaThreshold):
    continue

  # Create blank mask matching image dimensions for current ROI.
  regionMask = np.zeros_like(caseSeg)
  # Select current contour from the list of detected contours.
  regionCnt = contours[i]
  # Fill contour area in the mask to create binary ROI representation.
  cv2.fillPoly(regionMask, [regionCnt], 255)
  # Apply mask to original image to isolate the anatomical structure.
  roi = cv2.bitwise_and(caseImg, regionMask)
  # Calculate bounding box coordinates around the masked region.
  x, y, w, h = cv2.boundingRect(roi)
  # Crop the region from the original image using bounding box coordinates.
  cropped = roi[y:y + h, x:x + w]
  # Add cropped ROI to the collection of extracted regions.
  regions.append(cropped)

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
