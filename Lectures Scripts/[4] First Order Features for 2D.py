'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 20th, 2024
# Last Modification Date: May 21st, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Define the paths to the input image and segmentation mask.
caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the liver image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the liver segmentation mask.

# Load the images in grayscale mode.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the liver image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Calculate first-order features for 2D images.
# With normalization and ignoring zeros (background).
# fos, hist2D = FirstOrderFeatures2D(caseImg, caseSeg, isNorm=True, ignoreZeros=True)
# With normalization and without ignoring zeros (background).
fos, hist2D = FirstOrderFeatures2D(caseImg, caseSeg, isNorm=True, ignoreZeros=False)
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
  dpi=300,  # Set the resolution of the saved image.
  bbox_inches="tight",  # Ensure the entire plot is saved without cropping.
)

plt.show()  # Display the histogram plot.
plt.close()  # Close the figure.
plt.clf()  # Clear the current figure.
