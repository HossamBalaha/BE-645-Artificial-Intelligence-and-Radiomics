'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: June 1st, 2025
# Last Modification Date: June 2nd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import tqdm  # For progress bar in loops.
import numpy as np  # For numerical operations.
import pandas as pd  # For data manipulation and analysis.
import matplotlib.pyplot as plt  # For plotting images and results.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Define the dataset path.
datasetPath = r"../../Datasets/Brain Tumor Dataset Segmentation & Classification/DATASET/Segmentation"

# Define parameters for the first order and GLCM calculation.
d = 1  # Distance between voxel pairs.
theta = 0  # Angle (in degrees) for the direction of voxel pairs.
# Keep it False unless you are sure that the GLCM can be transposed.
isSymmetric = False  # Whether to make the GLCM symmetric.
isNorm = True  # Whether to normalize the GLCM.
ignoreZeros = True  # Whether to ignore zero-valued pixels.
targetSize = (128, 128)  # Target size for resizing images.
doPlotting = True  # Whether to plot the histograms and GLCM visualizations.

# Check if plotting is enabled.
if (doPlotting):
  # Define the path for storing visualizations.
  visualStoragePath = r"Data/Brain Tumor Dataset Segmentation & Classification Visualizations"
  # Create the directory for storing visualizations if it doesn't exist.
  os.makedirs(visualStoragePath, exist_ok=True)

# Convert theta to radians.
theta = np.radians(theta)

# List all classes in the dataset directory.
classes = os.listdir(datasetPath)

# Create a list to store the features extracted from each image.
history = []

# Iterate through each class to process images.
for cls in tqdm.tqdm(classes, desc="Processing classes."):
  # Get the path for the current class.
  clsPath = os.path.join(datasetPath, cls)

  # List all files in the class directory.
  files = os.listdir(clsPath)

  # Filter out files that are images (not masks).
  files = [f for f in files if "_mask" not in f]

  # Iterate through each file in the class directory.
  for file in tqdm.tqdm(files, desc=f"Processing class {cls}."):
    # Get the full path of the file.
    filePath = os.path.join(clsPath, file)

    # Find the corresponding mask file.
    maskPath = filePath.replace(".png", "_mask.png")

    # Read the image and mask.
    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

    # Check if the image and mask are read correctly.
    if (image is None or mask is None):
      print(f"Error reading image or mask for file: {filePath} or {maskPath}. Skipping.")
      continue

    # Resize the image and mask to the target size.
    image = cv2.resize(image, targetSize, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, targetSize, interpolation=cv2.INTER_CUBIC)

    # Ensure the mask is binary (0 or 255).
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Extract the Region of Interest (ROI) using the segmentation mask.
    # Apply bitwise AND operation to extract the ROI.
    roi = cv2.bitwise_and(image, mask)

    # Crop the ROI to remove unnecessary background.
    # Get the bounding box coordinates of the ROI.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop the ROI using the bounding box coordinates.
    cropped = roi[y:y + h, x:x + w]

    # Calculate first order features.
    firstOrderFeatures, hist2D = FirstOrderFeatures2D(
      image,  # The original image.
      mask,  # The segmentation mask.
      isNorm=isNorm,  # Whether to normalize the features.
      ignoreZeros=ignoreZeros,  # Whether to ignore zero-valued pixels.
    )

    # Calculate the GLCM co-occurrence matrix.
    coMatrix = CalculateGLCMCooccuranceMatrix(
      cropped,  # The cropped ROI image.
      d,  # Distance between voxel pairs.
      theta,  # Angle (in radians) for the direction of voxel pairs.
      isSymmetric=isSymmetric,  # Whether to make the GLCM symmetric.
      isNorm=isNorm,  # Whether to normalize the features.
      ignoreZeros=ignoreZeros,  # Whether to ignore zero-valued pixels.
    )

    # Calculate GLCM features from the co-occurrence matrix.
    glcmFeatures = CalculateGLCMFeaturesOptimized(coMatrix)

    allFeatures = {
      "Filename": file,  # Store the filename.
      **firstOrderFeatures,  # Unpack first order features.
      **glcmFeatures,  # Unpack GLCM features.
      "Class"   : cls,  # Store the class label.
    }

    # Append the features to the history list.
    history.append(allFeatures)

    # Check if plotting is enabled.
    if (doPlotting):
      # Optional visualization and storage of the first order histogram.
      histStoragePath = os.path.join(
        visualStoragePath,  # Directory to store the visualizations.
        file.replace(".png", "_Histogram.png"),  # Name of the file to save the histogram visualization.
      )

      # Plot the histogram.
      min = int(firstOrderFeatures["Min"])  # Get the minimum pixel value from the features.
      max = int(firstOrderFeatures["Max"])  # Get the maximum pixel value from the features.
      plt.figure()  # Create a new figure for the plot.
      plt.bar(np.arange(min, max + 1), hist2D)  # Plot the histogram as a bar chart.
      plt.title("2D Histogram")  # Set the title of the plot.
      plt.xlabel("Pixel Value")  # Label the x-axis.
      plt.ylabel("Frequency")  # Label the y-axis.
      plt.tight_layout()  # Adjust the layout for better visualization.

      # Save the histogram plot as an image file.
      plt.savefig(
        histStoragePath,  # Path to save the histogram image.
        dpi=300,  # Set the resolution of the saved image.
        bbox_inches="tight",  # Ensure the entire plot is saved without cropping.
      )

      plt.close()  # Close the figure.
      plt.clf()  # Clear the current figure.

      # Optional visualization and storage of the GLCM.
      coStoragePath = os.path.join(
        visualStoragePath,  # Directory to store the visualizations.
        file.replace(".png", "_CoMatrix.png"),  # Name of the file to save the GLCM visualization.
      )

      # Display the cropped image and the co-occurrence matrix.
      plt.figure()  # Create a new figure.
      plt.subplot(1, 2, 1)  # Create a subplot in the first position.
      plt.imshow(cropped, cmap="gray")  # Display the cropped image in grayscale.
      plt.title("Cropped Image")  # Set the title of the subplot.
      plt.axis("off")  # Hide the axes.
      plt.colorbar()  # Add a color bar to show intensity values.
      plt.tight_layout()  # Adjust the layout for better visualization.

      plt.subplot(1, 2, 2)  # Create a subplot in the second position.
      plt.imshow(coMatrix, cmap="gray")  # Display the GLCM in grayscale.
      plt.title("Co-occurrence Matrix")  # Set the title of the subplot.
      plt.colorbar()  # Add a color bar to show intensity values.
      plt.tight_layout()  # Adjust the layout for better visualization.

      # Save the figure with high resolution.
      plt.savefig(
        coStoragePath,  # Path to save the GLCM co-occurrence matrix image.
        dpi=300,  # Set the resolution of the saved image.
        bbox_inches="tight",  # Ensure the entire plot is saved without cropping.
      )
      plt.close()  # Close the figure.
      plt.clf()  # Clear the current figure.

# Convert the history list to a DataFrame.
featuresDF = pd.DataFrame(history)
# Save the features DataFrame to a CSV file.
featuresDF.to_csv(
  "Data/Brain Tumor Dataset Segmentation & Classification Features.csv",  # Path to save the CSV file.
  index=False,  # Do not include the index in the CSV file.
)
