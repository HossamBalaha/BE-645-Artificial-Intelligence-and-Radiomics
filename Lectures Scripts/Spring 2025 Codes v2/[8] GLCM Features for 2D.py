'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 29th, 2024
# Last Modification Date: Jan 23rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.


def CalculateGLCMCooccuranceMatrix(image, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True):
  """
  Calculate the Gray-Level Co-occurrence Matrix (GLCM) for a given image.

  Args:
      image (numpy.ndarray): The input image as a 2D NumPy array.
      d (int): The distance between pixel pairs.
      theta (float): The angle (in radians) for the direction of pixel pairs.
      isSymmetric (bool): Whether to make the GLCM symmetric. Default is False.
      isNorm (bool): Whether to normalize the GLCM. Default is True.
      ignoreZeros (bool): Whether to ignore zero-valued pixels. Default is True.

  Returns:
      coMatrix (numpy.ndarray): The calculated GLCM.
  """
  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  if (d < 1):
    raise ValueError("The distance between voxel pairs should be greater than or equal to 1.")
  elif (d >= N):
    raise ValueError("The distance between voxel pairs should be less than the number of unique intensity levels.")

  # Initialize the co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N))  # Create an N x N matrix filled with zeros.

  # Iterate over each pixel in the image to calculate the GLCM.
  for xLoc in range(image.shape[1]):  # Loop through columns.
    for yLoc in range(image.shape[0]):  # Loop through rows.
      startLoc = (yLoc, xLoc)  # Current pixel location (row, column).

      # Calculate the target pixel location based on distance and angle.
      xTarget = xLoc + np.round(d * np.cos(theta))  # Target column.
      yTarget = yLoc - np.round(d * np.sin(theta))  # Target row.
      endLoc = (int(yTarget), int(xTarget))  # Target pixel location.

      # Check if the target location is within the bounds of the image.
      if (
        (endLoc[0] < 0)  # Target row is above the top edge.
        or (endLoc[0] >= image.shape[0])  # Target row is below the bottom edge.
        or (endLoc[1] < 0)  # Target column is to the left of the left edge.
        or (endLoc[1] >= image.shape[1])  # Target column is to the right of the right edge.
      ):
        continue  # Skip this pair if the target is out of bounds.

      if (ignoreZeros):
        # Skip the calculation if the pixel values are zero.
        if ((image[endLoc] == 0) or (image[startLoc] == 0)):
          continue

      # Increment the co-occurrence matrix at the corresponding location.
      # (- minA) is added to work with matrices that does not start from 0.
      coMatrix[image[endLoc] - minA, image[startLoc] - minA] += 1  # Increment the count for the pair (start, end).

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)  # Divide each element by the sum of all elements.

  return coMatrix  # Return the calculated GLCM.


def CalculateGLCMFeatures(coMatrix):
  """
  Calculate texture features from a Gray-Level Co-occurrence Matrix (GLCM).

  Args:
      coMatrix (numpy.ndarray): The GLCM as a 2D NumPy array.

  Returns:
      features (dict): A dictionary containing the calculated texture features.
  """
  N = coMatrix.shape[0]  # Number of unique intensity levels.

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)  # Sum of the squares of all elements in the GLCM.

  # Calculate the contrast in the direction of theta.
  contrast = 0.0  # Initialize contrast.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      contrast += (i - j) ** 2 * coMatrix[i, j]  # Weighted sum of squared differences.

  # Calculate the homogeneity of the co-occurrence matrix.
  homogeneity = 0.0  # Initialize homogeneity.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

  # Calculate the entropy of the co-occurrence matrix.
  entropy = 0.0  # Initialize entropy.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      if (coMatrix[i, j] > 0):  # Check if the value is greater than zero.
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

  # Calculate the correlation of the co-occurrence matrix.
  totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
  meanX = 0.0  # Initialize mean of rows.
  meanY = 0.0  # Initialize mean of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
      meanY += j * coMatrix[i, j]  # Weighted sum of column indices.
  meanX /= totalSum  # Calculate mean of rows.
  meanY /= totalSum  # Calculate mean of columns.

  stdDevX = 0.0  # Initialize standard deviation of rows.
  stdDevY = 0.0  # Initialize standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      stdDevX += (i - meanX) ** 2 * coMatrix[i, j]  # Weighted sum of squared row differences.
      stdDevY += (j - meanY) ** 2 * coMatrix[i, j]  # Weighted sum of squared column differences.

  correlation = 0.0  # Initialize correlation.
  stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
  stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      correlation += (
        (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)
      )  # Weighted sum of normalized differences.

  # Calculate the dissimilarity of the co-occurrence matrix.
  dissimilarity = 0.0  # Initialize dissimilarity.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      dissimilarity += np.abs(i - j) * coMatrix[i, j]  # Weighted sum of absolute differences.

  # Return the calculated features as a dictionary.
  return {
    "Energy"       : energy,  # Energy of the GLCM.
    "Contrast"     : contrast,  # Contrast of the GLCM.
    "Homogeneity"  : homogeneity,  # Homogeneity of the GLCM.
    "Entropy"      : entropy,  # Entropy of the GLCM.
    "Correlation"  : correlation,  # Correlation of the GLCM.
    "Dissimilarity": dissimilarity,  # Dissimilarity of the GLCM.
    "TotalSum"     : totalSum,  # Sum of all elements in the GLCM.
    "MeanX"        : meanX,  # Mean of rows.
    "MeanY"        : meanY,  # Mean of columns.
    "StdDevX"      : stdDevX,  # Standard deviation of rows.
    "StdDevY"      : stdDevY,  # Standard deviation of columns.
  }


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
theta = np.radians(theta)  # Convert theta to radians.
isSymmetric = False  # Whether to make the GLCM symmetric.
isNorm = True  # Whether to normalize the GLCM.
ignoreZeros = True  # Whether to ignore zero-valued pixels.

# Calculate the GLCM using the defined function.
coMatrix = CalculateGLCMCooccuranceMatrix(
  cropped,  # Input image.
  d,  # Distance between pixel pairs.
  theta,  # Angle (in radians) for the direction of pixel pairs.
  isSymmetric=isSymmetric,  # Whether to make the GLCM symmetric.
  isNorm=isNorm,  # Whether to normalize the GLCM.
  ignoreZeros=ignoreZeros,  # Whether to ignore zero-valued pixels.
)

# Calculate texture features from the GLCM.
features = CalculateGLCMFeatures(coMatrix)

# Print the GLCM features.
for key in features:
  print(key, ":", np.round(features[key], 4))  # Print each feature and its value.

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

plt.show()  # Display the figure.
plt.close()  # Close the figure to free up memory.
