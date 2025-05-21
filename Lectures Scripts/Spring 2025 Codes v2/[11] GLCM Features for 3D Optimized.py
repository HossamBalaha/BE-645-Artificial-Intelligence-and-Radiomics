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


def ReadVolume(caseImgPaths, caseSegPaths):
  """
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Args:
      caseImgPaths (list): List of paths to the 2D slices of the volume.
      caseSegPaths (list): List of paths to the segmentation masks of the slices.

  Returns:
      volumeCropped (numpy.ndarray): A 3D NumPy array representing the preprocessed volume.
  """
  volumeCropped = []  # Initialize a list to store the cropped slices.

  # Loop through each slice and its corresponding segmentation mask.
  for i in range(len(caseImgPaths)):
    # Check if the files exist.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load the slice and segmentation mask in grayscale mode.
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the slice.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

    # Extract the Region of Interest (ROI) using the segmentation mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

    # Crop the ROI to remove unnecessary background.
    x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
    cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

    if (np.sum(cropped) <= 0):
      raise ValueError("The cropped image is empty. Please check the segmentation mask.")

    # Append the cropped slice to the list.
    volumeCropped.append(cropped)

  # Determine the maximum width and height across all cropped slices.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])  # Maximum width.
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])  # Maximum height.

  # Pad each cropped slice to match the maximum width and height.
  for i in range(len(volumeCropped)):
    # Calculate the padding size.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]  # Horizontal padding.
    deltaHeight = maxHeight - volumeCropped[i].shape[0]  # Vertical padding.

    # Add padding to the cropped image and place the image in the center.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],  # Image to pad.
      deltaHeight // 2,  # Top padding.
      deltaHeight - deltaHeight // 2,  # Bottom padding.
      deltaWidth // 2,  # Left padding.
      deltaWidth - deltaWidth // 2,  # Right padding.
      cv2.BORDER_CONSTANT,  # Padding type.
      value=0  # Padding value.
    )

    # Replace the cropped slice with the padded slice.
    volumeCropped[i] = padded.copy()

  # Convert the list of slices to a 3D NumPy array.
  volumeCropped = np.array(volumeCropped)

  return volumeCropped  # Return the preprocessed 3D volume.


def CalculateGLCM3DCooccuranceMatrix(volume, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True):
  """
  Calculate the 3D Gray-Level Co-occurrence Matrix (GLCM) for a given volume.

  Args:
      volume (numpy.ndarray): The 3D volume as a NumPy array.
      d (int): The distance between voxel pairs.
      theta (float): The angle (in radians) for the direction of voxel pairs.
      isSymmetric (bool): Whether to make the GLCM symmetric. Default is False.
      isNorm (bool): Whether to normalize the GLCM. Default is True.
      ignoreZeros (bool): Whether to ignore zero-valued voxels. Default is True.

  Returns:
      coMatrix (numpy.ndarray): The calculated 3D GLCM.
  """

  # Determine the number of unique intensity levels in the volume.
  minA = np.min(volume)  # Minimum intensity value.
  maxA = np.max(volume)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  noOfSlices = volume.shape[0]  # Number of slices in the volume.

  # Initialize the 3D co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N, noOfSlices))

  if (d < 1):
    raise ValueError("The distance between voxel pairs should be greater than or equal to 1.")
  elif (d >= noOfSlices):
    raise ValueError("The distance between voxel pairs should be less than the number of slices.")
  elif (d >= N):
    raise ValueError("The distance between voxel pairs should be less than the number of unique intensity levels.")

  # Iterate over each voxel in the volume to calculate the GLCM.
  for xLoc in range(volume.shape[2]):  # Loop through columns.
    for yLoc in range(volume.shape[1]):  # Loop through rows.
      for zLoc in range(volume.shape[0]):  # Loop through slices.
        startLoc = (zLoc, yLoc, xLoc)  # Current voxel location (slice, row, column).

        # Calculate the target voxel location based on distance and angle.
        xTarget = xLoc + np.round(d * np.cos(theta) * np.sin(theta))  # Target column.
        yTarget = yLoc - np.round(d * np.sin(theta) * np.sin(theta))  # Target row.
        zTarget = zLoc + np.round(d * np.cos(theta))  # Target slice.
        endLoc = (int(zTarget), int(yTarget), int(xTarget))  # Target voxel location.

        # Check if the target location is within the bounds of the volume.
        if (
          (endLoc[0] < 0)  # Target slice is below the bottom slice.
          or (endLoc[0] >= volume.shape[0])  # Target slice is above the top slice.
          or (endLoc[1] < 0)  # Target row is above the top edge.
          or (endLoc[1] >= volume.shape[1])  # Target row is below the bottom edge.
          or (endLoc[2] < 0)  # Target column is to the left of the left edge.
          or (endLoc[2] >= volume.shape[2])  # Target column is to the right of the right edge.
        ):
          continue  # Skip this pair if the target is out of bounds.

        if (ignoreZeros):
          # Skip the calculation if the voxel values are zero.
          if ((volume[endLoc] == 0) or (volume[startLoc] == 0)):
            continue

        # Increment the co-occurrence matrix at the corresponding location.
        # (- minA) is added to work with matrices that does not start from 0.
        coMatrix[volume[endLoc] - minA, volume[startLoc] - minA] += 1  # Increment the count for the pair (start, end).

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)  # Divide each element by the sum of all elements.

  return coMatrix  # Return the calculated 3D GLCM.


def CalculateGLCMFeatures3D(coMatrix):
  """
  Calculate texture features from a 3D Gray-Level Co-occurrence Matrix (GLCM).

  Args:
      coMatrix (numpy.ndarray): The 3D GLCM as a NumPy array.

  Returns:
      features (dict): A dictionary containing the calculated texture features.
  """
  d, h, w = coMatrix.shape  # Dimensions of the GLCM.

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)  # Sum of the squares of all elements in the GLCM.

  # Initialize variables for texture features.
  contrast = 0.0  # Initialize contrast.
  homogeneity = 0.0  # Initialize homogeneity.
  entropy = 0.0  # Initialize entropy.
  dissimilarity = 0.0  # Initialize dissimilarity.
  meanX = 0.0  # Initialize mean of rows.
  meanY = 0.0  # Initialize mean of columns.
  meanZ = 0.0  # Initialize mean of slices.

  # Loop through each element in the GLCM to calculate texture features.
  for i in range(d):  # Loop through rows.
    for j in range(h):  # Loop through columns.
      for k in range(w):  # Loop through slices.
        # Calculate the contrast in the direction of theta.
        contrast += (i - j) ** 2 * coMatrix[i, j, k]  # Weighted sum of squared differences.

        # Calculate the homogeneity of the co-occurrence matrix.
        homogeneity += coMatrix[i, j, k] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

        # Calculate the entropy of the co-occurrence matrix.
        if coMatrix[i, j, k] > 0:  # Check if the value is greater than zero.
          entropy -= coMatrix[i, j, k] * np.log(coMatrix[i, j, k])  # Sum of -p * log(p).

        # Calculate the mean of the co-occurrence matrix.
        meanX += i * coMatrix[i, j, k]  # Weighted sum of row indices.
        meanY += j * coMatrix[i, j, k]  # Weighted sum of column indices.
        meanZ += k * coMatrix[i, j, k]  # Weighted sum of slice indices.

        # Calculate the dissimilarity of the co-occurrence matrix.
        dissimilarity += np.abs(i - j) * coMatrix[i, j, k]  # Weighted sum of absolute differences.

  totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
  meanX /= totalSum  # Calculate mean of rows.
  meanY /= totalSum  # Calculate mean of columns.
  meanZ /= totalSum  # Calculate mean of slices.

  # Calculate the standard deviation of rows, columns, and slices.
  stdDevX = 0.0  # Initialize standard deviation of rows.
  stdDevY = 0.0  # Initialize standard deviation of columns.
  stdDevZ = 0.0  # Initialize standard deviation of slices.
  for i in range(d):  # Loop through rows.
    for j in range(h):  # Loop through columns.
      for k in range(w):  # Loop through slices.
        stdDevX += (i - meanX) ** 2 * coMatrix[i, j, k]  # Weighted sum of squared row differences.
        stdDevY += (j - meanY) ** 2 * coMatrix[i, j, k]  # Weighted sum of squared column differences.
        stdDevZ += (k - meanZ) ** 2 * coMatrix[i, j, k]  # Weighted sum of squared slice differences.

  # Calculate the correlation of the co-occurrence matrix.
  correlation = 0.0  # Initialize correlation.
  stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
  stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
  stdDevZ = np.sqrt(stdDevZ)  # Calculate standard deviation of slices.
  for i in range(d):  # Loop through rows.
    for j in range(h):  # Loop through columns.
      for k in range(w):  # Loop through slices.
        correlation += (
          (i - meanX) * (j - meanY) * (k - meanZ) * coMatrix[i, j, k] / (stdDevX * stdDevY * stdDevZ)
        )  # Weighted sum of normalized differences.

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
    "MeanZ"        : meanZ,  # Mean of slices.
    "StdDevX"      : stdDevX,  # Standard deviation of rows.
    "StdDevY"      : stdDevY,  # Standard deviation of columns.
    "StdDevZ"      : stdDevZ,  # Standard deviation of slices.
  }


# Define the paths to the volume slices and segmentation masks.
caseImgPaths = [
  r"Data/Volume Slices/Volume Slice 65.bmp",  # Path to the first slice.
  r"Data/Volume Slices/Volume Slice 66.bmp",  # Path to the second slice.
  r"Data/Volume Slices/Volume Slice 67.bmp",  # Path to the third slice.
]
caseSegPaths = [
  r"Data/Segmentation Slices/Segmentation Slice 65.bmp",  # Path to the first segmentation mask.
  r"Data/Segmentation Slices/Segmentation Slice 66.bmp",  # Path to the second segmentation mask.
  r"Data/Segmentation Slices/Segmentation Slice 67.bmp",  # Path to the third segmentation mask.
]

# Define parameters for the GLCM calculation.
d = 2  # Distance between voxel pairs.
theta = 0  # Angle (in degrees) for the direction of voxel pairs.
theta = np.radians(theta)  # Convert theta to radians.
# Keep it False unless you are sure that the GLCM can be transposed.
isSymmetric = False  # Whether to make the GLCM symmetric.
isNorm = True  # Whether to normalize the GLCM.
ignoreZeros = True  # Whether to ignore zero-valued pixels.

# Read and preprocess the 3D volume.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Calculate the 3D GLCM using the defined function.
coMatrix = CalculateGLCM3DCooccuranceMatrix(
  volumeCropped,  # 3D volume.
  d,  # Distance between voxel pairs.
  theta,  # Angle for the direction of voxel pairs
  isSymmetric=isSymmetric,  # Whether to make the GLCM symmetric.
  isNorm=isNorm,  # Whether to normalize the GLCM.
  ignoreZeros=ignoreZeros,  # Whether to ignore zero-valued pixels.
)

# Calculate texture features from the 3D GLCM.
features = CalculateGLCMFeatures3D(coMatrix)

# Print the GLCM features.
for key in features:
  print(key, ":", np.round(features[key], 4))  # Print each feature and its value.
