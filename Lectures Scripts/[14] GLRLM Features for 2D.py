'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 2nd, 2024
# Last Modification Date: Jan 28th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


def CalculateGLRLMRunLengthMatrix(matrix, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate the Gray-Level Run-Length Matrix (GLRLM) for a given 2D matrix.

  The GLRLM is a statistical tool used to quantify the texture of an image by
  analyzing the runs of pixels with the same intensity level in a specific direction.

  Parameters:
  -----------
  matrix : numpy.ndarray
      A 2D matrix representing the image or data for which the GLRLM is to be calculated.

  theta : float
      The angle (in radians) specifying the direction in which runs are to be analyzed.
      The direction is determined by the cosine and sine of this angle.

  isNorm : bool, optional (default=True)
      If True, the resulting GLRLM is normalized by dividing by the total number of runs.
      Normalization ensures that the matrix represents probabilities rather than counts.

  ignoreZeros : bool, optional (default=True)
      If True, runs with zero intensity are ignored in the calculation of the GLRLM.
      This is useful when zero values represent background or irrelevant data.

  Returns:
  --------
  rlMatrix : numpy.ndarray
      A 2D matrix representing the Gray-Level Run-Length Matrix. The rows correspond to
      intensity levels, and the columns correspond to run lengths. If `isNorm` is True,
      the matrix is normalized.
  """
  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(matrix)  # Find the minimum intensity value in the matrix.
  maxA = np.max(matrix)  # Find the maximum intensity value in the matrix.
  N = maxA - minA + 1  # Calculate the number of unique intensity levels.
  R = np.max(matrix.shape)  # Determine the maximum possible run length.

  rlMatrix = np.zeros((N, R))  # Initialize the run-length matrix with zeros.
  seenMatrix = np.zeros(matrix.shape)  # Initialize a matrix to track seen pixels.
  # Negative sign is used to ensure the direction is consistent with the angle.
  dx = -int(np.round(np.cos(theta)))  # Calculate the x-direction step based on theta.
  dy = int(np.round(np.sin(theta)))  # Calculate the y-direction step based on theta.

  for i in range(matrix.shape[0]):  # Iterate over each row in the matrix.
    for j in range(matrix.shape[1]):  # Iterate over each column in the matrix.
      # Skip the pixel if it has already been processed.
      if (seenMatrix[i, j] == 1):
        continue

      seenMatrix[i, j] = 1  # Mark the current pixel as seen.
      currentPixel = matrix[i, j]  # Get the intensity value of the current pixel.
      d = 1  # Initialize the run length distance.

      # Check consecutive pixels in the direction specified by theta.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1  # Mark the pixel as seen.
          d += 1  # Increment the run length distance.
        else:
          break  # Stop if the run ends.

      # Skip zero-intensity runs if ignoreZeros is True.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update the run-length matrix.
      # (- minA) is added to work with matrices that does not start from 0.
      rlMatrix[currentPixel - minA, d - 1] += 1

  if (isNorm):
    # Normalize the run-length matrix by dividing by the total number of runs.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix  # Return the computed run-length matrix.


def CalculateGLRLMFeatures(rlMatrix, image):
  """
  Calculate texture features from a Gray-Level Run-Length Matrix (GLRLM).

  This function computes various texture features based on the GLRLM, which is derived
  from an image. These features are commonly used in texture analysis and image processing.

  Parameters:
  -----------
  rlMatrix : numpy.ndarray
      A 2D Gray-Level Run-Length Matrix (GLRLM) computed from an image. The rows represent
      intensity levels, and the columns represent run lengths.

  image : numpy.ndarray
      The original 2D image from which the GLRLM was derived. This is used to determine
      the number of gray levels and the total number of pixels.

  Returns:
  --------
  features : dict
      A dictionary containing the following texture features:
      - "Short Run Emphasis"          : Emphasizes short runs in the image.
      - "Long Run Emphasis"           : Emphasizes long runs in the image.
      - "Gray Level Non-Uniformity"   : Measures the variability of gray levels.
      - "Run Length Non-Uniformity"   : Measures the variability of run lengths.
      - "Run Percentage"              : Ratio of runs to the total number of pixels.
      - "Low Gray Level Run Emphasis" : Emphasizes runs with low gray levels.
      - "High Gray Level Run Emphasis": Emphasizes runs with high gray levels.
  """

  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(image)  # Find the minimum intensity value in the matrix.
  maxA = np.max(image)  # Find the maximum intensity value in the matrix.
  N = maxA - minA + 1  # Calculate the number of unique intensity levels.
  R = np.max(image.shape)  # Maximum run length.

  rlN = np.sum(rlMatrix)  # Total number of runs in the matrix.

  # Short Run Emphasis: Emphasizes short runs in the image.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Long Run Emphasis: Emphasizes long runs in the image.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Gray Level Non-Uniformity: Measures the variability of gray levels.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Sum of each row.
  ) / rlN

  # Run Length Non-Uniformity: Measures the variability of run lengths.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Sum of each column.
  ) / rlN

  # Run Percentage: Ratio of runs to the total number of pixels.
  rp = rlN / np.prod(image.shape)

  # Low Gray Level Run Emphasis: Emphasizes runs with low gray levels.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # High Gray Level Run Emphasis: Emphasizes runs with high gray levels.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  return {
    "Short Run Emphasis"          : sre,
    "Long Run Emphasis"           : lre,
    "Gray Level Non-Uniformity"   : gln,
    "Run Length Non-Uniformity"   : rln,
    "Run Percentage"              : rp,
    "Low Gray Level Run Emphasis" : lgre,
    "High Gray Level Run Emphasis": hgre,
  }  # Return a dictionary of computed features.


caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the input image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the segmentation mask.

# Check if the files exist.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load the images.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the grayscale image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Extract the ROI using the segmentation mask.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Crop the ROI to the bounding box of the segmented region.
x, y, w, h = cv2.boundingRect(roi)
cropped = roi[y:y + h, x:x + w]

# Check if the cropped image is empty.
if (np.sum(cropped) <= 0):
  raise ValueError("The cropped image is empty. Please check the segmentation mask.")

theta = 0  # Angle for run-length matrix calculation.
thetaRad = np.radians(theta)  # Convert angle to radians.

# Compute the run-length matrix for the cropped image.
rlMatrix = CalculateGLRLMRunLengthMatrix(cropped, thetaRad, isNorm=True, ignoreZeros=True)

# Calculate GLRLM features from the run-length matrix.
features = CalculateGLRLMFeatures(rlMatrix, cropped)

# Print the GLRLM features.
print(f"At angle {theta} degrees:")
for key in features:
  print(key, ":", np.round(features[key], 4))  # Print each feature with 4 decimal places.
