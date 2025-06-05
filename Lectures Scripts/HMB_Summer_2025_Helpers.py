'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 21st, 2025
# Last Modification Date: Jun 5th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


# ===========================================================================================
# Function(s) for reading and preprocessing 3D medical imaging data.
# ===========================================================================================

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


# ===========================================================================================
# Function(s) for calculating first-order statistical features from an image.
# ===========================================================================================

def FirstOrderFeatures2D(img, mask, isNorm=True, ignoreZeros=True):
  """
  Calculate first-order statistical features from an image using a mask.

  Args:
      img (numpy.ndarray): The input image as a 2D NumPy array.
      mask (numpy.ndarray): The binary mask as a 2D NumPy array.
      isNorm (bool): Flag to indicate whether to normalize the histogram.
      ignoreZeros (bool): Flag to indicate whether to ignore zeros in the histogram.

  Returns:
      results (dict): A dictionary containing the calculated first-order features.
      hist2D (numpy.ndarray): The histogram of the pixel values in the region of interest.
  """
  # Extract the Region of Interest (ROI) using the mask.
  roi = cv2.bitwise_and(img, mask)  # Apply bitwise AND operation to extract the ROI.

  # Crop the ROI to remove unnecessary background.
  x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
  cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

  # Calculate the histogram of the cropped ROI.
  minVal = int(np.min(cropped))  # Find the minimum pixel value in the cropped ROI.
  maxVal = int(np.max(cropped))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [minVal, maxVal].
  for i in range(minVal, maxVal + 1):
    hist2D.append(np.count_nonzero(cropped == i))  # Count occurrences of the value `i` in the cropped ROI.
  hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

  # If ignoreZeros is True, set the first bin (background) to zero.
  if (ignoreZeros and (minVal == 0)):
    # Ignore the background (assumed to be the first bin in the histogram).
    hist2D = hist2D[1:]  # Remove the first bin (background).
    minVal += 1  # Adjust the minimum value to exclude the background.

  # Calculate the total count of values in the histogram before normalization.
  freqCount = np.sum(hist2D)  # Sum all frequencies in the histogram.

  # Normalize the histogram if the flag is set.
  if (isNorm):
    # Normalize the histogram to represent probabilities.
    hist2D = hist2D / np.sum(hist2D)  # Divide each bin by the total count to normalize.

  # Calculate the total count of values from the histogram after normalization.
  count = np.sum(hist2D)  # Sum all probabilities in the normalized histogram.

  # Determine the range of values in the histogram.
  rng = np.arange(minVal, maxVal + 1)  # Create an array of values from `minVal` to `maxVal`.

  # Calculate the sum of values from the histogram.
  sumVal = np.sum(hist2D * rng)  # Multiply each value by its frequency and sum the results.

  # Calculate the mean (average) value from the histogram.
  mean = sumVal / count  # Divide the total sum by the total count.

  # Calculate the variance from the histogram.
  variance = np.sum(hist2D * (rng - mean) ** 2) / count  # Measure of the spread of the data.

  # Calculate the standard deviation from the histogram.
  stdDev = np.sqrt(variance)  # Square root of the variance.

  # Calculate the skewness from the histogram.
  skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)  # Measure of asymmetry in the data.

  # Calculate the kurtosis from the histogram.
  kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)  # Measure of the "tailedness" of the data.

  # Calculate the excess kurtosis from the histogram.
  exKurtosis = kurtosis - 3  # Excess kurtosis relative to a normal distribution.

  # Store the results in a dictionary.
  results = {
    "Min"               : minVal,  # Minimum pixel value.
    "Max"               : maxVal,  # Maximum pixel value.
    "Count"             : count,  # Total count of pixels after normalization.
    "Frequency Count"   : freqCount,  # Total count of pixels before normalization.
    "Sum"               : sumVal,  # Sum of pixel values.
    "Mean"              : mean,  # Mean pixel value.
    "Variance"          : variance,  # Variance of pixel values.
    "Standard Deviation": stdDev,  # Standard deviation of pixel values.
    "Skewness"          : skewness,  # Skewness of pixel values.
    "Kurtosis"          : kurtosis,  # Kurtosis of pixel values.
    "Excess Kurtosis"   : exKurtosis,  # Excess kurtosis of pixel values.
  }

  return results, hist2D


# ===========================================================================================
# Function(s) for calculating Gray-Level Co-occurrence Matrix (GLCM) and its features.
# ===========================================================================================

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
        if ((image[startLoc] == 0) or (image[endLoc] == 0)):
          continue

      # (- minA) is added to work with matrices that does not start from 0.
      # Increment the count for the pair (start, end).
      # image[startLoc] and image[endLoc] are the intensity values at the start and end locations.
      startPixel = image[startLoc] - minA  # Adjust start pixel value.
      endPixel = image[endLoc] - minA  # Adjust end pixel value.

      # Increment the co-occurrence matrix at the corresponding location.
      coMatrix[endPixel, startPixel] += 1

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    # Divide each element by the sum of all elements.
    # 1e-6 is added to avoid division by zero.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix  # Return the calculated GLCM.


def CalculateGLCMFeaturesOptimized(coMatrix):
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

  # Initialize variables for texture features.
  contrast = 0.0  # Initialize contrast.
  homogeneity = 0.0  # Initialize homogeneity.
  entropy = 0.0  # Initialize entropy.
  dissimilarity = 0.0  # Initialize dissimilarity.
  meanX = 0.0  # Initialize mean of rows.
  meanY = 0.0  # Initialize mean of columns.

  # Loop through each element in the GLCM to calculate texture features.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      # Calculate the contrast in the direction of theta.
      contrast += (i - j) ** 2 * coMatrix[i, j]  # Weighted sum of squared differences.

      # Calculate the homogeneity of the co-occurrence matrix.
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

      # Calculate the entropy of the co-occurrence matrix.
      if (coMatrix[i, j] > 0):  # Check if the value is greater than zero.
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

      # Calculate the dissimilarity of the co-occurrence matrix.
      dissimilarity += np.abs(i - j) * coMatrix[i, j]  # Weighted sum of absolute differences.

      # Calculate the mean of the co-occurrence matrix.
      meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
      meanY += j * coMatrix[i, j]  # Weighted sum of column indices.

  totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
  meanX /= totalSum  # Calculate mean of rows.
  meanY /= totalSum  # Calculate mean of columns.

  # Calculate the standard deviation of rows and columns.
  stdDevX = 0.0  # Initialize standard deviation of rows.
  stdDevY = 0.0  # Initialize standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      stdDevX += (i - meanX) ** 2 * coMatrix[i, j]  # Weighted sum of squared row differences.
      stdDevY += (j - meanY) ** 2 * coMatrix[i, j]  # Weighted sum of squared column differences.

  # Calculate the correlation of the co-occurrence matrix.
  correlation = 0.0  # Initialize correlation.
  stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
  stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      correlation += (
        (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)
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
    "StdDevX"      : stdDevX,  # Standard deviation of rows.
    "StdDevY"      : stdDevY,  # Standard deviation of columns.
  }


# ===========================================================================================
# Function(s) for calculating Gray-Level Run-Length Matrix (GLRLM) and its features.
# ===========================================================================================

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

  # Calculate minimum intensity value in the input matrix for intensity range adjustment.
  minA = np.min(matrix)
  # Calculate maximum intensity value in the input matrix for intensity range adjustment.
  maxA = np.max(matrix)
  # Determine total number of distinct gray levels by calculating intensity range span.
  N = maxA - minA + 1
  # Find maximum potential run length based on largest matrix dimension.
  R = np.max(matrix.shape)

  # Initialize empty GLRLM matrix with dimensions (intensity levels × max run length).
  rlMatrix = np.zeros((N, R))
  # Create tracking matrix to prevent duplicate processing of pixels in runs.
  seenMatrix = np.zeros(matrix.shape)
  # Calculate x-direction step using cosine (negative for coordinate system alignment).
  dx = int(np.round(np.cos(theta)))
  # Calculate y-direction step using sine of the analysis angle.
  dy = int(np.round(np.sin(theta)))

  # Adjust direction for specific angles to ensure consistent run direction.
  if (theta in [np.radians(45), np.radians(135)]):
    dx = -dx  # Adjust x-direction for 45 and 135 degrees.
    dy = dy  # Keep y-direction unchanged for 45 and 135 degrees.

  # Iterate through each row index of the input matrix.
  for i in range(matrix.shape[0]):
    # Iterate through each column index of the input matrix.
    for j in range(matrix.shape[1]):
      # Skip already processed pixels to prevent duplicate counting.
      if (seenMatrix[i, j] == 1):
        continue

      # Mark current pixel as processed in tracking matrix.
      seenMatrix[i, j] = 1
      # Store intensity value of current pixel for run comparison.
      currentPixel = matrix[i, j]
      # Initialize run length counter for current streak.
      d = 1

      # Explore consecutive pixels in specified direction until boundary or value change.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        # Check if subsequent pixel matches current intensity value.
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          # Mark matching pixel as processed in tracking matrix.
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          # Increment run length counter for continued streak.
          d += 1
        else:
          # Exit loop when streak breaks (different value encountered).
          break

      # Skip zero-value runs if configured to ignore background.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update GLRLM by incrementing count at corresponding intensity-runlength position.
      # (Adjust intensity index by minimum value for proper matrix positioning)
      rlMatrix[currentPixel - minA, d - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    # Add small epsilon to prevent division by zero in empty matrices.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  # Return computed Gray-Level Run-Length Matrix.
  return rlMatrix


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

  # Determine minimum intensity value in the original image.
  minA = np.min(image)
  # Determine maximum intensity value in the original image.
  maxA = np.max(image)
  # Calculate total number of distinct gray levels in the image.
  N = maxA - minA + 1
  # Get maximum possible run length from image dimensions.
  R = np.max(image.shape)

  # Calculate total number of runs recorded in the GLRLM.
  rlN = np.sum(rlMatrix)

  # Calculate Short Run Emphasis (SRE) emphasizing shorter runs through inverse squared weighting.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Long Run Emphasis (LRE) emphasizing longer runs through squared weighting.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Gray Level Non-Uniformity (GLN) measuring gray level distribution consistency.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Row sums squared
  ) / rlN

  # Calculate Run Length Non-Uniformity (RLN) measuring run length distribution consistency.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Column sums squared
  ) / rlN

  # Calculate Run Percentage (RP) indicating proportion of image occupied by runs.
  rp = rlN / np.prod(image.shape)

  # Calculate Low Gray Level Run Emphasis (LGRE) weighting low intensities more heavily.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Calculate High Gray Level Run Emphasis (HGRE) weighting high intensities more heavily.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Package computed features into a dictionary with descriptive keys.
  return {
    "Total Runs"                         : rlN,
    "Short Run Emphasis (SRE)"           : sre,
    "Long Run Emphasis (LRE)"            : lre,
    "Gray Level Non-Uniformity (GLN)"    : gln,
    "Run Length Non-Uniformity (RLN)"    : rln,
    "Run Percentage (RP)"                : rp,
    "Low Gray Level Run Emphasis (LGRE)" : lgre,
    "High Gray Level Run Emphasis (HGRE)": hgre,
  }


def CalculateGLRLM3DRunLengthMatrix(volume, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate 3D Gray-Level Run-Length Matrix (GLRLM) for volumetric texture analysis.

  Parameters:
      volume (numpy.ndarray): 3D array of intensity values (z, y, x dimensions)
      theta (float): Analysis angle in radians determining 3D direction vector
      isNorm (bool): Enable matrix normalization to probability distribution
      ignoreZeros (bool): Exclude zero-valued voxels from run calculations

  Returns:
      rlMatrix (numpy.ndarray): 2D matrix of size (intensity levels × max run length)
  """
  # Calculate intensity range parameters for matrix indexing.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels
  R = np.max(volume.shape)  # Maximum possible run length

  # Initialize empty GLRLM and pixel tracking matrix.
  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(volume.shape)

  # Calculate directional components using spherical coordinates.
  dx = int(np.round(np.cos(theta) * np.sin(theta)))  # X-axis step
  dy = int(np.round(np.sin(theta) * np.sin(theta)))  # Y-axis step
  dz = int(np.round(np.cos(theta)))  # Z-axis step

  # Iterate through all voxels in 3D volume.
  for i in range(volume.shape[0]):  # Z-dimension
    for j in range(volume.shape[1]):  # Y-dimension
      for k in range(volume.shape[2]):  # X-dimension
        # Skip previously processed voxels.
        if (seenMatrix[i, j, k] == 1):
          continue

        # Mark current voxel as processed.
        seenMatrix[i, j, k] = 1
        currentVal = volume[i, j, k]
        runLength = 1

        # Extend run along specified direction until value change.
        while (
          (i + runLength * dz >= 0) and
          (i + runLength * dz < volume.shape[0]) and
          (j + runLength * dy >= 0) and
          (j + runLength * dy < volume.shape[1]) and
          (k + runLength * dx >= 0) and
          (k + runLength * dx < volume.shape[2])
        ):
          if (volume[i + runLength * dz, j + runLength * dy, k + runLength * dx] == currentVal):
            seenMatrix[i + runLength * dz, j + runLength * dy, k + runLength * dx] = 1
            runLength += 1
          else:
            break

        # Skip zero-value runs if configured.
        if (ignoreZeros and currentVal == 0):
          continue

        # Update GLRLM with current run information.
        rlMatrix[currentVal - minA, runLength - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix


# ===========================================================================================
# Custom Function(s) for Preprocessing Special Tasks.
# ===========================================================================================


def PreprocessBrainTumorDatasetFigshare1512427(
  datasetPath,  # Path to the .mat file containing the image data.
  storagePath,  # Path to save the converted image.
  isResize=False,  # Flag to indicate whether to resize the image.
  newSize=(256, 256),  # New size for resizing the image if isResize is True.
  separateFolders=False,  # Flag to indicate whether to save images and masks in separate folders.
):
  """
  Preprocess the Brain Tumor Dataset from Figshare 1512427.
  Link: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

  Args:
      datasetPath (str): Path to the .mat files containing the images data.
      storagePath (str): Path to save the converted image.
      isResize (bool): Flag to indicate whether to resize the image.
      newSize (tuple): New size for resizing the image if isResize is True.
      separateFolders (bool): Flag to indicate whether to save images and masks in separate folders.

  Returns:
      None
  """
  # Install using pip install hdf5storage.
  import hdf5storage, os, tqdm  # Import necessary libraries.

  # Check if the datasetPath folder exists.
  if (not os.path.exists(datasetPath)):
    raise FileNotFoundError(f"The dataset path '{datasetPath}' does not exist.")

  # List all files in the dataset path.
  files = os.listdir(datasetPath)

  # Filter the files to include only .mat files.
  files = [file for file in files if file.endswith(".mat")]  # Keep only .mat files.

  # Key to access the image data in the loaded dictionary.
  key = "cjdata"

  # Create the storage path if it does not exist.
  os.makedirs(storagePath, exist_ok=True)  # Create the directory if it does not exist.

  # Loop through each file in the dataset.
  for file in tqdm.tqdm(files):
    # Construct the full path to the .mat file.
    filePath = os.path.join(datasetPath, file)

    # Load the image data from the .mat file.
    data = hdf5storage.loadmat(filePath)  # Load the .mat file.

    # Extract the data using the specified key.
    matData = data[key][0]

    # Extract the label (tumor type) from the data.
    label = str(int(matData[0][0].squeeze()))

    # Extract the image data from the loaded dictionary.
    imgData = matData[2]  # The image data is the third element in the array.

    # Extract the mask from the loaded dictionary.
    maskData = matData[4]  # The mask is the fifth element in the array.

    # Convert the image data to a NumPy array.
    imgData = np.array(imgData, dtype=np.float32)  # Convert to float32 for processing.
    # Convert the mask data to a NumPy array.
    maskData = np.array(maskData, dtype=np.float32)  # Convert to float32 for processing.

    # Find the min and max pixel values in the image.
    minImg, maxImg = np.min(imgData), np.max(imgData)
    # Normalize the image data to the range [0, 255].
    imgData = (imgData - minImg) / (maxImg - minImg) * 255.0  # Normalize to [0, 255].
    imgData = imgData.astype(np.uint8)  # Convert to uint8 for image representation.

    # Find the min and max pixel values in the mask.
    minMask, maxMask = np.min(maskData), np.max(maskData)
    # Normalize the mask data to the range [0, 255].
    maskData = (maskData - minMask) / (maxMask - minMask) * 255.0  # Normalize to [0, 255].
    maskData = cv2.threshold(maskData, 0, 255, cv2.THRESH_BINARY)[1]  # Convert to binary mask.
    maskData = maskData.astype(np.uint8)  # Convert to uint8 for binary mask representation.

    # If isResize is True, resize the image and mask to the new size.
    if (isResize):
      imgData = cv2.resize(imgData, newSize, interpolation=cv2.INTER_CUBIC)  # Resize the image.
      maskData = cv2.resize(maskData, newSize, interpolation=cv2.INTER_CUBIC)  # Resize the mask.

    # Get the base name of the file.
    fileName = os.path.basename(file)
    # Get the file name without extension.
    fileNameNoExt = os.path.splitext(fileName)[0]

    if (separateFolders):
      # Construct the full path for the image directory.
      imagesFolder = os.path.join(storagePath, "images", label)  # Create a directory for images.
      # Create the image directory if it does not exist.
      os.makedirs(imagesFolder, exist_ok=True)  # Create the directory for images.

      # Construct the full path for the mask directory.
      masksFolder = os.path.join(storagePath, "masks", label)  # Create a directory for masks.
      # Create the mask directory if it does not exist.
      os.makedirs(masksFolder, exist_ok=True)  # Create the directory for masks.

      # Construct the full path for the image and mask files.
      imagePath = os.path.join(imagesFolder, f"{fileNameNoExt}.png")  # Save the image as PNG.
      maskPath = os.path.join(masksFolder, f"{fileNameNoExt}.png")  # Save the mask as PNG.

    else:
      # Construct the full path for the label directory.
      labelPath = os.path.join(storagePath, label)  # Create a directory for the label.
      # Create the label directory if it does not exist.
      os.makedirs(labelPath, exist_ok=True)  # Create the directory for the label.

      # Construct the full path for the image and mask files.
      imagePath = os.path.join(labelPath, f"{fileNameNoExt}.png")  # Save the image as PNG.
      maskPath = os.path.join(labelPath, f"{fileNameNoExt}_mask.png")  # Save the mask as PNG.

    # Save the image and mask to the specified paths.
    cv2.imwrite(imagePath, imgData)  # Save the image.
    cv2.imwrite(maskPath, maskData)  # Save the mask.
