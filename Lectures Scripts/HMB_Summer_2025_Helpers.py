'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 21st, 2025
# Last Modification Date: Jul 7th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import sys  # For system-specific parameters and functions.
import cv2  # For image processing tasks.
import pandas as pd  # For data manipulation and analysis.
# Install the `shutup` library to suppress warnings: pip install shutup
import shutup  # To suppress the differ warnings.
import trimesh  # For 3D mesh operations and visualization.
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting and visualization.
# You can import specific classes from scikit-learn as needed.
from sklearn.preprocessing import *  # Import all preprocessing classes from scikit-learn.
from sklearn.metrics import *  # Import all metrics classes from scikit-learn.
from sklearn.model_selection import *  # Import all model selection classes from scikit-learn.

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 8)

# Suppress warnings from the `shutup` library.
shutup.please()  # Suppress all warnings from the `shutup` library.


# ===========================================================================================
# Function(s) for reading and preprocessing 3D medical imaging data.
# ===========================================================================================

def ReadVolume(caseImgPaths, caseSegPaths, raiseErrors=True):
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
      # Raise an error if the cropped image is empty and raiseErrors is True.
      if (raiseErrors):
        raise ValueError("The cropped image is empty. Please check the segmentation mask.")
      else:
        # If raiseErrors is False, skip the empty slice and continue processing.
        continue

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


def ReadVolumeSpecificClasses(caseImgPaths, caseSegPaths, specificClasses=[]):
  """
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Args:
      caseImgPaths (list): List of file paths to medical image slices in BMP format.
      caseSegPaths (list): List of file paths to segmentation masks matching the slices.
      specificClasses (list): List of specific classes to include in the segmentation.
        If empty, all classes are included.

  Returns:
      volumeCropped (numpy.ndarray): 3D array of preprocessed and aligned medical imaging data.
  """
  # Initialize empty list to store processed slices.
  volumeCropped = []

  # Process each image-segmentation pair in the input lists.
  for i in range(len(caseImgPaths)):
    # Verify both image and segmentation files exist before processing.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load grayscale medical image slice (8-bit depth).
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)
    # Load corresponding binary segmentation mask.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)

    # Check if specific classes are provided for segmentation.
    if (specificClasses):
      # Create a mask for the specific classes.
      mask = np.zeros_like(caseSeg)
      for classId in specificClasses:
        mask[caseSeg == classId] = 255
      caseSeg = mask

    # Extract region of interest using bitwise AND operation between image and mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Calculate bounding box coordinates of non-zero region in ROI.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop image to tight bounding box around segmented area.
    cropped = roi[y:y + h, x:x + w]

    # Validate cropped slice contains actual data (not just background).
    if (np.sum(cropped) <= 0):
      continue  # Skip empty slices.

    # Add processed slice to volume list.
    volumeCropped.append(cropped)

  # Check if any slices were successfully processed.
  if (len(volumeCropped) == 0):
    raise ValueError("No slices were successfully processed. Please check the input data.")

  # Determine maximum dimensions across all slices for padding alignment.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])

  # Standardize slice dimensions through symmetric padding.
  for i in range(len(volumeCropped)):
    # Calculate required padding for width and height dimensions.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]
    deltaHeight = maxHeight - volumeCropped[i].shape[0]

    # Apply padding to create uniform slice dimensions.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],
      deltaHeight // 2,  # Top padding (integer division)
      deltaHeight - deltaHeight // 2,  # Bottom padding (remainder)
      deltaWidth // 2,  # Left padding
      deltaWidth - deltaWidth // 2,  # Right padding
      cv2.BORDER_CONSTANT,  # Padding style (constant zero values)
      value=0
    )

    # Update volume with padded slice.
    volumeCropped[i] = padded.copy()

  # Convert list of 2D slices into 3D numpy array (z, y, x).
  volumeCropped = np.array(volumeCropped)

  return volumeCropped


def ExtractMultipleObjectsFromROI(
  caseImg, caseSeg,
  targetSize=(256, 256),
  cntAreaThreshold=0,
  sortByX=True,
):
  '''
  Extracts multiple objects from a region of interest (ROI) in a medical image.

  Parameters:
    caseImg (numpy.ndarray): The input medical image.
    caseSeg (numpy.ndarray): The segmentation mask indicating regions of interest.
    targetSize (tuple): The target size for resizing images.
    cntAreaThreshold (int): Minimum contour area to consider for extraction.
    sortByX (bool): Whether to sort extracted regions by their x-coordinate.

  Returns:
    list: A list of extracted regions from the image.
  '''

  # Resize images to standard dimensions using cubic interpolation for quality.
  caseImg = cv2.resize(caseImg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize scan image.
  caseSeg = cv2.resize(caseSeg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize mask image.

  # Binarize segmentation mask by thresholding to ensure only 0/255 values.
  caseSeg[caseSeg > 0] = 255  # Convert any positive values to pure white.

  # or you can apply:
  # caseSeg = cv2.threshold(caseSeg, 127, 255, cv2.THRESH_BINARY)[1]  # Binarize mask.

  # Perform sanity check on the segmentation mask to ensure valid content.
  if (np.sum(caseSeg) <= 0):  # Calculate sum of all pixel values in mask.
    # Raise error if mask contains no white pixels to prevent empty processing.
    raise ValueError("The mask is completely black/empty. Please check the segmentation mask.")

  # Detect contours in the segmentation mask using simple approximation method.
  contours = cv2.findContours(caseSeg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if (sortByX):  # Check if sorting by x-coordinate is requested.
    # Sort detected contours from left-to-right based on their x-coordinate.
    contours = sorted(contours[0], key=lambda x: cv2.boundingRect(x)[0], reverse=False)
  else:
    # If not sorting, just extract contours without sorting.
    contours = contours[0] if (len(contours) == 2) else contours[1]

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

  if (sortByX):  # Check if sorting by x-coordinate is requested.
    # Sort extracted regions by their x-coordinate to maintain left-to-right order.
    regions = sorted(regions, key=lambda x: cv2.boundingRect(x)[0], reverse=False)

  # Return the list of extracted regions for further processing.
  return regions


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


def FirstOrderFeatures2DV2(data, isNorm=True, ignoreZeros=True):
  """
  Calculate first-order statistical features from an image using a mask.

  Args:
      data (numpy.ndarray): The input image as a 2D NumPy array.
      isNorm (bool): Flag to indicate whether to normalize the histogram.
      ignoreZeros (bool): Flag to indicate whether to ignore zeros in the histogram.

  Returns:
      results (dict): A dictionary containing the calculated first-order features.
      hist2D (numpy.ndarray): The histogram of the pixel values in the region of interest.
  """

  # Calculate the histogram of the cropped ROI.
  minVal = int(np.min(data))  # Find the minimum pixel value in the cropped ROI.
  maxVal = int(np.max(data))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [minVal, maxVal].
  for i in range(minVal, maxVal + 1):
    hist2D.append(np.count_nonzero(data == i))  # Count occurrences of the value `i` in the cropped ROI.
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


def CalculateGLCMCooccuranceMatrix3D(volume, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True):
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
      coMatrix (numpy.ndarray): The calculated GLCM.
  """

  # Determine the number of unique intensity levels in the volume.
  minA = np.min(volume)  # Minimum intensity value.
  maxA = np.max(volume)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  noOfSlices = volume.shape[0]  # Number of slices in the volume.

  # Initialize the co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N))

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
          # Skip the calculation if the pixel values are zero.
          if ((volume[startLoc] == 0) or (volume[endLoc] == 0)):
            continue

        # (- minA) is added to work with matrices that does not start from 0.
        # Increment the count for the pair (start, end).
        # volume[startLoc] and volume[endLoc] are the intensity values at the start and end locations.
        startPixel = volume[startLoc] - minA  # Adjust start pixel value.
        endPixel = volume[endLoc] - minA  # Adjust end pixel value.

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
      # (Adjust intensity index by minimum value for proper matrix positioning).
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


def CalculateGLRLMRunLengthMatrix3D(volume, theta, isNorm=True, ignoreZeros=True):
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
        # (Adjust intensity index by minimum value for proper matrix positioning).
        rlMatrix[currentVal - minA, runLength - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix


# ===========================================================================================
# Function(s) for calculating Gray-Level Size-Zone Matrix (GLSZM) and its features.
# ===========================================================================================

def FindConnectedRegions(image, connectivity=4):
  """
  Finds connected regions in a 2D image based on pixel connectivity.

  Parameters:
      image (numpy.ndarray): A 2D NumPy array representing the input image.
                             Each element represents a pixel value.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 4: 4-connectivity (up, down, left, right).
                          - 8: 8-connectivity (includes diagonals).

  Returns:
      dict: A dictionary where keys are unique pixel values from the image,
            and values are lists of sets. Each set contains the coordinates
            (i, j) of pixels belonging to a connected region for that pixel value.
  """

  def RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity=4):
    """
    Recursive helper function to find all connected pixels for a given starting pixel.

    Parameters:
        i (int): Row index of the current pixel.
        j (int): Column index of the current pixel.
        currentPixel (int): The pixel value being processed.
        region (set): A set to store the coordinates of connected pixels.
        seenMatrix (numpy.ndarray): A 2D matrix to track visited pixels.
        connectivity (int): The type of connectivity (4 or 8).

    Returns:
        None: The function modifies the `region` and `seenMatrix` in place.
    """
    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if row index is out of bounds.
      (i >= image.shape[0]) or
      (j < 0) or
      (j >= image.shape[1]) or
      (image[i, j] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j) in region) or  # Check if the pixel has already been added to the region.
      (seenMatrix[i, j] == 1)  # Check if the pixel has already been seen.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j))
    seenMatrix[i, j] = 1

    # Recursively check the neighboring pixels (up, left, down, right).
    RecursiveHelper(i - 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j - 1, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i + 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j + 1, currentPixel, region, seenMatrix, connectivity)

    # If 8-connectivity is specified, also check diagonal neighbors.
    if (connectivity == 8):
      RecursiveHelper(i - 1, j - 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i - 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j - 1, currentPixel, region, seenMatrix, connectivity)

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(image.shape)

  # Dictionary to store regions grouped by pixel values.
  regions = {}

  # Iterate over each pixel in the image.
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      # Skip if the pixel has already been processed.
      if (seenMatrix[i, j]):
        continue

      # Get the current pixel value.
      currentPixel = image[i, j]

      # Initialize a list for this pixel value if it doesn't exist.
      if (currentPixel not in regions):
        regions[currentPixel] = []

      # Initialize a new region set for the current pixel.
      region = set()

      # Use the helper function to find all connected pixels.
      RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity)

      # Add the region to the dictionary if it contains any pixels.
      if (len(region) > 0):
        regions[currentPixel].append(region)

  # Return the dictionary of regions.
  return regions


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4, isNorm=False, ignoreZeros=False):
  """
  Calculate the Size-Zone Matrix for a given image based on connected regions.

  Parameters:
      image (numpy.ndarray): A 2D NumPy array representing the input image.
                             Each element represents a pixel value.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 4: 4-connectivity (up, down, left, right).
                          - 8: 8-connectivity (includes diagonals).
      isNorm (bool): Whether to normalize the size-zone matrix.
      ignoreZeros (bool): Whether to ignore zero pixel values.

  Returns:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      szDict (dict): A dictionary where keys are unique pixel values from the image,
                      and values are lists of sets. Each set contains the coordinates
                      (i, j) of pixels belonging to a connected region for that pixel value.
      N (int): The number of unique pixel values in the image.
      Z (int): The maximum size of any region in the dictionary.
  """

  if (image.ndim != 2):
    raise ValueError("The input image must be a 2D array.")

  if (connectivity not in [4, 8]):
    raise ValueError("Connectivity must be either 4 or 8.")

  if (image.size == 0):
    raise ValueError("The input image is empty.")

  if (np.max(image) == 0):
    raise ValueError("The input image is completely black.")

  # Find connected regions in the image.
  szDict = FindConnectedRegions(image, connectivity=connectivity)

  # Determine the number of unique pixel values in the image.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  # Find the maximum size of any region in the dictionary.
  # By iterating over all zones of all pixel values and getting the length of the largest zone.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentPixel, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Increment the count for the corresponding pixel value and region size.
      # (Adjust intensity index by minimum value for proper matrix positioning).
      szMatrix[currentPixel - minA, len(zone) - 1] += 1

  szMatrixSum = np.sum(szMatrix)

  if (szMatrixSum == 0):
    return szMatrix, szDict, N, Z

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize the size-zone matrix.
    # Add small epsilon to avoid division by zero.
    szMatrix = szMatrix / (np.sum(szMatrix) + 1e-6)

  return szMatrix, szDict, N, Z


def CalculateGLSZMFeatures(szMatrix, data, N, Z):
  """
  Calculate the features of the Size-Zone Matrix (GLSZM).

  Parameters:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      N (int): The number of unique pixel values in the image.
      Z (int): The maximum size of any region in the dictionary.

  Returns:
      dict: A dictionary containing the calculated features.
  """
  # Calculate the total number of zones in the size-zone matrix.
  # Sum all values in the size-zone matrix to get the total zone count.
  szN = np.sum(szMatrix)

  # Small Zone Emphasis.
  sze = np.sum(
    szMatrix / ((np.arange(1, Z + 1) ** 2) + 1e-10),  # Divide each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Emphasis.
  lze = np.sum(
    szMatrix * ((np.arange(1, Z + 1) ** 2) + 1e-10),  # Multiply each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Gray Level Non-Uniformity.
  gln = np.sum(
    np.sum(szMatrix, axis=1) ** 2,  # Sum each row and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Non-Uniformity.
  zsn = np.sum(
    np.sum(szMatrix, axis=0) ** 2,  # Sum each column and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Percentage.
  # Divide the total number of zones by the total number of pixels.
  zp = szN / np.prod(data.shape)

  # Gray Level Variance.
  glv = np.sum(
    # Compute variance for each gray level.
    (np.sum(szMatrix, axis=1)) *
    ((np.arange(1, N + 1) - np.mean(np.arange(1, N + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Variance.
  zsv = np.sum(
    # Compute variance for zone sizes.
    (np.sum(szMatrix, axis=0)) *
    ((np.arange(1, Z + 1) - np.mean(np.arange(1, Z + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Entropy.
  log = np.log2(szMatrix + 1e-10)  # Compute log base 2 of the size-zone matrix.
  log[log == -np.inf] = 0  # Replace -inf with 0.
  log[log < 0] = 0  # Replace negative values with 0.
  zse = np.sum(
    # Compute entropy for zone sizes.
    szMatrix * log,
  ) / szN  # Normalize by the total number of zones.

  # Low Gray Level Zone Emphasis.
  lgze = np.sum(
    # Divide each gray level by its squared value.
    szMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # High Gray Level Zone Emphasis.
  hgze = np.sum(
    # Multiply each gray level by its squared value.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone Low Gray Level Emphasis.
  # Adding 1e-10 to avoid division by zero.
  szlge = np.sum(
    # Combine small zone and low gray level emphasis.
    szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + 1e-10),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone High Gray Level Emphasis.
  szhge = np.sum(
    # Combine small zone and high gray level emphasis.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / ((np.arange(1, Z + 1) ** 2) + 1e-10),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Low Gray Level Emphasis.
  lzgle = np.sum(
    # Combine large zone and low gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone High Gray Level Emphasis.
  lzhge = np.sum(
    # Combine large zone and high gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  return {
    "Small Zone Emphasis (SZE)"                  : sze,
    "Large Zone Emphasis (LZE)"                  : lze,
    "Gray Level Non-Uniformity (GLN)"            : gln,
    "Zone Size Non-Uniformity (ZSN)"             : zsn,
    "Zone Percentage (ZP)"                       : zp,
    "Gray Level Variance (GLV)"                  : glv,
    "Zone Size Variance (ZSV)"                   : zsv,
    "Zone Size Entropy (ZSE)"                    : zse,
    "Low Gray Level Zone Emphasis (LGZE)"        : lgze,
    "High Gray Level Zone Emphasis (HGZE)"       : hgze,
    "Small Zone Low Gray Level Emphasis (SZLGE)" : szlge,
    "Small Zone High Gray Level Emphasis (SZHGE)": szhge,
    "Large Zone Low Gray Level Emphasis (LZGLE)" : lzgle,
    "Large Zone High Gray Level Emphasis (LZHGE)": lzhge,
  }


def FindConnectedRegions3D(volume, connectivity=6):
  """
  Finds connected regions in a 3D volume based on pixel connectivity.
  Parameters:
      volume (numpy.ndarray): A 3D NumPy array representing the input volume.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 6: 6-connectivity (faces only).
                          - 26: 26-connectivity (faces, edges, and corners).
  Returns:
      dict: A dictionary where keys are unique pixel values from the volume,
            and values are lists of sets. Each set contains the coordinates
            (i, j, k) of pixels belonging to a connected region for that pixel value.
  """

  def RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity=6):
    """
    Recursive helper function to find all connected pixels for a given starting pixel.
    Parameters:
        i (int): Z-axis index of the current pixel.
        j (int): Y-axis index of the current pixel.
        k (int): X-axis index of the current pixel.
        currentPixel (int): The pixel value being processed.
        region (set): A set to store the coordinates of connected pixels.
        seenMatrix (numpy.ndarray): A 3D matrix to track visited pixels.
        connectivity (int): The type of connectivity (6 or 26).
    Returns:
        None: The function modifies the `region` and `seenMatrix` in place.
    """
    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if Z-axis index is out of bounds.
      (i >= volume.shape[0]) or
      (j < 0) or  # Check if Y-axis index is out of bounds.
      (j >= volume.shape[1]) or
      (k < 0) or  # Check if X-axis index is out of bounds.
      (k >= volume.shape[2]) or
      (volume[i, j, k] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j, k) in region) or  # Check if the pixel has already been added to the region.
      (seenMatrix[i, j, k] == 1)  # Check if the pixel has already been seen.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j, k))  # Add the pixel coordinates to the region set.
    seenMatrix[i, j, k] = 1  # Mark the pixel as seen.

    # Recursively check the neighboring pixels (faces only for 6-connectivity).
    RecursiveHelper(i - 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor below.
    RecursiveHelper(i + 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor above.
    RecursiveHelper(i, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor left.
    RecursiveHelper(i, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor right.
    RecursiveHelper(i, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor behind.
    RecursiveHelper(i, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor front.

    # If 26-connectivity is specified, also check diagonal neighbors (edges and corners).
    if (connectivity == 26):
      # k is fixed => same slice (4 pixels).
      RecursiveHelper(i - 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

      # k - 1 => pre-slice (8 pixels).
      RecursiveHelper(i - 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

      # k + 1 => post-slice (8 pixels).
      RecursiveHelper(i - 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(volume.shape)  # Create a 3D matrix of zeros.

  # Dictionary to store regions grouped by pixel values.
  regions = {}  # Keys are pixel values, values are lists of sets.

  # Iterate over each voxel in the volume.
  for i in range(volume.shape[0]):  # Loop over Z-axis.
    for j in range(volume.shape[1]):  # Loop over Y-axis.
      for k in range(volume.shape[2]):  # Loop over X-axis.
        # Skip if the voxel has already been processed.
        if (seenMatrix[i, j, k]):
          continue  # Skip already processed voxels.

        # Get the current voxel value.
        currentPixel = volume[i, j, k]  # Retrieve the intensity value of the voxel.

        # Initialize a list for this pixel value if it doesn't exist.
        if (currentPixel not in regions):
          regions[currentPixel] = []  # Create a new list for this intensity value.

        # Initialize a new region set for the current voxel.
        region = set()  # Create an empty set to store connected voxel coordinates.

        # Use the helper function to find all connected voxels.
        RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity)  # Find connected region.

        # Add the region to the dictionary if it contains any voxels.
        if (len(region) > 0):
          regions[currentPixel].append(region)  # Append the region to the list for this intensity value.

  # Return the dictionary of regions.
  return regions  # Return the dictionary containing connected regions.


def CalculateGLSZMSizeZoneMatrix3D(volume, connectivity=6, isNorm=True, ignoreZeros=True):
  """
  Calculate the Size-Zone Matrix for a 3D volume based on connected regions.
  Parameters:
      volume (numpy.ndarray): A 3D NumPy array representing the input volume.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 6: 6-connectivity (faces only).
                          - 26: 26-connectivity (faces, edges, and corners).
      isNorm (bool): Whether to normalize the size-zone matrix.
      ignoreZeros (bool): Whether to ignore zero pixel values.
  Returns:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      szDict (dict): A dictionary where keys are unique pixel values from the volume,
                     and values are lists of sets. Each set contains the coordinates
                     (i, j, k) of pixels belonging to a connected region for that pixel value.
      N (int): The number of unique pixel values in the volume.
      Z (int): The maximum size of any region in the dictionary.
  """

  if (volume.ndim != 3):
    raise ValueError("The input volume must be a 3D array.")

  if (connectivity not in [6, 26]):
    raise ValueError("Connectivity must be either 6 or 26.")

  if (volume.size == 0):
    raise ValueError("The input volume is empty.")

  if (np.max(volume) == 0):
    raise ValueError("The input volume is completely black.")

  # Find connected regions in the volume.
  szDict = FindConnectedRegions3D(volume, connectivity=connectivity)  # Identify connected regions.

  # Determine the number of unique pixel values in the volume.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels

  # Find the maximum size of any region in the dictionary.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])  # Find the largest connected region size.

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))  # Create a 2D matrix to store size-zone counts.

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentVal, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentVal == 0)):
        continue  # Skip zero-valued regions if ignoreZeros is True.

      # Increment the count for the corresponding pixel value and region size.
      szMatrix[currentVal - minA, len(zone) - 1] += 1  # Update the size-zone matrix.

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize by total sum to avoid division by zero.
    szMatrix = szMatrix / (np.sum(szMatrix) + 1e-6)

    # Return the size-zone matrix, dictionary, and metadata.
  return szMatrix, szDict, N, Z  # Return the computed outputs.


# ===========================================================================================
# Function(s) for handling the local binary patterns (LBP).
# ===========================================================================================
def BuildLBPKernel(
  distance=1,  # Distance parameter to determine the size of the kernel.
  theta=135,  # Angle parameter to rotate the kernel (default is 135 degrees).
  isClockwise=False,  # Direction of rotation (False means counterclockwise).
):
  """
  Build a kernel matrix for Local Binary Pattern (LBP) computation.
  The kernel is generated based on the specified distance and angle (theta).
  The kernel is a square matrix of size (2 * distance + 1) x (2 * distance + 1).
  The kernel is filled with powers of 2, representing the weights of the pixels
  in the LBP computation. The kernel is rotated by the specified angle (theta)
  in a clockwise or counterclockwise direction.

  Args:
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the kernel rotation.
    isClockwise (bool): Direction of rotation (True for clockwise, False for counterclockwise).

  Returns:
    np.ndarray: A kernel matrix representing the LBP pattern weights.
  """

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")

  # Calculate the total number of elements on the edges.
  noOfElements = 8 * distance  # Total number of edge elements is 8 * distance.

  # Calculate the angle between consecutive elements.
  angle = 360.0 / float(noOfElements)  # Divide 360 degrees by the total number of edge elements.

  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")

  # Check if the angle (theta) is not a multiple of (angle) degrees, raising a ValueError if true.
  if (theta % angle != 0):
    raise ValueError("Theta must be a multiple of the angle between elements.")

  # Calculate the size of the matrix.
  n = 2 * distance + 1  # The size of the kernel is (2 * distance + 1) x (2 * distance + 1).

  # Initialize the matrix with zeros.
  kernel = np.zeros((n, n), dtype=np.uint32)  # Create a zero-filled matrix of size n x n.

  # Generate the coordinates for the edges of the kernel in a clockwise order.
  coordinates = []  # List to store the edge coordinates of the kernel.

  # Add coordinates for the leftmost column (top to bottom).
  for row in range(0, n):  # Iterate over rows from top to bottom.
    coordinates.append((row, 0))  # Append (row, 0) for the leftmost column.

  # Add coordinates for the bottommost row (left to right).
  for col in range(0, n):  # Iterate over columns from left to right.
    coordinates.append((n - 1, col))  # Append (n-1, col) for the bottommost row.

  # Add coordinates for the rightmost column (bottom to top).
  for row in range(n - 1, -1, -1):  # Iterate over rows from bottom to top.
    coordinates.append((row, n - 1))  # Append (row, n-1) for the rightmost column.

  # Add coordinates for the topmost row (right to left).
  for col in range(n - 1, -1, -1):  # Iterate over columns from right to left.
    coordinates.append((0, col))  # Append (0, col) for the topmost row.

  # Remove the repeated coordinates.
  for i in range(len(coordinates) - 1, 0, -1):  # Iterate from the end to the beginning.
    if (coordinates[i] == coordinates[i - 1]):  # Check if the current coordinate is equal to the previous one.
      coordinates.pop(i)  # Remove the current coordinate if it is a duplicate.
  # Remove the last coordinate if it is equal to the first one.
  if (coordinates[-1] == coordinates[0]):  # Check if the last coordinate is equal to the first one.
    coordinates.pop(-1)  # Remove the last coordinate if it is a duplicate.

  # Calculate the shift required to rotate the kernel by the given theta.
  thetaShift = int((theta - 135) / angle)  # Determine how many positions to shift based on theta.

  # Rotate the coordinates list by thetaShift positions.
  coordinates = coordinates[thetaShift:] + coordinates[:thetaShift]  # Shift the coordinates list.

  # If the rotation direction is clockwise, rotate the kernel counterclockwise.
  if (isClockwise):
    # Reverse the order of coordinates except the first one.
    coordinates = [coordinates[0]] + coordinates[1:][::-1]

  # Assign powers of 2 to the edge elements in the kernel.
  counter = 0  # Counter to track the current power of 2.

  # Iterate through the shifted coordinates and assign values to the kernel.
  for i in range(len(coordinates)):  # Loop through all edge coordinates.
    x = coordinates[i][0]  # Extract the x-coordinate.
    y = coordinates[i][1]  # Extract the y-coordinate.
    if (kernel[y, x] == 0):  # Check if the position is still zero (not yet assigned).
      kernel[y, x] = 2 ** counter  # Assign 2^counter to the current position.
      counter += 1  # Increment the counter for the next power of 2.

  # Transpose the kernel to match the expected orientation.
  kernel = kernel.T

  return kernel  # Return the final kernel matrix.


def LocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  """
  Compute the Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The LBP is a texture descriptor that encodes local patterns in the image,
  making it useful for various image analysis tasks.

  Args:
    matrix (np.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    np.ndarray: LBP matrix with the same shape as the input image, containing LBP values.
  """

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")
  # Check if the distance exceeds half of the image dimensions, raising a ValueError if true.
  if (distance > matrix.shape[0] // 2 or distance > matrix.shape[1] // 2):
    raise ValueError("Distance must be less than half of the matrix dimensions.")
  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")

  # Calculate the total number of elements on the edges.
  noOfElements = 8 * distance  # Total number of edge elements is 8 * distance.

  # Calculate the angle between consecutive elements.
  angle = 360.0 / float(noOfElements)  # Divide 360 degrees by the total number of edge elements.

  # Check if the angle (theta) is not a multiple of {angle} degrees, raising a ValueError if true.
  if (theta % angle != 0):
    raise ValueError(f"Theta must be a multiple of {angle} degrees.")

  # Calculate the size of the kernel window based on the distance parameter.
  windowSize = distance * 2 + 1
  # Determine the center coordinates of the kernel window.
  centerX = windowSize // 2
  centerY = windowSize // 2

  # Build the LBP kernel using the specified parameters.
  kernel = BuildLBPKernel(
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
  )

  # Initialize an empty matrix to store the computed LBP values.
  lbpMatrix = np.zeros(matrix.shape, dtype=np.uint32)

  # Pad the input matrix with zeros to handle boundary conditions during convolution.
  paddedA = np.pad(matrix, distance, mode="constant", constant_values=0)

  # Iterate through each pixel in the input matrix to compute its LBP value.
  for y in range(distance, matrix.shape[0] + distance):
    for x in range(distance, matrix.shape[1] + distance):
      # Extract the region of interest (ROI) around the current pixel.
      region = paddedA[
               y - distance:y + distance + 1,
               x - distance:x + distance + 1
               ]
      # Compare each pixel in the ROI with the center pixel to create a binary mask.
      comp = region >= region[centerY, centerX]
      # Compute the LBP value for the current pixel by summing the weighted kernel values.
      lbpMatrix[y - distance, x - distance] = np.sum(kernel[comp])

  # Normalize the LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum LBP values in the matrix.
    minValue = np.min(lbpMatrix)
    maxValue = np.max(lbpMatrix)
    # Normalize the LBP values to the range [0, 255].
    lbpMatrix = ((lbpMatrix - minValue) / (maxValue - minValue) * 255)
    # Ensure the LBP matrix is of type uint8.
    lbpMatrix = lbpMatrix.astype(np.uint8)

  # Return the computed LBP matrix.
  return lbpMatrix


def UniformLocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  """
  Compute the Uniform Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The Uniform LBP is a variant of LBP that focuses on uniform patterns,
  making it useful for texture analysis and classification tasks.
  The uniform patterns are defined as those with at most two transitions
  between 0 and 1 in the binary representation of the LBP value.

  Args:
    matrix (np.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    np.ndarray: Uniform LBP matrix with the same shape as the input image, containing LBP values.
  """

  # Run the standard LBP function to get the LBP matrix.
  lbpMatrix = LocalBinaryPattern2D(
    matrix,
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
    normalizeLBP=False,  # No need to normalize here.
  )

  # Initialize an empty matrix to store the uniform LBP values.
  uniformMatrix = np.zeros(matrix.shape, dtype=np.uint32)

  # Iterate through each pixel in the LBP matrix to compute uniform LBP values.
  for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
      # Convert the LBP value to binary representation with 8 * distance bits.
      binary = np.binary_repr(
        lbpMatrix[y, x],
        width=8 * distance,
      )
      # Count the number of transitions (0 to 1 or 1 to 0) in the binary representation.
      transitions = 0
      for i in range(1, len(binary)):
        # Count transitions between consecutive bits.
        if (binary[i] != binary[i - 1]):
          transitions += 1

      # If the number of transitions is less than or equal to 2, assign the LBP value.
      if (transitions <= 2):
        # Assign the LBP value to the uniform matrix.
        uniformMatrix[y, x] = int(binary, 2)

  # Normalize the uniform LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum uniform LBP values in the matrix.
    minValue = np.min(uniformMatrix)
    maxValue = np.max(uniformMatrix)
    # Normalize the uniform LBP values to the range [0, 255].
    uniformMatrix = ((uniformMatrix - minValue) / (maxValue - minValue) * 255).astype(np.uint8)

  # Ensure the uniform LBP matrix is of type uint8.
  uniformMatrix = uniformMatrix.astype(np.uint8)

  # Return the computed uniform LBP matrix.
  return uniformMatrix


# ===========================================================================================
# Function(s) for calculating shape features.
# ===========================================================================================
def ShapeFeatures2D(matrix):
  """
  Calculate shape features of a given binary matrix in 2D.
  The function computes various shape features such as area, perimeter,
  centroid, bounding box, aspect ratio, compactness, eccentricity,
  convex hull area, extent, solidity, major and minor axis lengths,
  orientation, and roundness.
  Args:
    matrix: A matrix representing the binary image or segmented region.
  Returns:
    A dictionary containing the calculated shape features.
  """

  # Check if the input matrix is empty or not.
  if (matrix is None or matrix.size == 0):
    # Raise error if the matrix is empty.
    raise ValueError("The input matrix is empty. Please provide a valid matrix.")

  # Calculate the Shape Features:

  # 1. Area.
  # Counts the number of non-zero pixels in the cropped image.
  area = cv2.countNonZero(matrix)

  # 2. Perimeter.
  # Finds contours in the matrix image.
  contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Identifies the largest contour.
  largestContour = max(contours, key=cv2.contourArea)
  # Calculates the perimeter of the largest contour.
  perimeter = cv2.arcLength(largestContour, True)

  # 3. Centroid.
  # Computes moments of the matrix.
  moments = cv2.moments(matrix)
  # Calculates the X-coordinate of the centroid.
  centroidX = int(moments["m10"] / moments["m00"])
  # Calculates the Y-coordinate of the centroid.
  centroidY = int(moments["m01"] / moments["m00"])

  # 4. Bounding Box.
  # Recalculates the bounding box for the matrix.
  x, y, w, h = cv2.boundingRect(matrix)

  # 5. Aspect Ratio.
  # Computes the aspect ratio of the bounding box.
  aspectRatio = w / h

  # 6. Compactness.
  # Calculates compactness using perimeter and area.
  compactness = (perimeter ** 2) / (4 * np.pi * area)

  # 7. Eccentricity.
  # Computes normalized second-order moment mu20.
  mu20 = moments["mu20"] / moments["m00"]
  # Computes normalized second-order moment mu02.
  mu02 = moments["mu02"] / moments["m00"]
  # Calculates eccentricity based on moments.
  eccentricity = np.sqrt(1 - (mu02 / mu20))

  # 8. Convex Hull.
  # Finds the convex hull of the largest contour.
  smallestConvexHull = cv2.convexHull(largestContour)
  # Calculates the area of the convex hull.
  convexHullArea = cv2.contourArea(smallestConvexHull)

  # 9. Extent (or Rectangularity).
  # Computes the extent as the ratio of contour area to bounding box area.
  extent = area / (w * h)

  # 10. Solidity.
  # Calculates solidity as the ratio of contour area to convex hull area.
  solidity = area / convexHullArea

  # 11. Major Axis Length.
  # Computes the length of the major axis using the second-order moment mu20.
  majorAxisLength = 2 * np.sqrt(moments["m20"] / moments["m00"])

  # 12. Minor Axis Length.
  # Computes the length of the minor axis using the second-order moment mu02.
  minorAxisLength = 2 * np.sqrt(moments["m02"] / moments["m00"])

  # 13. Orientation.
  # Calculates orientation angle as the angle of the major axis of the ellipse.
  orientation = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

  # 14. Roundness.
  # Computes roundness based on area and perimeter.
  roundness = (4 * area) / (np.pi * perimeter ** 2)

  # 15. Symmetry.
  # Flip the matrix horizontally and vertically.
  flippedHorizontal = np.fliplr(matrix)
  flippedVertical = np.flipud(matrix)
  # Calculate the symmetry score for horizontal flipping.
  horizontalSymmetry = np.sum(matrix == flippedHorizontal) / area
  # Calculate the symmetry score for vertical flipping.
  verticalSymmetry = np.sum(matrix == flippedVertical) / area
  # Calculate the average symmetry score.
  symmetry = (horizontalSymmetry + verticalSymmetry) / 2.0

  # 16. Elongation.
  # Calculate the elongation based on the major and minor axis lengths.
  elongation = majorAxisLength / minorAxisLength

  # 17. Thinness Ratio.
  # Calculate the thinness ratio based on the perimeter and area.
  thinnessRatio = np.power(perimeter, 2) / area

  # 18. Convexity.
  # Convexity measures how close the shape is to being convex.
  # It is the ratio of the perimeter of the convex hull to the perimeter of the shape.
  # True: The contour is closed, False: The contour is open.
  convexHullPerimeter = cv2.arcLength(smallestConvexHull, True)
  convexity = convexHullPerimeter / perimeter

  # 19. Sparseness.
  # Sparseness measures how "spread out" the shape is.
  # Calculate the area of the bounding box.
  boundingBoxArea = w * h
  # Compute sparseness as a measure of spread.
  sparseness = (np.sqrt(area / boundingBoxArea) - (area / boundingBoxArea))

  # 20. Curvature.
  # Curvature measures how sharply the contour bends at each point.
  curvatures = []
  for i in range(len(largestContour)):
    # Loop through all points in the largest contour.
    p1 = largestContour[i - 1][0]  # Previous point.
    p2 = largestContour[i][0]  # Current point.
    p3 = largestContour[(i + 1) % len(largestContour)][0]  # Next point.
    # Calculate the curvature using the cross product and dot product.
    v1 = p2 - p1  # Vector from p1 to p2.
    v2 = p3 - p2  # Vector from p2 to p3.
    crossProduct = np.cross(v1, v2)  # Cross product of the vectors.
    dotProduct = np.dot(v1, v2)  # Dot product of the vectors.
    angle = np.arctan2(crossProduct, dotProduct)  # Angle between the vectors.
    curvatures.append(angle)  # Append the curvature to the list.
  # Calculate the average curvature.
  averageCurvature = np.mean(curvatures)
  # Calculate the standard deviation of curvature.
  stdCurvature = np.std(curvatures)

  # Return all calculated features as a dictionary.
  return {
    "Area"               : area,
    "Perimeter"          : perimeter,
    "Centroid X"         : centroidX,
    "Centroid Y"         : centroidY,
    "Bounding Box X"     : x,
    "Bounding Box Y"     : y,
    "Bounding Box W"     : w,
    "Bounding Box H"     : h,
    "Aspect Ratio"       : aspectRatio,
    "Compactness"        : compactness,
    "Eccentricity"       : eccentricity,
    "Convex Hull Area"   : convexHullArea,
    "Extent"             : extent,
    "Solidity"           : solidity,
    "Major Axis Length"  : majorAxisLength,
    "Minor Axis Length"  : minorAxisLength,
    "Orientation"        : orientation,
    "Roundness"          : roundness,
    "Horizontal Symmetry": horizontalSymmetry,
    "Vertical Symmetry"  : verticalSymmetry,
    "Symmetry"           : symmetry,
    "Elongation"         : elongation,
    "Thinness Ratio"     : thinnessRatio,
    "Convexity"          : convexity,
    "Sparseness"         : sparseness,
    "Curvature"          : averageCurvature,
    "Std Curvature"      : stdCurvature,
  }


def ShapeFeatures3D(volume):
  """
  Calculate 3D shape features of a given binary or labeled volume.
  The function computes various geometric and topological properties such as volume,
  surface area, compactness, sphericity, elongation, flatness, rectangularity,
  spherical disproportion, and Euler number. These features are derived from the
  mesh representation of the input volume using marching cubes.

  Args:
    volume (numpy.ndarray): A 3D binary or labeled matrix representing the object.

  Returns:
    dict: A dictionary containing the calculated 3D shape features.
  """

  # Converts an (n, m, p) matrix into a mesh, using marching_cubes.
  # Marching cubes algorithm generates a triangular mesh from the volume data.
  mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volume)

  # 1. Volume.
  # Computes the total number of non-zero voxels in the volume.
  volume = np.sum(volume)

  # 2. Surface Area.
  # Calculates the total surface area of the mesh generated by marching cubes.
  surfaceArea = mesh.area

  # 3. Surface to Volume Ratio.
  # Measures the ratio of surface area to volume, indicating compactness.
  surfaceToVolumeRatio = surfaceArea / volume

  # 4. Compactness.
  # Quantifies how closely the shape resembles a sphere, based on volume and surface area.
  compactness = (volume ** (2 / 3)) / (6 * np.sqrt(np.pi) * surfaceArea)

  # 5. Sphericity.
  # Measures how spherical the shape is, normalized by volume and surface area.
  sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surfaceArea

  # Bounding Box.
  # Computes the bounding box of the mesh and extracts its dimensions.
  bbox = mesh.bounding_box.bounds
  Lmax = np.max(bbox[1] - bbox[0])  # Maximum length of the bounding box.
  Lmin = np.min(bbox[1] - bbox[0])  # Minimum length of the bounding box.
  Lint = np.median(bbox[1] - bbox[0])  # Intermediate length of the bounding box.

  # 6. Elongation.
  # Measures the ratio of the longest dimension to the shortest dimension of the bounding box.
  elongation = Lmax / Lmin

  # 7. Flatness.
  # Measures the ratio of the shortest dimension to the intermediate dimension of the bounding box.
  flatness = Lmin / Lint

  # 8. Rectangularity.
  # Measures how efficiently the shape fills its bounding box, as the ratio of volume to bounding box volume.
  bboxVolume = np.prod(bbox[1] - bbox[0])  # Volume of the bounding box.
  rectangularity = volume / bboxVolume

  # 9. Euler Number.
  # Represents the topological characteristic of the shape, computed from the mesh.
  eulerNumber = mesh.euler_number

  # Return all calculated features as a dictionary.
  return {
    "Volume"                 : volume,
    "Surface Area"           : surfaceArea,
    "Surface to Volume Ratio": surfaceToVolumeRatio,
    "Compactness"            : compactness,
    "Sphericity"             : sphericity,
    "Elongation"             : elongation,
    "Flatness"               : flatness,
    "Rectangularity"         : rectangularity,
    "Euler Number"           : eulerNumber
  }


# ===========================================================================================
# Function(s) for calculating performance.
# ===========================================================================================

def CalculatePerformanceMetrics(
  confMatrix,
  eps=1e-10,
  addWeightedAverage=False,
):
  """
  Calculate performance metrics from a confusion matrix.

  Parameters:
  confMatrix (list of list): Confusion matrix as a nested list.
  eps (float): Small value to avoid division by zero.
  addWeightedAverage (bool): Whether to include weighted averages in the output.

  Returns:
  dict: A dictionary containing performance metrics including:
        - True Positives (TP)
        - False Positives (FP)
        - False Negatives (FN)
        - True Negatives (TN)
        - Macro Precision
        - Macro Recall
        - Macro F1
        - Macro Accuracy
        - Macro Specificity
        - Micro Precision
        - Micro Recall
        - Micro F1
        - Micro Accuracy
        - Micro Specificity
        - Weights
        - Weighted Precision
        - Weighted Recall
        - Weighted F1
        - Weighted Accuracy
        - Weighted Specificity
  """
  # Convert the confusion matrix to a NumPy array for easier manipulation.
  confMatrix = np.array(confMatrix)

  # Check if the confusion matrix is for binary classification.
  noOfClasses = confMatrix.shape[0]
  if (noOfClasses > 2):
    # Calculate True Positives (TP) as the diagonal elements of the confusion matrix.
    TP = np.diag(confMatrix)
    # Calculate False Positives (FP) as the sum of each column minus the TP.
    FP = np.sum(confMatrix, axis=0) - TP
    # Calculate False Negatives (FN) as the sum of each row minus the TP.
    FN = np.sum(confMatrix, axis=1) - TP
    # Calculate True Negatives (TN) as the total sum of the matrix minus TP, FP, and FN.
    TN = np.sum(confMatrix) - (TP + FP + FN)
  else:
    # For binary classification, the confusion matrix is a 2x2 matrix.
    # Unravel the confusion matrix to get the TP, FP, FN, and TN.
    # The order of the elements is TN, FP, FN, TP.
    TN, FP, FN, TP = confMatrix.ravel()

  # Avoid division by zero by adding a small epsilon value.
  TP = TP + eps
  FP = FP + eps
  FN = FN + eps
  TN = TN + eps

  # Create a dictionary to hold the calculated performance metrics and
  # the TP, FP, FN, TN vectors.
  metrics = {
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
  }

  # Calculate precision using macro averaging: mean of TP / (TP + FP) for each class.
  precision = np.mean(TP / (TP + FP))
  # Calculate recall using macro averaging: mean of TP / (TP + FN) for each class.
  recall = np.mean(TP / (TP + FN))
  # Calculate F1 score using macro averaging: harmonic mean of precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using macro averaging: sum of TP and TN divided by the total sum of the matrix.
  accuracy = np.mean(TP + TN) / np.sum(confMatrix)
  # Calculate specificity using macro averaging: mean of TN / (TN + FP) for each class.
  specificity = np.mean(TN / (TN + FP))

  metrics.update({
    "Macro Precision"  : precision,
    "Macro Recall"     : recall,
    "Macro F1"         : f1,
    "Macro Accuracy"   : accuracy,
    "Macro Specificity": specificity,
  })

  # If requested, calculate the macro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Macro Average": avg,
    })

  # Calculate precision using micro averaging: sum of TP divided by the sum of TP and FP.
  precision = np.sum(TP) / np.sum(TP + FP)
  # Calculate recall using micro averaging: sum of TP divided by the sum of TP and FN.
  recall = np.sum(TP) / np.sum(TP + FN)
  # Calculate F1 score using micro averaging: harmonic mean of precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using micro averaging: sum of TP and TN divided by TP, TN, FP, and FN.
  accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
  # Calculate specificity using micro averaging: sum of TN divided by the sum of TN and FP.
  specificity = np.sum(TN) / np.sum(TN + FP)

  metrics.update({
    "Micro Precision"  : precision,
    "Micro Recall"     : recall,
    "Micro F1"         : f1,
    "Micro Accuracy"   : accuracy,
    "Micro Specificity": specificity,
  })

  # If requested, calculate the micro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Micro Average": avg,
    })

  # Calculate the number of samples per class by summing the rows of the confusion matrix.
  samples = np.sum(confMatrix, axis=1)

  # Calculate the weights for each class as the proportion of samples in that class.
  weights = samples / np.sum(confMatrix)

  # Calculate precision using weighted averaging: sum of precision per class multiplied by weights.
  precision = np.sum(TP / (TP + FP) * weights)
  # Calculate recall using weighted averaging: sum of recall per class multiplied by weights.
  recall = np.sum(TP / (TP + FN) * weights)
  # Calculate F1 score using weighted averaging: harmonic mean of weighted precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using weighted averaging: sum of TP and TN divided by the total sum of the matrix.
  accuracy = np.sum((TP + TN) * weights) / np.sum(confMatrix)
  # Calculate specificity using weighted averaging: sum of specificity per class multiplied by weights.
  specificity = np.sum(TN / (TN + FP) * weights)

  metrics.update({
    "Weights"             : weights,
    "Weighted Precision"  : precision,
    "Weighted Recall"     : recall,
    "Weighted F1"         : f1,
    "Weighted Accuracy"   : accuracy,
    "Weighted Specificity": specificity,
  })

  # If requested, calculate the weighted average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Weighted Average": avg,
    })

  return metrics


# ===========================================================================================
# Function(s) for machine learning classification.
# ===========================================================================================

def GetScalerObject(scalerName):
  """
  Get the scaler object based on the given name.

  Parameters:
      scalerName (str): Name of the scaler.

  Returns:
      scaler (object): Scaler object.
  """
  if (scalerName == "Standard"):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler()
  elif (scalerName == "MinMax"):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler()
  elif (scalerName == "Robust"):
    from sklearn.preprocessing import RobustScaler
    return RobustScaler()
  elif (scalerName == "MaxAbs"):
    from sklearn.preprocessing import MaxAbsScaler
    return MaxAbsScaler()
  elif (scalerName == "QT"):
    from sklearn.preprocessing import QuantileTransformer
    return QuantileTransformer()
  elif (scalerName == "Normalizer"):
    from sklearn.preprocessing import Normalizer
    return Normalizer()
  # You can add more scalers as needed.
  else:
    raise ValueError("Invalid scaler name.")


def GetMLClassificationModelObject(modelName, hyperparameters={}):
  """
  Get the machine learning classification model object based on the given name.

  Parameters:
      modelName (str): Name of the model.

  Returns:
      model (object): Model object.
  """
  if (modelName == "MLP"):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(**hyperparameters)
  elif (modelName == "RF"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**hyperparameters)
  elif (modelName == "AB"):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(**hyperparameters)
  elif (modelName == "KNN"):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**hyperparameters)
  elif (modelName == "DT"):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(**hyperparameters)
  elif (modelName == "SVC"):
    from sklearn.svm import SVC
    return SVC(**hyperparameters)
  elif (modelName == "GNB"):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**hyperparameters)
  elif (modelName == "LR"):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**hyperparameters)
  elif (modelName == "SGD"):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(**hyperparameters)
  elif (modelName == "GB"):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**hyperparameters)
  elif (modelName == "Bagging"):
    from sklearn.ensemble import BaggingClassifier
    return BaggingClassifier(**hyperparameters)
  elif (modelName == "ETs"):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(**hyperparameters)
  elif (modelName == "XGB"):
    from xgboost import XGBClassifier
    return XGBClassifier(**hyperparameters)
  elif (modelName == "LGBM"):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**hyperparameters)
  elif (modelName == "Voting"):
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(**hyperparameters)
  elif (modelName == "Stacking"):
    from sklearn.ensemble import StackingClassifier
    return StackingClassifier(**hyperparameters)
  # You can add more models as needed.
  else:
    raise ValueError("Invalid model name.")


def MachineLearningClassificationV1(
  datasetFilePath,  # Dataset file name (CSV format).
  scalerName,  # Name of the scaler to use.
  modelName,  # Name of the machine learning classification model.
  testRatio=0.2,  # Ratio of the test data.
  targetColumn="Class",  # Name of the target column in the dataset.
  dropFirstColumn=True,  # Whether to drop the first column (usually an index or ID).
):
  """
  Perform machine learning classification on the given dataset.

  Parameters:
      datasetFilePath (str): Dataset file name (CSV format).
      scalerName (str): Name of the scaler to use.
      modelName (str): Name of the machine learning classification model.
      testRatio (float): Ratio of the test data.
      targetColumn (str): Name of the target column in the dataset.
      dropFirstColumn (bool): Whether to drop the first column (usually an index or ID).

  Returns:
      metrics (dict): Dictionary containing the calculated performance metrics.
  """

  # Read the CSV file into a pandas DataFrame.
  data = pd.read_csv(datasetFilePath)

  # Check if dropFirstColumn is True, then drop the first column.
  if (dropFirstColumn):
    # Drop the first column if it is not the target column.
    if (data.columns[0] != targetColumn):
      data = data.drop(data.columns[0], axis=1)

  # Drop empty columns from the DataFrame.
  data = data.dropna(axis=1, how="all")

  # Drop rows with null or empty values from the DataFrame.
  data = data.dropna()

  # Features (X) are all columns except the "Class" column.
  X = data.drop(targetColumn, axis=1)

  # Target (y) is the "Class" column.
  y = data[targetColumn]

  # Encode the target labels into numerical values using LabelEncoder.
  le = LabelEncoder()
  yEnc = le.fit_transform(y)
  labels = le.classes_

  # Split the data into training and testing sets.
  xTrain, xTest, yTrain, yTest = train_test_split(
    X, yEnc,
    test_size=testRatio,
    random_state=np.random.randint(0, 1000),
    stratify=yEnc,
  )

  # Create a scaler object to scale the features.
  scaler = GetScalerObject(scalerName)

  # Fit the scaler on the training data and transform it.
  xTrain = scaler.fit_transform(xTrain)

  # Transform the test data using the fitted scaler.
  xTest = scaler.transform(xTest)

  # Train a model on the training data.
  model = GetMLClassificationModelObject(modelName)
  model.fit(xTrain, yTrain)

  # Evaluate the model by making predictions on the test data.
  predTest = model.predict(xTest)

  # Calculate the confusion matrix using the true and predicted labels.
  cm = confusion_matrix(yTest, predTest)

  # Calculate performance metrics.
  metrics = CalculatePerformanceMetrics(
    cm,  # Pass the confusion matrix.
    eps=1e-10,  # Small value to avoid division by zero.
    addWeightedAverage=True,  # Whether to include weighted averages in the output.
  )

  # Display the confusion matrix using ConfusionMatrixDisplay.
  disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,  # Pass the confusion matrix.
    display_labels=le.classes_,  # Use the encoded labels for display.
  )
  # Create a plot for the confusion matrix.
  fig, ax = plt.subplots(figsize=(8, 8))
  disp.plot(
    cmap=plt.cm.Blues,  # Set the color map.
    values_format="d",  # Set the format of the values.
    xticks_rotation="horizontal",  # Set the x-axis labels rotation.
    colorbar=True,  # Show the color bar.
    ax=ax,  # Set the axis.
  )
  plt.title("Confusion Matrix", fontsize=16)  # Set the title.
  plt.xlabel("Predicted Label", fontsize=14)  # Set the axis labels.
  plt.ylabel("True Label", fontsize=14)  # Set the axis labels.
  plt.tight_layout()  # Adjust the layout to fit the plot.

  pltObject = plt.gcf()  # Get the current figure object.

  # Create a dictionary to hold the objects for saving.
  objects = {
    "Model"       : model,
    "Scaler"      : scaler,
    "LabelEncoder": le,
  }

  # Return the performance metrics, plot object, and objects for saving.
  return metrics, pltObject, objects


# ===========================================================================================
# Custom Function(s) for Preprocessing Special Tasks.
# ===========================================================================================

def MedMnistDatasetDetails(cls, split="train"):
  """
  Get the details of the MedMNIST dataset for a specific split (train, val, test).

  Args:
    cls (class): The MedMNIST dataset class to use for loading the data.
    split (str): The split of the dataset to load (default is "train").

  Returns:
    dict: A dictionary containing dataset information such as number of samples,
          categories, and other metadata.
  """
  # Load the dataset for the specified split and download if necessary.
  if (not cls):  # Check if cls is None or not provided.
    from medmnist import AdrenalMNIST3D
    dataset = AdrenalMNIST3D(split=split, download=True)
  else:  # If cls is provided, use it to load the dataset.
    dataset = cls(split=split, download=True)

  # Return the dataset information.
  return dataset.info


def MedMnistLoaderIterator(cls, split="train"):
  """
  Load and preprocess the MedMNIST dataset for a specific split (train, val, test).
  This function yields preprocessed volume data, category labels, and indices
  for each record in the dataset.

  Args:
    cls (class): The MedMNIST dataset class to use for loading the data.
    split (str): The split of the dataset to load (default is "train").

  Returns:
    generator: A generator that yields tuples of (volume, category, index).
  """

  # Load the dataset for the specified split and download if necessary.
  if (not cls):  # Check if cls is None or not provided.
    # If cls is None, use the AdrenalMNIST3D dataset.
    from medmnist import AdrenalMNIST3D
    dataset = AdrenalMNIST3D(split=split, download=True)
  else:  # If cls is provided, use it to load the dataset.
    # Load the dataset using the provided class.
    dataset = cls(split=split, download=True)
  # Print dataset information for reference.
  for info in dataset.info:
    print(f"{info}: {dataset.info[info]}")
  # Extract the label dictionary for mapping indices to categories.
  categoriesDict = dataset.info["label"]
  # Iterate through the dataset with a progress bar.
  for i, record in enumerate(dataset):
    # Extract the volume data from the record.
    volume = record[0][0]
    # Normalize the volume to the range [0, 1].
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    # Scale the normalized volume to the range [0, 255].
    volume = volume * 255
    # Convert the volume to an unsigned 8-bit integer format.
    volume = volume.astype(np.uint8)
    # Extract the category index from the record.
    categoryIdx = record[1][0]
    # Map the category index to its corresponding label.
    category = categoriesDict[str(categoryIdx)]
    # Yield the preprocessed volume, category, and index.
    yield volume, category, i


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
