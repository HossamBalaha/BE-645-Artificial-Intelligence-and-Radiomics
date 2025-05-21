'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 21st, 2025
# Last Modification Date: May 21st, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


def FirstOrderFeatures2D(img, mask, isNorm=True, ignoreZeros=True):
  """
  Calculate first-order statistical features from an image using a mask.

  Args:
      img (numpy.ndarray): The input image as a 2D NumPy array.
      mask (numpy.ndarray): The binary mask as a 2D NumPy array.
      isNorm (bool): Flag to indicate whether to normalize the histogram.

  Returns:
      results (dict): A dictionary containing the calculated first-order features.
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
  if (ignoreZeros):
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
